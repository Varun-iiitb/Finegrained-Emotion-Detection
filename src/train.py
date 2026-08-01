"""Open-vocab Option-B head + training + F_s eval (DG-DQA, Prompt 6 GATE).

Adds the alignment head on top of the DQF body and trains it (encoders are
frozen and already cached — only DQFBody + head train). Open-vocabulary by
construction: the fused [CLS] is projected to the BGE text-emotion space and
cosine-matched against frozen emotion-word ANCHORS.

Loss = multi-positive InfoNCE between the fused embedding and the word anchors
(each clip's gold-label words are positives, all other vocab words negatives),
plus the DQF multi-granularity auxiliary losses on {t,a,v,ta,tv,av}. Random
per-modality masking is applied during training.

Selection / thresholding: a held-out slice of the TRAIN pool (MER-Caption+) is
used to tune the emission threshold and pick the best checkpoint by F_s. The
clean OV-MERD set (eval/) is ONLY used for the final reported F_s — never for
selection (see CLAUDE.md §6/§10). F_s = case2 (synonym level), drop OOV.

GATE: report OV-MERD F_s mean±std over >=3 seeds; must clear ~50 F_s, else stop
and fix features before adding DG description-grounding complexity.

Run (quick smoke):  python src/train.py --seeds "0" --epochs 3 --limit 2000
Run (gate, zh):     python src/train.py --text zh --seeds "0 1 2" --epochs 20
Run (gate, en):     python src/train.py --text en --seeds "0 1 2" --epochs 20
"""

import argparse
import copy
import glob
import json
import os
import random
import statistics
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DQFBody                       # noqa: E402
from dataset import CacheDataset, collate        # noqa: E402
from metric import WheelGrouper, score_group_sets  # noqa: E402


class AlignHead(nn.Module):
    """Project the fused [CLS] -> L2-normalized embedding in the BGE space."""

    def __init__(self, d_in: int, d_out: int = 1024, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_anchors(vocab: List[str], template: str, device: str,
                  cache_path: str) -> torch.Tensor:
    """Frozen emotion-word anchors: encode each templated vocab word with the
    SAME text encoder (BGE-M3), L2-normalized. Cached + reused if template matches."""
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if d.get("template") == template and d.get("vocab") == vocab:
            print(f"[anchors] loaded {cache_path} {tuple(d['anchors'].shape)}")
            return d["anchors"].to(device)
    print(f"[anchors] building from BGE-M3, template={template!r} …")
    from extract_features import TextEncoder
    te = TextEncoder(device)
    emb = te.encode_pooled([template.format(word=w) for w in vocab], max_len=32)
    torch.save({"vocab": vocab, "anchors": emb, "template": template}, cache_path)
    del te
    torch.cuda.empty_cache()
    print(f"[anchors] built {tuple(emb.shape)} -> {cache_path}")
    return emb.to(device)


def multipos_infonce(z, anchors, pos, tau):
    """Multi-positive InfoNCE: -mean over positives of log-softmax sim."""
    logp = F.log_softmax((z @ anchors.t()) / tau, dim=1)      # [B,V]
    return -(logp * pos).sum(1) / pos.sum(1).clamp_min(1.0)    # [B]


def pos_mask(label_words, word2idx, V, device):
    pm = torch.zeros(len(label_words), V, device=device)
    for i, words in enumerate(label_words):
        for w in words:
            j = word2idx.get(w)
            if j is not None:
                pm[i, j] = 1.0
    return pm


def to_device(b, dev):
    feats = {m: b[m].to(dev) for m in "tav"}
    pads = {m: b[m + "_pad"].to(dev) for m in "tav"}
    return feats, pads


@torch.no_grad()
def collect_sims(model, head, loader, anchors, dev):
    model.eval(); head.eval()
    sims, golds = [], []
    for b in loader:
        feats, pads = to_device(b, dev)
        z = head(model(feats, pads)["tav"])
        sims.append((z @ anchors.t()).cpu())
        golds += b["label_words"]
    return torch.cat(sims, 0), golds


def _pred_words(sim_row, vocab, thr):
    idx = (sim_row > thr).nonzero().flatten()
    if idx.numel() == 0:                      # never emit empty -> top-1
        idx = sim_row.argmax().view(1)
    return [vocab[j] for j in idx.tolist()]


def fs_at(sims, gold_sets, vocab, grouper, thr):
    pred_sets = [grouper.to_group_set(_pred_words(sims[i], vocab, thr), "case2", "drop")
                 for i in range(sims.size(0))]
    return score_group_sets(gold_sets, pred_sets)["f_s"]


def best_threshold(sims, golds, vocab, grouper, grid):
    gold_sets = [grouper.to_group_set(g, "case2", "drop") for g in golds]
    best_fs, best_thr = -1.0, grid[0]
    for thr in grid:
        fs = fs_at(sims, gold_sets, vocab, grouper, thr)
        if fs > best_fs:
            best_fs, best_thr = fs, thr
    return best_fs, best_thr


def run_seed(seed, args, anchor_vocab, label2idx, anchors, grouper, dev, grid):
    vocab, word2idx = anchor_vocab, label2idx   # anchors + gold->anchor-index map
    set_seed(seed)
    V = len(vocab)
    names = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(args.cache_dir, "train", "*.pt"))]
    if args.limit:
        names = sorted(names)[:args.limit]
    random.Random(seed).shuffle(names)
    n_val = max(1, int(len(names) * args.val_frac))
    val_names, fit_names = names[:n_val], names[n_val:]

    mk = lambda split, nm, sh: DataLoader(
        CacheDataset(args.cache_dir, split, args.text, nm),
        batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
        collate_fn=collate)
    fit_dl = mk("train", fit_names, True)
    val_dl = mk("train", val_names, False)
    ov_dl = mk("eval", None, False)

    model = DQFBody(1024, 768, 768, d_model=args.d_model, n_heads=args.heads,
                    n_layers=args.layers).to(dev)
    head = AlignHead(args.d_model).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    best = {"vfs": -1.0, "thr": grid[0], "state": None}
    for ep in range(args.epochs):
        model.train(); head.train()
        tot, nb = 0.0, 0
        for b in fit_dl:
            feats, pads = to_device(b, dev)
            mm = DQFBody.random_modality_mask(len(b["name"]), dev, args.p_drop)
            gran = model(feats, pads, modality_mask=mm)
            pm = pos_mask(b["label_words"], word2idx, V, dev)
            valid = pm.sum(1) > 0
            if valid.sum() == 0:
                continue
            loss = multipos_infonce(head(gran["tav"]), anchors, pm, args.tau)[valid].mean()
            aux = sum(multipos_infonce(head(gran[g]), anchors, pm, args.tau)[valid].mean()
                      for g in ("t", "a", "v", "ta", "tv", "av"))
            loss = loss + args.aux_weight * (aux / 6.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sims, golds = collect_sims(model, head, val_dl, anchors, dev)
        vfs, vthr = best_threshold(sims, golds, vocab, grouper, grid)
        print(f"  seed{seed} ep{ep:02d} loss={tot/max(nb,1):.3f} "
              f"val_Fs={vfs*100:.2f} thr={vthr:.3f}", flush=True)
        if vfs > best["vfs"]:
            best = {"vfs": vfs, "thr": vthr,
                    "state": (copy.deepcopy(model.state_dict()),
                              copy.deepcopy(head.state_dict()))}

    model.load_state_dict(best["state"][0]); head.load_state_dict(best["state"][1])
    sims, golds = collect_sims(model, head, ov_dl, anchors, dev)
    gold_sets = [grouper.to_group_set(g, "case2", "drop") for g in golds]
    ov_fs = fs_at(sims, gold_sets, vocab, grouper, best["thr"]) * 100
    print(f"  seed{seed}: OV-MERD F_s={ov_fs:.2f}  "
          f"(val_Fs={best['vfs']*100:.2f} thr={best['thr']:.3f})", flush=True)
    return ov_fs


def main() -> int:
    ap = argparse.ArgumentParser(description="DG-DQA Option-B head + train + F_s GATE")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/SRIP/data/cache"))
    ap.add_argument("--vocab", default=os.path.expanduser("~/SRIP/data/manifests/vocab.json"))
    ap.add_argument("--wheel", default=os.path.expanduser(
        "~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel"))
    ap.add_argument("--text", choices=["zh", "en"], default="zh")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anchor-template", default="a feeling of {word}")
    ap.add_argument("--anchor-level", choices=["word", "canonical"], default="word",
                    help="'word'=one anchor per vocab word (Option-B as specified); "
                         "'canonical'=collapse to wheel case2 canonicals (aligns the "
                         "contrastive target with how F_s scores)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N train clips")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = json.load(open(args.vocab, encoding="utf-8"))["words"]
    grouper = WheelGrouper(args.wheel)

    # anchor vocabulary + gold-word -> anchor-index map (word or canonical level)
    if args.anchor_level == "canonical":
        word2canon = {}
        for w in vocab:
            gs = grouper.to_group_set([w], "case2", "drop")
            if gs:
                word2canon[w] = next(iter(gs))
        anchor_vocab = sorted(set(word2canon.values()))
        cidx = {c: i for i, c in enumerate(anchor_vocab)}
        label2idx = {w: cidx[word2canon[w]] for w in word2canon}
        print(f"[anchors] canonical level: {len(vocab)} words -> "
              f"{len(anchor_vocab)} canonical anchors")
    else:
        anchor_vocab = vocab
        label2idx = {w: i for i, w in enumerate(vocab)}
    anchors = build_anchors(anchor_vocab, args.anchor_template, dev,
                            os.path.join(args.cache_dir, f"anchors_{args.anchor_level}.pt"))
    grid = [round(x, 4) for x in np.arange(0.05, 0.55, 0.025)]
    seeds = [int(s) for s in args.seeds.split()]

    print(f"\n=== GATE run: text={args.text} level={args.anchor_level} "
          f"seeds={seeds} epochs={args.epochs} ===")
    results = [run_seed(s, args, anchor_vocab, label2idx, anchors, grouper, dev, grid)
               for s in seeds]
    mean = statistics.mean(results)
    std = statistics.pstdev(results) if len(results) > 1 else 0.0
    print(f"\n=== RESULT text={args.text}: OV-MERD F_s = {mean:.2f} ± {std:.2f} "
          f"over {len(seeds)} seeds {[round(r,2) for r in results]} ===")
    print(f"GATE (~50 F_s): {'PASS' if mean >= 50.0 else 'FAIL'} (mean {mean:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
