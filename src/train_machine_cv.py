"""Machine-data (MER-Caption+) WITHIN-DISTRIBUTION training: train/val/test split
AND k-fold CV, with the full DQF fusion body + open-vocab AlignHead, plus
per-epoch history saved to disk for training plots.

WHY (advisor directive, 2026-06-25): the old protocol "train on machine auto
labels / eval on human OV-MERD" is *cross-data* validation — the auto and human
label distributions disagree (F_s agreement ≈ 27), which hard-caps OV-MERD F_s at
~28 no matter the model (see CLAUDE.md §8/§9). The advisor does NOT want that.
Instead: evaluate WITHIN the machine-annotated distribution, the same way we did
5-fold CV on the human set — train on machine folds, test on a held-out machine
fold. Labels are now self-consistent, and there is plenty of data (30,779 clips),
so we use the heavy DQF approach (no small-data overfit concern).

INTERPRETATION (state this when reporting): this measures "can DQF predict the
auto-annotator's openset labels" (within-distribution generalization), NOT "can it
predict human emotions" (that remains the human-CV number, ≈44.6). Report both.

Two modes (both select the checkpoint + threshold on F_s, never on loss):
  --mode split : one fixed train/val/test split (default 80/10/10). Best for a
                 clean single training curve. Headline = test F_s at the val-
                 selected threshold + best-by-val checkpoint.
  --mode cv    : k-fold CV. Fold k is the test set; the other folds are split into
                 a val slice (1/5 of the rest) + the fit set. Report mean±std test
                 F_s over folds. Per-fold history saved for overlay plots.

Outputs (under --run-dir/<run-name>/):
  history.json : list of per-epoch records {fold, epoch, train_loss, val_fs, val_thr}
  result.json  : final metrics (test F_s per fold + mean/std, args, label space)
  → feed to  python src/plot_training.py --run <run-dir>/<run-name>

Run (smoke):  python src/train_machine_cv.py --mode split --epochs 3 --limit 2000
Run (split):  python src/train_machine_cv.py --mode split --epochs 20
Run (5-fold): python src/train_machine_cv.py --mode cv --folds 5 --epochs 20
"""

import argparse
import copy
import glob
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DQFBody                                       # noqa: E402
from dataset import CacheDataset, collate                        # noqa: E402
from metric import WheelGrouper, score_group_sets                # noqa: E402
from train import (AlignHead, set_seed, build_anchors, multipos_infonce,   # noqa: E402
                   pos_mask, to_device, collect_sims, fs_at, best_threshold)
from train_hybrid import desc_contrastive                        # noqa: E402


# --------------------------------------------------------------------------- #
#  one training run (shared by split mode and each CV fold)                    #
# --------------------------------------------------------------------------- #

def make_model(args, dev):
    model = DQFBody(1024, 768, 768, d_model=args.d_model, n_heads=args.heads,
                    n_layers=args.layers).to(dev)
    head = AlignHead(args.d_model).to(dev)
    return model, head


def train_one_epoch(model, head, loader, opt, anchors, word2idx, V, args, dev):
    """Multi-positive InfoNCE over emotion-word anchors + DQF multi-granularity
    aux (+ optional DG description-grounding). Returns mean batch loss."""
    model.train(); head.train()
    tot, nb = 0.0, 0
    for b in loader:
        feats, pads = to_device(b, dev)
        mm = DQFBody.random_modality_mask(len(b["name"]), dev, args.p_drop)
        gran = model(feats, pads, modality_mask=mm)
        pm = pos_mask(b["label_words"], word2idx, V, dev)
        valid = pm.sum(1) > 0
        if valid.sum() == 0:
            continue
        z = head(gran["tav"])
        loss = multipos_infonce(z, anchors, pm, args.tau)[valid].mean()
        aux = sum(multipos_infonce(head(gran[g]), anchors, pm, args.tau)[valid].mean()
                  for g in ("t", "a", "v", "ta", "tv", "av"))
        loss = loss + args.aux_weight * (aux / 6.0)
        if args.desc_weight > 0 and "desc_emb" in b:
            desc = F.normalize(b["desc_emb"].to(dev), dim=-1)
            loss = loss + args.desc_weight * desc_contrastive(z, desc, args.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def run_one(fit_dl, val_dl, te_dl, args, anchor_vocab, word2idx, anchors,
            grouper, dev, grid, fold: int, history: List[Dict]) -> Dict:
    """Train DQF+head from scratch, select on val F_s, score test. Appends one
    record per epoch to `history`. Returns the fold result dict."""
    V = len(anchor_vocab)
    model, head = make_model(args, dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    best = {"vfs": -1.0, "thr": grid[0], "epoch": -1, "state": None}
    for ep in range(args.epochs):
        loss = train_one_epoch(model, head, fit_dl, opt, anchors, word2idx, V, args, dev)
        sims, golds = collect_sims(model, head, val_dl, anchors, dev)
        vfs, vthr = best_threshold(sims, golds, anchor_vocab, grouper, grid)
        history.append({"fold": fold, "epoch": ep, "train_loss": round(loss, 5),
                        "val_fs": round(vfs * 100, 4), "val_thr": round(vthr, 4)})
        print(f"  fold{fold} ep{ep:02d} loss={loss:.3f} val_Fs={vfs*100:.2f} "
              f"thr={vthr:.3f}", flush=True)
        if vfs > best["vfs"]:
            best = {"vfs": vfs, "thr": vthr, "epoch": ep,
                    "state": (copy.deepcopy(model.state_dict()),
                              copy.deepcopy(head.state_dict()))}

    # final test score at the val-selected checkpoint + threshold
    model.load_state_dict(best["state"][0]); head.load_state_dict(best["state"][1])
    sims, golds = collect_sims(model, head, te_dl, anchors, dev)
    gold_sets = [grouper.to_group_set(g, "case2", "drop") for g in golds]
    test_fs = fs_at(sims, gold_sets, anchor_vocab, grouper, best["thr"]) * 100
    print(f"  fold{fold}: TEST F_s={test_fs:.2f}  (best_val={best['vfs']*100:.2f} "
          f"@ep{best['epoch']} thr={best['thr']:.3f})", flush=True)
    return {"fold": fold, "test_fs": round(test_fs, 4),
            "best_val_fs": round(best["vfs"] * 100, 4),
            "best_epoch": best["epoch"], "best_thr": round(best["thr"], 4)}


# --------------------------------------------------------------------------- #
#  data splitting                                                             #
# --------------------------------------------------------------------------- #

def all_train_names(cache_dir: str, limit: Optional[int], seed: int) -> List[str]:
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(cache_dir, "train", "*.pt")))
    if limit:
        names = names[:limit]
    random.Random(seed).shuffle(names)
    return names


def split_three(names: List[str], fracs: Tuple[float, float, float]
                ) -> Tuple[List[str], List[str], List[str]]:
    n = len(names)
    ntr = int(n * fracs[0]); nva = int(n * fracs[1])
    return names[:ntr], names[ntr:ntr + nva], names[ntr + nva:]


def make_loader(args, names, shuffle):
    return DataLoader(CacheDataset(args.cache_dir, "train", args.text, names),
                      batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, collate_fn=collate)


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #

def build_anchor_space(vocab, grouper, args, dev, cache_dir):
    """Return (anchor_vocab, word2idx, anchors). word2idx maps each gold word to
    its anchor index (identity for word-level; collapsed for canonical-level)."""
    if args.anchor_level == "canonical":
        word2canon = {}
        for w in vocab:
            gs = grouper.to_group_set([w], "case2", "drop")
            if gs:
                word2canon[w] = next(iter(gs))
        anchor_vocab = sorted(set(word2canon.values()))
        cidx = {c: i for i, c in enumerate(anchor_vocab)}
        word2idx = {w: cidx[word2canon[w]] for w in word2canon}
        print(f"[anchors] canonical: {len(vocab)} words -> {len(anchor_vocab)} anchors")
    else:
        anchor_vocab = vocab
        word2idx = {w: i for i, w in enumerate(vocab)}
        print(f"[anchors] word level: {len(vocab)} anchors")
    anchors = build_anchors(anchor_vocab, args.anchor_template, dev,
                            os.path.join(cache_dir, f"anchors_{args.anchor_level}.pt"))
    return anchor_vocab, word2idx, anchors


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Machine-data within-distribution DQF training (split + CV) with history")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/SRIP/data/cache"))
    ap.add_argument("--vocab", default=os.path.expanduser("~/SRIP/data/manifests/vocab.json"))
    ap.add_argument("--wheel", default=os.path.expanduser(
        "~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel"))
    ap.add_argument("--run-dir", default=os.path.expanduser("~/SRIP/runs"))
    ap.add_argument("--run-name", default=None, help="defaults to mode+text+level")
    ap.add_argument("--mode", choices=["split", "cv"], default="split")
    ap.add_argument("--folds", type=int, default=5, help="cv mode only")
    ap.add_argument("--train-frac", type=float, default=0.8, help="split mode")
    ap.add_argument("--val-frac", type=float, default=0.1, help="split mode")
    ap.add_argument("--text", choices=["zh", "en"], default="zh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--desc-weight", type=float, default=0.0,
                    help="DG description-grounding weight (0=off; auto descs)")
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anchor-template", default="a feeling of {word}")
    ap.add_argument("--anchor-level", choices=["word", "canonical"], default="canonical")
    ap.add_argument("--label-map", default=None,
                    help="curated repaired_label_map.json: use it for BOTH gold "
                         "grouping and the anchor set (193 canonicals incl. new ones)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N clips")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    tag = "_curated" if args.label_map else ""
    run_name = args.run_name or f"machine_{args.mode}_{args.text}_{args.anchor_level}{tag}"
    out_dir = os.path.join(args.run_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    vocab = json.load(open(args.vocab, encoding="utf-8"))["words"]
    if args.label_map:
        # curated map drives gold grouping AND the anchor space (so the model can
        # predict humor/neutral/serious, not just be penalized on them)
        from relabel_map import load_repaired
        rg = load_repaired(args.label_map)
        cmap = dict(rg.m)
        anchor_vocab = sorted(set(cmap.values()))
        for c in anchor_vocab:
            cmap.setdefault(c, c)               # predicted canonical names self-map
        cidx = {c: i for i, c in enumerate(anchor_vocab)}
        word2idx = {w: cidx[cmap[w]] for w in cmap}
        from relabel_map import RepairedGrouper
        grouper = RepairedGrouper(cmap)
        anchors = build_anchors(anchor_vocab, args.anchor_template, dev,
                                os.path.join(args.cache_dir, "anchors_curated.pt"))
        print(f"[label-map] curated: {len(anchor_vocab)} canonicals "
              f"(incl new), {len(cmap)} word entries")
    else:
        grouper = WheelGrouper(args.wheel)
        anchor_vocab, word2idx, anchors = build_anchor_space(
            vocab, grouper, args, dev, args.cache_dir)
    grid = [round(x, 4) for x in np.arange(0.05, 0.55, 0.025)]

    names = all_train_names(args.cache_dir, args.limit, args.seed)
    history: List[Dict] = []
    fold_results: List[Dict] = []

    print(f"\n=== MACHINE-CV {args.mode.upper()} | run={run_name} | text={args.text} "
          f"| {len(anchor_vocab)} anchors | n={len(names)} clips ===", flush=True)

    if args.mode == "split":
        fit, val, te = split_three(names, (args.train_frac, args.val_frac,
                                           1.0 - args.train_frac - args.val_frac))
        print(f"  split: fit={len(fit)} val={len(val)} test={len(te)}", flush=True)
        res = run_one(make_loader(args, fit, True), make_loader(args, val, False),
                      make_loader(args, te, False), args, anchor_vocab, word2idx,
                      anchors, grouper, dev, grid, fold=0, history=history)
        fold_results.append(res)
    else:  # cv
        folds = [list(f) for f in np.array_split(names, args.folds)]
        for k in range(args.folds):
            te = folds[k]
            rest = [n for j, f in enumerate(folds) if j != k for n in f]
            nv = max(1, len(rest) // 5)
            val, fit = rest[:nv], rest[nv:]
            print(f"  fold{k}: fit={len(fit)} val={len(val)} test={len(te)}", flush=True)
            res = run_one(make_loader(args, fit, True), make_loader(args, val, False),
                          make_loader(args, te, False), args, anchor_vocab, word2idx,
                          anchors, grouper, dev, grid, fold=k, history=history)
            fold_results.append(res)

    test_scores = [r["test_fs"] for r in fold_results]
    mean = statistics.mean(test_scores)
    std = statistics.pstdev(test_scores) if len(test_scores) > 1 else 0.0
    result = {"run_name": run_name, "mode": args.mode, "text": args.text,
              "anchor_level": args.anchor_level, "n_clips": len(names),
              "n_anchors": len(anchor_vocab), "epochs": args.epochs,
              "test_fs_mean": round(mean, 4), "test_fs_std": round(std, 4),
              "folds": fold_results, "args": vars(args)}
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== RESULT {run_name}: machine within-dist TEST F_s = "
          f"{mean:.2f} ± {std:.2f} over {len(test_scores)} "
          f"{'fold(s)' if args.mode=='cv' else 'split'} "
          f"{[round(s,1) for s in test_scores]} ===")
    print(f"  saved -> {out_dir}/history.json + result.json")
    print(f"  NOTE: this is WITHIN-DISTRIBUTION (predict auto labels). "
          f"Human-relevance number is still the human-CV (~44.6).")
    print(f"  plot:  python src/plot_training.py --run {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
