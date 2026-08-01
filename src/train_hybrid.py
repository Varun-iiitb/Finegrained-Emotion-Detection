"""Hybrid training (DG-DQA): pretrain on the MER-Caption+ pool, then fine-tune
on human OV-MERD via k-fold CV.

Motivation (see CLAUDE.md §8/§9, 2026-06-20): training a discriminative model on
the MER-Caption+ AUTO openset labels caps OV-MERD F_s at ~27 (auto-vs-human label
agreement). The features are fine (human-only linear probe ≈ 41.7). So:

  Stage 1 (pretrain, on the 30K pool): learn the representation with
    multi-positive InfoNCE over canonical emotion-word anchors
    + DQF multi-granularity aux
    + DG description-grounding (symmetric InfoNCE between the fused embedding and
      the clip's BGE description embedding `desc_emb` — the "DG" in DG-DQA).
  Stage 2 (fine-tune, k-fold CV on the human set): adapt to the human label
    distribution; report human-CV F_s mean±std. Each fold = fit/val/test split;
    threshold + best checkpoint chosen on the fold's val, scored on its test.

The pool excludes the human eval clips (548 overlap already removed in data_prep),
so pretraining never sees eval clips. F_s = case2 (synonym), drop OOV.

Run:  python src/train_hybrid.py --pre-epochs 15 --ft-epochs 40 --folds 5
"""

import argparse
import copy
import glob
import json
import os
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DQFBody                                   # noqa: E402
from dataset import CacheDataset, collate                    # noqa: E402
from metric import WheelGrouper, score_group_sets            # noqa: E402
from train import (AlignHead, set_seed, build_anchors, multipos_infonce,      # noqa: E402
                   pos_mask, to_device, collect_sims, fs_at, best_threshold)


def desc_contrastive(z, desc, tau):
    """Symmetric in-batch InfoNCE between fused embeddings z and their own
    description embeddings desc (both L2-normed, BGE space)."""
    logits = (z @ desc.t()) / tau                # [B,B]
    labels = torch.arange(z.size(0), device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def make_model(args, dev):
    model = DQFBody(1024, 768, 768, d_model=args.d_model, n_heads=args.heads,
                    n_layers=args.layers).to(dev)
    head = AlignHead(args.d_model).to(dev)
    return model, head


def train_epoch(model, head, loader, anchors, word2idx, V, args, dev, use_desc):
    model.train(); head.train()
    opt = train_epoch.opt
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
        if use_desc and "desc_emb" in b:
            desc = F.normalize(b["desc_emb"].to(dev), dim=-1)
            loss = loss + args.desc_weight * desc_contrastive(z, desc, args.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def pretrain(args, anchors, word2idx, V, grouper, dev, grid, human_dl):
    set_seed(args.seed)
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(args.cache_dir, "train", "*.pt")))
    if args.limit:
        names = names[:args.limit]
    dl = DataLoader(CacheDataset(args.cache_dir, "train", args.text, names),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate)
    model, head = make_model(args, dev)
    train_epoch.opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=1e-4)
    print(f"[pretrain] {len(names)} pool clips, DG desc-grounding ON "
          f"(w={args.desc_weight})")
    for ep in range(args.pre_epochs):
        loss = train_epoch(model, head, dl, anchors, word2idx, V, args, dev, use_desc=True)
        if ep % 3 == 0 or ep == args.pre_epochs - 1:
            sims, golds = collect_sims(model, head, human_dl, anchors, dev)
            gsets = [grouper.to_group_set(g, "case2", "drop") for g in golds]
            zs = max(fs_at(sims, gsets, args.canon, grouper, t) for t in grid) * 100
            print(f"  [pretrain] ep{ep:02d} loss={loss:.3f} "
                  f"human zero-shot F_s(oracle-thr)={zs:.2f}", flush=True)
    return copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())


def finetune_cv(args, pre_state, anchors, word2idx, V, grouper, dev, grid):
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(args.cache_dir, "eval", "*.pt")))
    import random
    random.Random(args.seed).shuffle(names)
    folds = [list(f) for f in np.array_split(names, args.folds)]
    fold_fs = []
    for k in range(args.folds):
        te = folds[k]
        rest = [n for j, f in enumerate(folds) if j != k for n in f]
        nv = max(1, len(rest) // 5)
        val, fit = rest[:nv], rest[nv:]
        mk = lambda nm, sh: DataLoader(
            CacheDataset(args.cache_dir, "eval", args.text, nm),
            batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
            collate_fn=collate)
        fit_dl, val_dl, te_dl = mk(fit, True), mk(val, False), mk(te, False)

        model, head = make_model(args, dev)
        model.load_state_dict(pre_state[0]); head.load_state_dict(pre_state[1])
        if args.finetune == "head":
            for p in model.parameters():
                p.requires_grad = False
        params = [p for p in list(model.parameters()) + list(head.parameters())
                  if p.requires_grad]
        train_epoch.opt = torch.optim.AdamW(params, lr=args.ft_lr, weight_decay=1e-4)

        best = {"vfs": -1.0, "thr": grid[0], "state": None}
        for ep in range(args.ft_epochs):
            train_epoch(model, head, fit_dl, anchors, word2idx, V, args, dev, use_desc=False)
            sims, golds = collect_sims(model, head, val_dl, anchors, dev)
            vfs, vthr = best_threshold(sims, golds, args.canon, grouper, grid)
            if vfs > best["vfs"]:
                best = {"vfs": vfs, "thr": vthr,
                        "state": (copy.deepcopy(model.state_dict()),
                                  copy.deepcopy(head.state_dict()))}
        model.load_state_dict(best["state"][0]); head.load_state_dict(best["state"][1])
        sims, golds = collect_sims(model, head, te_dl, anchors, dev)
        gold_sets = [grouper.to_group_set(g, "case2", "drop") for g in golds]
        fs = fs_at(sims, gold_sets, args.canon, grouper, best["thr"]) * 100
        fold_fs.append(fs)
        print(f"  [finetune] fold{k}: human-test F_s={fs:.2f} "
              f"(val={best['vfs']*100:.2f} thr={best['thr']:.3f})", flush=True)
    return fold_fs


def main():
    ap = argparse.ArgumentParser(description="DG-DQA hybrid pretrain + human CV finetune")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/SRIP/data/cache"))
    ap.add_argument("--vocab", default=os.path.expanduser("~/SRIP/data/manifests/vocab.json"))
    ap.add_argument("--wheel", default=os.path.expanduser(
        "~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel"))
    ap.add_argument("--text", choices=["zh", "en"], default="zh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--finetune", choices=["head", "full"], default="full")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--desc-weight", type=float, default=0.5)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anchor-template", default="a feeling of {word}")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = json.load(open(args.vocab, encoding="utf-8"))["words"]
    grouper = WheelGrouper(args.wheel)
    word2canon = {}
    for w in vocab:
        gs = grouper.to_group_set([w], "case2", "drop")
        if gs:
            word2canon[w] = next(iter(gs))
    canon = sorted(set(word2canon.values()))
    cidx = {c: i for i, c in enumerate(canon)}
    word2idx = {w: cidx[word2canon[w]] for w in word2canon}
    args.canon = canon
    V = len(canon)
    anchors = build_anchors(canon, args.anchor_template, dev,
                            os.path.join(args.cache_dir, "anchors_canonical.pt"))
    grid = [round(x, 4) for x in np.arange(0.05, 0.55, 0.025)]

    # human loader for pretrain zero-shot monitoring
    human_dl = DataLoader(CacheDataset(args.cache_dir, "eval", args.text),
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=collate)

    print(f"\n=== HYBRID: {V} canonical anchors | pretrain {args.pre_epochs}ep "
          f"-> finetune({args.finetune}) {args.folds}-fold {args.ft_epochs}ep ===")
    pre_state = pretrain(args, anchors, word2idx, V, grouper, dev, grid, human_dl)
    fold_fs = finetune_cv(args, pre_state, anchors, word2idx, V, grouper, dev, grid)
    mean, std = statistics.mean(fold_fs), statistics.pstdev(fold_fs)
    print(f"\n=== RESULT: human-CV OV-MERD F_s = {mean:.2f} ± {std:.2f} "
          f"over {args.folds} folds {[round(x,1) for x in fold_fs]} ===")
    print(f"  reference: gate (auto-trained) ~28 | human linear-probe ~41.7 | gate bar ~50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
