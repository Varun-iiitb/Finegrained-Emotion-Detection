"""Linear probe on FROZEN mean-pooled features (DG-DQA gate diagnostic).

Measures the discriminative-transfer ceiling of each modality, decoupled from
the DQF fusion: mean-pool each cached modality sequence -> fit a linear
multi-label classifier over the wheel-canonical labels -> report OV-MERD F_s.
Threshold tuned on a held-out TRAIN slice (never on OV-MERD).

If even the all-modality probe is far below ~50, the bottleneck is the FEATURES;
if it is near/above ~50, the features are fine and the DQF body/head is the issue.

Run:  python src/linear_probe.py            # all configs: t, a, v, tav(concat)
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import WheelGrouper, score_group_sets  # noqa: E402

DIMS = {"t": ("X_text_zh", 1024), "a": ("X_audio", 768), "v": ("X_visual", 768)}


def load_pooled(cache_dir, split, names=None, cache=True):
    """Mean-pool each modality per clip. Caches the pooled tensors for reuse."""
    cpath = os.path.join(cache_dir, f"pooled_{split}.pt")
    if cache and names is None and os.path.exists(cpath):
        d = torch.load(cpath, weights_only=False)
        return d["t"], d["a"], d["v"], d["labels"]
    d = os.path.join(cache_dir, split)
    files = ([os.path.join(d, n + ".pt") for n in names] if names
             else sorted(glob.glob(os.path.join(d, "*.pt"))))
    T, A, V, L = [], [], [], []
    for f in files:
        r = torch.load(f, weights_only=False)
        T.append(r["X_text_zh"].float().mean(0))
        A.append(r["X_audio"].float().mean(0))
        V.append(r["X_visual"].float().mean(0))
        L.append(r["label_words"])
    T, A, V = torch.stack(T), torch.stack(A), torch.stack(V)
    if cache and names is None:
        torch.save({"t": T, "a": A, "v": V, "labels": L}, cpath)
    return T, A, V, L


def multihot(label_lists, word2idx, C):
    Y = torch.zeros(len(label_lists), C)
    for i, words in enumerate(label_lists):
        for w in words:
            j = word2idx.get(w)
            if j is not None:
                Y[i, j] = 1.0
    return Y


def fs_at(prob, gold_sets, canon, grouper, thr):
    pred = []
    for i in range(prob.size(0)):
        idx = (prob[i] > thr).nonzero().flatten()
        if idx.numel() == 0:
            idx = prob[i].argmax().view(1)
        pred.append(grouper.to_group_set([canon[j] for j in idx.tolist()], "case2", "drop"))
    return score_group_sets(gold_sets, pred)["f_s"]


def run_config(name, Xtr, Ytr, Xva, gold_va, Xov, gold_ov, canon, grouper, grid, dev, epochs):
    C = Ytr.size(1)
    head = nn.Linear(Xtr.size(1), C).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(Xtr.size(0), device=dev)
        for i in range(0, Xtr.size(0), bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(head(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pva = torch.sigmoid(head(Xva.to(dev))).cpu()
        pov = torch.sigmoid(head(Xov.to(dev))).cpu()
    best = (-1.0, grid[0])
    for thr in grid:
        fs = fs_at(pva, gold_va, canon, grouper, thr)
        if fs > best[0]:
            best = (fs, thr)
    ov = fs_at(pov, gold_ov, canon, grouper, best[1]) * 100
    ov_oracle = max(fs_at(pov, gold_ov, canon, grouper, t) for t in grid) * 100
    print(f"  [{name:10s}] OV-MERD F_s={ov:5.2f}  (val_Fs={best[0]*100:.2f} thr={best[1]:.3f}"
          f" | OV oracle-thr={ov_oracle:.2f})")
    return ov


def main():
    ap = argparse.ArgumentParser(description="Linear probe feature-ceiling diagnostic")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/SRIP/data/cache"))
    ap.add_argument("--vocab", default=os.path.expanduser("~/SRIP/data/manifests/vocab.json"))
    ap.add_argument("--wheel", default=os.path.expanduser(
        "~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); torch.manual_seed(args.seed)

    grouper = WheelGrouper(args.wheel)
    vocab = json.load(open(args.vocab, encoding="utf-8"))["words"]
    word2canon = {}
    for w in vocab:
        gs = grouper.to_group_set([w], "case2", "drop")
        if gs:
            word2canon[w] = next(iter(gs))
    canon = sorted(set(word2canon.values()))
    cidx = {c: i for i, c in enumerate(canon)}
    label2idx = {w: cidx[word2canon[w]] for w in word2canon}
    C = len(canon)
    print(f"canonical labels C={C}")

    print("loading + pooling features (cached after first run)…")
    Tt, At, Vt, Lt = load_pooled(args.cache_dir, "train")
    To, Ao, Vo, Lo = load_pooled(args.cache_dir, "eval")
    Ytr_full = multihot(Lt, label2idx, C)
    gold_ov = [grouper.to_group_set(g, "case2", "drop") for g in Lo]

    n = Tt.size(0)
    idx = list(range(n)); random.Random(args.seed).shuffle(idx)
    nv = max(1, int(n * args.val_frac))
    vi, fi = idx[:nv], idx[nv:]
    gold_va = [grouper.to_group_set(Lt[i], "case2", "drop") for i in vi]

    grid = [round(x, 4) for x in np.arange(0.1, 0.85, 0.025)]
    feats = {"t": (Tt, To), "a": (At, Ao), "v": (Vt, Vo)}

    print(f"\n=== LINEAR PROBE (feature ceiling), C={C}, epochs={args.epochs} ===")
    for name, mods in [("text", ["t"]), ("audio", ["a"]), ("visual", ["v"]),
                       ("t+a+v", ["t", "a", "v"])]:
        Xtr = torch.cat([feats[m][0] for m in mods], 1)
        Xov = torch.cat([feats[m][1] for m in mods], 1)
        run_config(name, Xtr[fi], Ytr_full[fi], Xtr[vi], gold_va,
                   Xov, gold_ov, canon, grouper, grid, dev, args.epochs)
    print("\nReading: if t+a+v is far below ~50 -> features are the ceiling (fix features).")
    print("         if t+a+v is near/above ~50 -> features fine, DQF/head is the issue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
