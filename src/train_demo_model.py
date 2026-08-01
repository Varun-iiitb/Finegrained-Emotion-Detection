"""Train + save ONE deployable model for the demo.

A simple linear multi-label head on the UPGRADED pooled features
(emotion2vec_plus_large audio + FER-ViT visual + BGE text), trained on the human
OV-MERD clips. This is the ~44.6-F_s recipe distilled into a single saved model
that demo.py loads. Threshold is tuned on a held-out slice, then the final head
is trained on all human clips.

Run:  python src/train_demo_model.py          # writes cache/demo_model.pt
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import WheelGrouper, score_group_sets  # noqa: E402

CACHE = os.path.expanduser("~/SRIP/data/cache")
WHEEL = os.path.expanduser("~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel")
# the encoders the demo will use (must match demo.py)
AUDIO_MODEL = "iic/emotion2vec_plus_large"
TEXT_MODEL = "BAAI/bge-m3"
FER_KEY, FER_MODEL = "trpakov", "trpakov/vit-face-expression"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def multihot(gold_sets, cidx, C):
    Y = torch.zeros(len(gold_sets), C)
    for i, gs in enumerate(gold_sets):
        for c in gs:
            if c in cidx:
                Y[i, cidx[c]] = 1.0
    return Y


def train_head(X, Y, epochs=200):
    head = nn.Linear(X.size(1), Y.size(1)).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
    X, Y = X.to(DEV), Y.to(DEV)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(head(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def main():
    grouper = WheelGrouper(WHEEL)
    pooled = torch.load(os.path.join(CACHE, "pooled_new_eval.pt"), weights_only=False)
    names = pooled["names"]
    stk = lambda dct: torch.stack([dct[n] for n in names])
    T, A, V = stk(pooled["text"]), stk(pooled["audio"]), stk(pooled["vis"][FER_KEY])
    X = torch.cat([T, A, V], 1)
    gold = [grouper.to_group_set(pooled["labels"][n], "case2", "drop") for n in names]
    canon = sorted(set().union(*gold))
    cidx = {c: i for i, c in enumerate(canon)}
    Y = multihot(gold, cidx, len(canon))
    print(f"train clips={len(names)}  feat_dim={X.size(1)}  classes={len(canon)}")

    # tune threshold on a held-out 20% slice
    idx = list(range(len(names))); random.Random(0).shuffle(idx)
    nv = len(idx) // 5
    vi, fi = idx[:nv], idx[nv:]
    head = train_head(X[fi], Y[fi])
    with torch.no_grad():
        pv = torch.sigmoid(head(X[vi].to(DEV))).cpu()
    gv = [gold[i] for i in vi]

    def fs(prob, gsets, thr):
        pred = []
        for i in range(prob.size(0)):
            ii = (prob[i] > thr).nonzero().flatten()
            if ii.numel() == 0:
                ii = prob[i].argmax().view(1)
            pred.append({canon[j] for j in ii.tolist()})
        return score_group_sets(gsets, pred)["f_s"]

    grid = [round(x, 4) for x in np.arange(0.1, 0.85, 0.025)]
    best_thr = max(grid, key=lambda t: fs(pv, gv, t))
    print(f"tuned threshold = {best_thr}  (held-out F_s={fs(pv,gv,best_thr)*100:.2f})")

    # final head on ALL human clips
    head = train_head(X, Y)
    out = {
        "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "in_dim": X.size(1), "canon": canon, "threshold": best_thr,
        "audio_model": AUDIO_MODEL, "text_model": TEXT_MODEL, "fer_model": FER_MODEL,
        "dims": {"text": T.size(1), "audio": A.size(1), "visual": V.size(1)},
        "order": ["text", "audio", "visual"],
    }
    path = os.path.join(CACHE, "demo_model.pt")
    torch.save(out, path)
    print(f"[saved] {path}  (linear {X.size(1)}->{len(canon)}, thr={best_thr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
