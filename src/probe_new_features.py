"""Validate UPGRADED encoders on the 1,532 human clips BEFORE full re-extraction.

Re-encodes the human (eval) clips with emotion2vec_plus_large (audio) and 3 FER
ViT backbones (visual), reuses the existing BGE text, and runs the human-CV
linear probe per modality + best combination. Compares to the old features
(audio 17.8 / visual 21.9 / text 24.5 / combined 41.7). Caches the pooled
features so reruns are instant. Diagnostic only.
"""

import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import WheelGrouper, score_group_sets  # noqa: E402

CACHE = os.path.expanduser("~/SRIP/data/cache")
AUD = os.path.expanduser("~/SRIP/data/MER2026_extracted/audio")
FAC = os.path.expanduser("~/SRIP/data/MER2026_extracted/openface_face")
WHEEL = os.path.expanduser("~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel")
FERS = {"trpakov": "trpakov/vit-face-expression",
        "dima806": "dima806/facial_emotions_image_detection",
        "motheecreator": "motheecreator/vit-Facial-Expression-Recognition"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def encode_all():
    cpath = os.path.join(CACHE, "pooled_new_eval.pt")
    if os.path.exists(cpath):
        print(f"[cache] loading {cpath}")
        return torch.load(cpath, weights_only=False)

    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(CACHE, "eval", "*.pt")))
    print(f"[encode] {len(names)} human clips")
    text, labels = {}, {}
    for n in names:
        r = torch.load(os.path.join(CACHE, "eval", n + ".pt"), weights_only=False)
        text[n] = r["X_text_zh"].float().mean(0)
        labels[n] = r["label_words"]

    print("[encode] audio: emotion2vec_plus_large …")
    from funasr import AutoModel as FAM
    am = FAM(model="iic/emotion2vec_plus_large", device="cuda:0",
             disable_update=True, disable_pbar=True)
    audio = {}
    for i, n in enumerate(names):
        rr = am.generate(os.path.join(AUD, n + ".wav"), granularity="frame",
                         extract_embedding=True, disable_pbar=True)
        audio[n] = torch.from_numpy(np.asarray(rr[0]["feats"], dtype=np.float32).mean(0))
        if i % 300 == 0:
            print(f"   audio {i}/{len(names)}", flush=True)
    del am
    torch.cuda.empty_cache()

    from transformers import AutoModel, AutoImageProcessor
    vis = {k: {} for k in FERS}
    for key, fid in FERS.items():
        print(f"[encode] visual: {fid} …")
        proc = AutoImageProcessor.from_pretrained(fid)
        mod = AutoModel.from_pretrained(fid).eval().to(DEV)
        for i, n in enumerate(names):
            faces = np.load(os.path.join(FAC, n, n + ".npy"))
            idx = np.linspace(0, len(faces) - 1, min(16, len(faces))).round().astype(int)
            imgs = [Image.fromarray(faces[j]) for j in idx]
            inp = proc(images=imgs, return_tensors="pt").to(DEV)
            with torch.no_grad():
                cls = mod(**inp).last_hidden_state[:, 0]      # [n,768]
            vis[key][n] = cls.mean(0).float().cpu()
            if i % 500 == 0:
                print(f"   {key} {i}/{len(names)}", flush=True)
        del mod
        torch.cuda.empty_cache()

    d = {"names": names, "text": text, "audio": audio, "vis": vis, "labels": labels}
    torch.save(d, cpath)
    print(f"[cache] saved {cpath}")
    return d


def cv_probe(X, gold, canon, folds=5, epochs=150, mlp=False):
    cidx = {c: i for i, c in enumerate(canon)}
    C = len(canon)
    Y = torch.zeros(len(gold), C)
    for i, gs in enumerate(gold):
        for c in gs:
            if c in cidx:
                Y[i, cidx[c]] = 1.0
    N = X.size(0)
    idx = list(range(N)); random.Random(0).shuffle(idx)
    fl = [list(f) for f in np.array_split(idx, folds)]
    grid = [round(x, 4) for x in np.arange(0.1, 0.85, 0.025)]

    def fs(prob, gs, thr):
        pred = []
        for i in range(prob.size(0)):
            ii = (prob[i] > thr).nonzero().flatten()
            if ii.numel() == 0:
                ii = prob[i].argmax().view(1)
            pred.append({canon[j] for j in ii.tolist()})
        return score_group_sets(gs, pred)["f_s"]

    out = []
    for k in range(folds):
        te = set(fl[k]); rest = [i for i in idx if i not in te]
        val, fit = rest[:len(rest)//5], rest[len(rest)//5:]
        if mlp:
            head = nn.Sequential(nn.Linear(X.size(1), 512), nn.GELU(),
                                 nn.Dropout(0.5), nn.Linear(512, C)).to(DEV)
        else:
            head = nn.Linear(X.size(1), C).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
        Xf, Yf = X[fit].to(DEV), Y[fit].to(DEV)
        for _ in range(epochs):
            loss = F.binary_cross_entropy_with_logits(head(Xf), Yf)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(head(X[val].to(DEV))).cpu()
            pt = torch.sigmoid(head(X[list(te)].to(DEV))).cpu()
        gv = [gold[i] for i in val]; gt = [gold[i] for i in te]
        best = max(grid, key=lambda t: fs(pv, gv, t))
        out.append(fs(pt, gt, best) * 100)
    return float(np.mean(out)), float(np.std(out))


def main():
    grouper = WheelGrouper(WHEEL)
    d = encode_all()
    names = d["names"]
    gold = [grouper.to_group_set(d["labels"][n], "case2", "drop") for n in names]
    canon = sorted(set().union(*gold))
    stk = lambda dct: torch.stack([dct[n] for n in names])
    T, A = stk(d["text"]), stk(d["audio"])

    print(f"\n=== NEW-FEATURE human-CV linear probe (C={len(canon)}) ===")
    print("  text   (BGE, old)       : %.2f ± %.2f" % cv_probe(T, gold, canon))
    print("  audio  (plus_large NEW) : %.2f ± %.2f" % cv_probe(A, gold, canon))
    best = None
    for key in FERS:
        V = stk(d["vis"][key])
        m, s = cv_probe(V, gold, canon)
        print(f"  visual ({key:13s} NEW): %.2f ± %.2f" % (m, s))
        if best is None or m > best[1]:
            best = (key, m, V)
    print(f"  --> best visual: {best[0]} ({best[1]:.2f})")
    comb = torch.cat([T, A, best[2]], 1)
    print("  COMBINED t+a+v (NEW)    : %.2f ± %.2f" % cv_probe(comb, gold, canon))

    print("\n=== head / normalization / ensemble (cached feats, instant) ===")
    l2 = lambda x: F.normalize(x, dim=1)
    combn = torch.cat([l2(T), l2(A), l2(best[2])], 1)
    print("  combined L2-norm  (lin) : %.2f ± %.2f" % cv_probe(combn, gold, canon))
    print("  combined          (mlp) : %.2f ± %.2f" % cv_probe(comb, gold, canon, mlp=True))
    print("  combined L2-norm  (mlp) : %.2f ± %.2f" % cv_probe(combn, gold, canon, mlp=True))
    ens = torch.cat([l2(T), l2(A)] + [l2(stk(d["vis"][k])) for k in FERS], 1)
    print("  3-FER ensemble L2 (lin) : %.2f ± %.2f" % cv_probe(ens, gold, canon))
    print("  3-FER ensemble L2 (mlp) : %.2f ± %.2f" % cv_probe(ens, gold, canon, mlp=True))
    print("\n  reference: OLD combined human-CV 41.7 | NEW combined 44.5 | gate ~50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
