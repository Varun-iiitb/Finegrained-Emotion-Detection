"""DG-DQA demo: predict emotions from a clip (audio + face + subtitle).

Two modes:
  --clip  <name>     run on a dataset clip (uses the extracted audio/face/subtitle)
  --video <path>     run on YOUR OWN video file (needs ffmpeg + opencv for audio +
                     face detection); pass the subtitle text with --subtitle-zh

Loads the saved demo model (cache/demo_model.pt from train_demo_model.py) and the
same frozen encoders, encodes the three modalities, and prints the predicted
emotions (plus the human gold labels when running on a dataset clip).

Examples:
  python src/demo.py --clip samplenew3_00000062
  python src/demo.py --video myclip.mp4 --subtitle-zh "你怎么能这样对我"
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXTR = os.path.expanduser("~/SRIP/data/MER2026_extracted")
MER = os.path.expanduser("~/SRIP/data/MER2026")
CACHE = os.path.expanduser("~/SRIP/data/cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Frozen encoders (loaded once)
# ---------------------------------------------------------------------------

class Encoders:
    def __init__(self, cfg):
        print("loading encoders…", flush=True)
        from funasr import AutoModel as FAM
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        self.audio = FAM(model=cfg["audio_model"], device="cuda:0",
                         disable_update=True, disable_pbar=True)
        self.fer = AutoModel.from_pretrained(cfg["fer_model"]).eval().to(DEV)
        self.fer_proc = AutoImageProcessor.from_pretrained(cfg["fer_model"])
        self.tok = AutoTokenizer.from_pretrained(cfg["text_model"])
        self.bge = AutoModel.from_pretrained(cfg["text_model"]).eval().to(DEV)
        self.dims = cfg["dims"]

    @torch.no_grad()
    def enc_audio(self, wav):
        if not wav or not os.path.exists(wav):
            return torch.zeros(self.dims["audio"])
        r = self.audio.generate(wav, granularity="frame", extract_embedding=True,
                                disable_pbar=True)
        return torch.from_numpy(np.asarray(r[0]["feats"], dtype=np.float32).mean(0))

    @torch.no_grad()
    def enc_faces(self, imgs):
        if not imgs:
            return torch.zeros(self.dims["visual"])
        inp = self.fer_proc(images=imgs, return_tensors="pt").to(DEV)
        cls = self.fer(**inp).last_hidden_state[:, 0]          # [n,768]
        return cls.mean(0).float().cpu()

    @torch.no_grad()
    def enc_text(self, text):
        if not text:
            return torch.zeros(self.dims["text"])
        enc = self.tok(text, truncation=True, max_length=128,
                       return_tensors="pt").to(DEV)
        return self.bge(**enc).last_hidden_state[0].mean(0).float().cpu()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def faces_from_npy(npy_path, n=16):
    arr = np.load(npy_path)                                    # [T,112,112,3] uint8
    idx = np.linspace(0, len(arr) - 1, min(n, len(arr))).round().astype(int)
    return [Image.fromarray(arr[i]) for i in idx]


def subtitle_for(name):
    import pandas as pd
    df = pd.read_csv(os.path.join(MER, "subtitle_chieng.csv"))
    row = df[df["name"] == name]
    if len(row) == 0:
        return None, None
    zh = row.iloc[0]["chinese"]
    en = row.iloc[0]["english"]
    s = lambda x: (None if (x is None or (isinstance(x, float) and np.isnan(x)))
                   else str(x))
    return s(zh), s(en)


def gold_for(name):
    p = os.path.join(CACHE, "eval", name + ".pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False).get("label_words")
    return None


def faces_from_video(path, n=16):
    """Sample n frames, detect+crop the largest face per frame (opencv)."""
    try:
        import cv2
    except ImportError:
        print("[warn] opencv not installed -> using whole center frames (no face crop)")
        cv2 = None
    if cv2 is None:
        return _frames_centercrop(path, n)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return []
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    idxs = np.linspace(0, total - 1, n).round().astype(int)
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            crop = frame[y:y + h, x:x + w]
        else:
            crop = frame                                       # fallback: whole frame
        imgs.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    cap.release()
    return imgs


def _frames_centercrop(path, n):
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    imgs = []
    for fi in np.linspace(0, max(total - 1, 0), n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok:
            imgs.append(Image.fromarray(frame[:, :, ::-1]))
    cap.release()
    return imgs


def audio_from_video(path):
    wav = os.path.join(tempfile.gettempdir(), "demo_audio.wav")
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", wav]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return wav
    except Exception as e:
        print(f"[warn] ffmpeg failed ({e}); audio modality skipped")
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, enc, wav, imgs, text, topn=8):
    feats = {"text": enc.enc_text(text), "audio": enc.enc_audio(wav),
             "visual": enc.enc_faces(imgs)}
    x = torch.cat([feats[k] for k in model["order"]]).unsqueeze(0)
    W = model["state_dict"]["weight"]; b = model["state_dict"]["bias"]
    prob = torch.sigmoid(x @ W.t() + b)[0]
    canon, thr = model["canon"], model["threshold"]
    above = [(canon[i], float(prob[i])) for i in range(len(canon)) if prob[i] > thr]
    above.sort(key=lambda t: -t[1])
    if not above:                                              # never empty: top-1
        j = int(prob.argmax()); above = [(canon[j], float(prob[j]))]
    ranked = sorted([(canon[i], float(prob[i])) for i in range(len(canon))],
                    key=lambda t: -t[1])[:topn]
    return above, ranked


def main():
    ap = argparse.ArgumentParser(description="DG-DQA emotion demo")
    ap.add_argument("--clip", default=None, help="dataset clip name")
    ap.add_argument("--video", default=None, help="path to your own video file")
    ap.add_argument("--subtitle-zh", default=None, help="subtitle for --video")
    ap.add_argument("--model", default=os.path.join(CACHE, "demo_model.pt"))
    args = ap.parse_args()

    model = torch.load(args.model, weights_only=False)
    enc = Encoders(model)

    if args.clip:
        name = args.clip
        wav = os.path.join(EXTR, "audio", name + ".wav")
        fnpy = os.path.join(EXTR, "openface_face", name, name + ".npy")
        imgs = faces_from_npy(fnpy) if os.path.exists(fnpy) else []
        zh, en = subtitle_for(name)
        gold = gold_for(name)
        print(f"\n=== CLIP {name} ===")
        print(f"subtitle (zh): {zh}")
        if en: print(f"subtitle (en): {en}")
    elif args.video:
        wav = audio_from_video(args.video)
        imgs = faces_from_video(args.video)
        zh = args.subtitle_zh
        gold = None
        print(f"\n=== VIDEO {os.path.basename(args.video)} ===")
        print(f"frames with faces: {len(imgs)} | audio: {'yes' if wav else 'no'}")
        print(f"subtitle (zh): {zh}")
    else:
        print("give --clip <name> or --video <path>"); return 2

    pred, ranked = predict(model, enc, wav, imgs, zh)
    print("\nPREDICTED EMOTIONS (above threshold):")
    for w, s in pred:
        print(f"   {w:18s} {s:.2f}")
    print("\n(top-8 by score, for reference):")
    print("   " + ", ".join(f"{w}({s:.2f})" for w, s in ranked))
    if gold:
        print(f"\nHUMAN GOLD LABELS: {gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
