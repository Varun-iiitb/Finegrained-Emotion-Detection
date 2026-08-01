"""Feature extraction with FROZEN encoders -> per-clip cache (DG-DQA project).

Runs three frozen encoders ONCE over a split's manifest and caches per-clip
SEQUENCES to disk (so the DQF fusion body can do Conv1D / Q-UA / Q-CA over the
time axis). See the data contract in CLAUDE.md section 5.

Encoders (frozen, .eval(), torch.no_grad(), on CUDA):
  audio  = emotion2vec      (FunASR `iic/emotion2vec_base`)  frame-level  -> [T_a, 768]
  visual = CLIP-ViT-L/14    (`openai/clip-vit-large-patch14`) per-face    -> [N_f, 768]
  text   = BGE-M3           (`BAAI/bge-m3`)  token states (zh & en)        -> [T_t, 1024]
                            and POOLED (CLS, L2-norm) for description + vocab words.

Per-clip cache file `cache/<split>/<name>.pt` (a dict, torch tensors):
  name, X_audio[T_a,768]f16, X_visual[N_f,768]f16, X_text_zh[T_zh,1024]f16,
  X_text_en[T_en,1024]f16, desc_emb[1024]f32 (train only), present{...}, label_words.
Also writes `cache/<split>_meta.json` and (once) `cache/vocab_emb.pt`.

Robust to missing media (zero placeholder + present=False + count). Resumable:
skips a clip whose cache file already exists. `--limit N` for debug.

Run (eval, smoke):  python src/extract_features.py \
    --manifest ~/SRIP/data/manifests/eval.jsonl --split eval \
    --out-dir ~/SRIP/data/cache --vocab ~/SRIP/data/manifests/vocab.json --limit 16
Run (train):        ... --manifest .../train.jsonl --split train --is-train
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

AUDIO_MODEL = "iic/emotion2vec_base"
VISUAL_MODEL = "openai/clip-vit-large-patch14"
TEXT_MODEL = "BAAI/bge-m3"
D_AUDIO, D_VISUAL, D_TEXT = 768, 768, 1024


# ---------------------------------------------------------------------------
# Frozen encoder wrappers
# ---------------------------------------------------------------------------

class AudioEncoder:
    """emotion2vec frame-level embeddings via FunASR -> [T_a, 768] float16."""

    def __init__(self, device: str = "cuda:0"):
        from funasr import AutoModel
        self.model = AutoModel(model=AUDIO_MODEL, device=device,
                               disable_update=True, disable_pbar=True)

    @torch.no_grad()
    def encode(self, wav_path: str, max_frames: int) -> torch.Tensor:
        res = self.model.generate(wav_path, granularity="frame",
                                  extract_embedding=True, disable_pbar=True)
        feats = np.asarray(res[0]["feats"], dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        if len(feats) > max_frames:               # uniform subsample, not truncate
            idx = np.linspace(0, len(feats) - 1, max_frames).round().astype(int)
            feats = feats[idx]
        return torch.from_numpy(feats).half()


class VisualEncoder:
    """CLIP-ViT-L/14 projected image embedding per sampled face -> [N_f, 768]."""

    def __init__(self, device: str = "cuda:0"):
        from transformers import CLIPModel, AutoProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(VISUAL_MODEL).eval().to(device)
        self.proc = AutoProcessor.from_pretrained(VISUAL_MODEL)

    @torch.no_grad()
    def encode(self, npy_path: str, n_faces: int) -> torch.Tensor:
        faces = np.load(npy_path)                  # [T,112,112,3] uint8
        if faces.ndim != 4 or faces.shape[0] == 0:
            raise ValueError(f"bad face array {faces.shape} in {npy_path}")
        T = faces.shape[0]
        idx = np.linspace(0, T - 1, min(n_faces, T)).round().astype(int)
        imgs = [Image.fromarray(faces[i]) for i in idx]
        inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
        vis = self.model.vision_model(pixel_values=inp["pixel_values"])
        emb = self.model.visual_projection(vis.pooler_output)   # [n,768]
        return emb.float().cpu().half()


class TextEncoder:
    """BGE-M3: token sequences (for fusion) and pooled CLS (L2-norm, for targets)."""

    def __init__(self, device: str = "cuda:0"):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_MODEL).eval().to(device)

    @torch.no_grad()
    def encode_tokens(self, text: str, max_len: int) -> torch.Tensor:
        enc = self.tok(text, truncation=True, max_length=max_len,
                       return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state[0]            # [T,1024]
        return out.float().cpu().half()

    @torch.no_grad()
    def encode_pooled(self, texts: List[str], max_len: int, bs: int = 64) -> torch.Tensor:
        """L2-normalized CLS embeddings for a list of texts -> [N,1024] float32."""
        chunks = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]      # [b,1024]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            chunks.append(cls.float().cpu())
        return torch.cat(chunks, 0)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _zeros(d: int) -> torch.Tensor:
    return torch.zeros(1, d, dtype=torch.float16)


def extract_split(records: List[dict], split: str, out_dir: str, is_train: bool,
                  audio: AudioEncoder, visual: VisualEncoder, text: TextEncoder,
                  caps: Dict[str, int]) -> Dict[str, int]:
    """Encode every clip in `records`, writing one cache file each. Resumable."""
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    miss = {"audio": 0, "visual": 0, "text_zh": 0, "text_en": 0, "desc": 0}
    done = skipped = 0

    for i, r in enumerate(records):
        name = r["name"]
        out_path = os.path.join(split_dir, f"{name}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        present = {}

        # audio
        try:
            if r.get("audio_path") and os.path.exists(r["audio_path"]):
                X_audio = audio.encode(r["audio_path"], caps["audio"])
                present["audio"] = True
            else:
                raise FileNotFoundError
        except Exception:
            X_audio = _zeros(D_AUDIO); present["audio"] = False; miss["audio"] += 1

        # visual
        try:
            if r.get("face_path") and os.path.exists(r["face_path"]):
                X_visual = visual.encode(r["face_path"], caps["faces"])
                present["visual"] = True
            else:
                raise FileNotFoundError       # NOTE: no video fallback (video not downloaded)
        except Exception:
            X_visual = _zeros(D_VISUAL); present["visual"] = False; miss["visual"] += 1

        # text zh / en
        zh, en = r.get("text_zh"), r.get("text_en")
        if zh:
            X_text_zh = text.encode_tokens(zh, caps["text"]); present["zh"] = True
        else:
            X_text_zh = _zeros(D_TEXT); present["zh"] = False; miss["text_zh"] += 1
        if en:
            X_text_en = text.encode_tokens(en, caps["text"]); present["en"] = True
        else:
            X_text_en = _zeros(D_TEXT); present["en"] = False; miss["text_en"] += 1

        # description anchor (train only)
        desc_emb = None
        if is_train:
            d = r.get("description")
            if d:
                desc_emb = text.encode_pooled([d], caps["desc"])[0]   # [1024] f32
                present["desc"] = True
            else:
                present["desc"] = False; miss["desc"] += 1

        rec = {"name": name, "X_audio": X_audio, "X_visual": X_visual,
               "X_text_zh": X_text_zh, "X_text_en": X_text_en,
               "present": present, "label_words": r.get("label_words", [])}
        if desc_emb is not None:
            rec["desc_emb"] = desc_emb
        torch.save(rec, out_path)
        done += 1
        if (done + skipped) % 200 == 0 or i == len(records) - 1:
            print(f"  [{split}] {done+skipped}/{len(records)}  "
                  f"(new={done} skip={skipped})  miss={miss}", flush=True)

    meta = {"split": split, "n": len(records), "new": done, "skipped": skipped,
            "missing": miss, "dims": {"audio": D_AUDIO, "visual": D_VISUAL, "text": D_TEXT},
            "caps": caps, "models": {"audio": AUDIO_MODEL, "visual": VISUAL_MODEL,
                                     "text": TEXT_MODEL}}
    with open(os.path.join(out_dir, f"{split}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def encode_vocab(vocab_path: str, out_dir: str, text: TextEncoder, max_len: int) -> int:
    """Encode each vocab emotion word -> L2-normed BGE-M3 embedding; cache once."""
    out_path = os.path.join(out_dir, "vocab_emb.pt")
    if os.path.exists(out_path):
        print(f"  vocab_emb exists, skipping ({out_path})")
        return 0
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)["words"]
    emb = text.encode_pooled(vocab, max_len)          # [V,1024] f32
    torch.save({"vocab": vocab, "word_emb": emb}, out_path)
    print(f"  vocab_emb: {emb.shape} -> {out_path}")
    return len(vocab)


def main() -> int:
    ap = argparse.ArgumentParser(description="DG-DQA frozen-encoder feature cache")
    ap.add_argument("--manifest", required=True, help="split jsonl from data_prep")
    ap.add_argument("--split", required=True, help="split name (train/eval)")
    ap.add_argument("--out-dir", required=True, help="cache root")
    ap.add_argument("--is-train", action="store_true",
                    help="also encode per-clip description anchor (desc_emb)")
    ap.add_argument("--vocab", default=None,
                    help="vocab.json -> also build cache/vocab_emb.pt (once)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N clips")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-audio-frames", type=int, default=512)
    ap.add_argument("--n-faces", type=int, default=16)
    ap.add_argument("--max-text-tokens", type=int, default=128)
    ap.add_argument("--max-desc-tokens", type=int, default=256)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"loaded {len(records)} records from {args.manifest}")
    os.makedirs(args.out_dir, exist_ok=True)
    caps = {"audio": args.max_audio_frames, "faces": args.n_faces,
            "text": args.max_text_tokens, "desc": args.max_desc_tokens}

    print("loading encoders (frozen)…")
    audio = AudioEncoder(args.device)
    visual = VisualEncoder(args.device)
    text = TextEncoder(args.device)

    if args.vocab:
        encode_vocab(args.vocab, args.out_dir, text, args.max_text_tokens)

    meta = extract_split(records, args.split, args.out_dir, args.is_train,
                         audio, visual, text, caps)
    print(f"\n[done] split={args.split}  new={meta['new']} skipped={meta['skipped']}")
    print(f"       dims: audio={D_AUDIO} visual={D_VISUAL} text={D_TEXT}")
    print(f"       missing: {meta['missing']}")
    print(f"       wrote {args.split}_meta.json -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
