"""Data preparation for the MER2026 dataset (DG-DQA project).

Parses the MER2026 label / subtitle / description CSVs, normalizes the
open-vocabulary emotion labels to lowercased word lists, joins per-clip
descriptions and Chinese/English transcripts, builds the emotion vocabulary
from the TRAINING labels, and writes one JSONL manifest per split. Optionally
resolves media (audio / OpenFace face / video) paths if the extracted media
root is given.

Split roles (see CLAUDE.md sections 4-5):
  EVAL  = track2_train_human.csv         (human, open-vocab) -> held out, never trained.
  TRAIN = track2_train_mercaptionplus.csv MINUS any name that appears in the
          human eval set (removes the 548-clip train/eval leakage).

Descriptions: track2_* CSVs carry NO description column in MER2026; descriptions
are joined from track3_candidate.csv (columns a1/a2) by `name`. Coverage is
sparse (~8.7% of train); a fuller source is MER2025 track3_train_mercaptionplus.

Manifest entry (one JSON object per line):
  name, label_words[list], description[str|null], text_zh[str|null],
  text_en[str|null], audio_path[str|null], face_path[str|null], video_path[str|null]

Run:  python src/data_prep.py --data-root ~/SRIP/data/MER2026 \
          --out-dir ~/SRIP/data/manifests
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Reuse the EXACT label-field parser the metric uses, so words stored here map
# identically through the emotion wheel later.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import string_to_list  # noqa: E402


NAME_COL = "name"
LABEL_COL_CANDIDATES = ["openset", "labels", "label", "emotions", "emotion",
                        "ov_label", "openset_label", "gt"]
DESC_COL_CANDIDATES = ["description", "desc", "reason", "reasons", "caption",
                       "captions", "explanation", "a1", "a2"]
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------

def detect_label_column(df: pd.DataFrame, csv_name: str) -> str:
    """Pick the emotion-label column. Prefer known names, else the single
    non-name column, else a column whose cells look like word lists."""
    cols = [c for c in df.columns if c != NAME_COL]
    low = {c.lower(): c for c in cols}
    for cand in LABEL_COL_CANDIDATES:
        if cand in low:
            print(f"  [{csv_name}] label column  -> '{low[cand]}' (matched '{cand}')")
            return low[cand]
    if len(cols) == 1:
        print(f"  [{csv_name}] label column  -> '{cols[0]}' (only non-name column)")
        return cols[0]
    # heuristic: most cells contain a comma or brackets
    best, best_frac = None, 0.0
    for c in cols:
        s = df[c].astype(str)
        frac = (s.str.contains(",") | s.str.contains(r"\[")).mean()
        if frac > best_frac:
            best, best_frac = c, frac
    if best is not None:
        print(f"  [{csv_name}] label column  -> '{best}' (list-like, {best_frac:.0%} cells)")
        return best
    raise ValueError(f"{csv_name}: could not detect a label column among {cols}")


def detect_description_column(df: pd.DataFrame, csv_name: str) -> Optional[str]:
    """Pick a free-text description column if one exists, else None."""
    low = {c.lower(): c for c in df.columns if c != NAME_COL}
    for cand in DESC_COL_CANDIDATES:
        if cand in low:
            print(f"  [{csv_name}] desc column   -> '{low[cand]}' (matched '{cand}')")
            return low[cand]
    print(f"  [{csv_name}] desc column   -> NONE found (cols={list(df.columns)})")
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_label_words(field) -> List[str]:
    """Open-set field -> ordered, de-duplicated, lowercased word list."""
    words, seen = [], set()
    for w in string_to_list(field):
        w = str(w).lower().strip()
        if w and w not in seen:
            seen.add(w)
            words.append(w)
    return words


def clean_text(v) -> Optional[str]:
    """Return a stripped string or None (for NaN / empty)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def load_subtitles(path: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """name -> (chinese, english) from subtitle_chieng.csv."""
    df = pd.read_csv(path)
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for _, r in df.iterrows():
        out[r[NAME_COL]] = (clean_text(r.get("chinese")), clean_text(r.get("english")))
    return out


def load_descriptions(path: str, csv_name: str) -> Dict[str, str]:
    """name -> description. Uses the first detected description column
    (a1 preferred). Rows with empty description are skipped."""
    df = pd.read_csv(path)
    col = detect_description_column(df, csv_name)
    if col is None:
        return {}
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        d = clean_text(r[col])
        if d:
            out[r[NAME_COL]] = d
    return out


# ---------------------------------------------------------------------------
# Media indexing (optional; only if the extracted media root is provided)
# ---------------------------------------------------------------------------

def index_files_by_stem(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    """Walk `root`, map lowercased filename stem -> full path for files whose
    extension is in `exts`."""
    idx: Dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return idx
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in exts:
                idx.setdefault(stem.lower(), os.path.join(dirpath, f))
    return idx


def index_face_npy(root: str) -> Dict[str, str]:
    """Map clip name -> its OpenFace face `.npy` under the openface_face root.

    Layout (MER2026): openface_face/<name>/<name>.npy, a uint8 array of shape
    [T, 112, 112, 3] (T aligned RGB face frames). Falls back to any .npy in the
    per-clip dir, or a loose <name>.npy."""
    idx: Dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return idx
    for entry in os.scandir(root):
        if entry.is_dir():
            cand = os.path.join(entry.path, entry.name + ".npy")
            if os.path.exists(cand):
                idx[entry.name.lower()] = cand
                continue
            for f in os.listdir(entry.path):
                if f.endswith(".npy"):
                    idx[entry.name.lower()] = os.path.join(entry.path, f)
                    break
        elif entry.name.endswith(".npy"):
            idx[os.path.splitext(entry.name)[0].lower()] = entry.path
    return idx


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def build_split(df: pd.DataFrame, label_col: str, csv_name: str,
                subs: Dict[str, Tuple[Optional[str], Optional[str]]],
                descs: Dict[str, str],
                audio_idx: Dict[str, str], face_idx: Dict[str, str],
                video_idx: Dict[str, str]) -> List[dict]:
    """Build manifest records for one split."""
    records: List[dict] = []
    for _, r in df.iterrows():
        name = r[NAME_COL]
        zh, en = subs.get(name, (None, None))
        key = str(name).lower()
        records.append({
            "name": name,
            "label_words": parse_label_words(r[label_col]),
            "description": descs.get(name),
            "text_zh": zh,
            "text_en": en,
            "audio_path": audio_idx.get(key),
            "face_path": face_idx.get(key),
            "video_path": video_idx.get(key),
        })
    return records


def summarize(split: str, recs: List[dict]) -> dict:
    """Compute + print per-split stats; return a small summary dict."""
    n = len(recs)
    lab_lens = [len(r["label_words"]) for r in recs]
    n_desc = sum(1 for r in recs if r["description"])
    n_zh = sum(1 for r in recs if r["text_zh"])
    n_en = sum(1 for r in recs if r["text_en"])
    n_audio = sum(1 for r in recs if r["audio_path"])
    n_face = sum(1 for r in recs if r["face_path"])
    med = statistics.median(lab_lens) if lab_lens else 0
    print(f"\n[{split}] n={n}  median labels/clip={med}  "
          f"desc={n_desc}/{n} ({n_desc/n:.1%})  "
          f"zh={n_zh}/{n}  en={n_en}/{n}  "
          f"audio={n_audio}/{n}  face={n_face}/{n}")
    return {"split": split, "n": n, "median_labels": med,
            "desc_coverage": n_desc, "zh": n_zh, "en": n_en,
            "audio": n_audio, "face": n_face}


def write_jsonl(path: str, recs: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="MER2026 data preparation")
    ap.add_argument("--data-root", required=True,
                    help="dir with MER2026 CSVs (e.g. ~/SRIP/data/MER2026)")
    ap.add_argument("--out-dir", required=True,
                    help="output dir for manifests + vocab")
    ap.add_argument("--desc-csv", default="track3_candidate.csv",
                    help="CSV providing per-clip descriptions (name + a1/a2). "
                         "Swap to MER2025 track3_train_mercaptionplus.csv for "
                         "full coverage.")
    ap.add_argument("--media-root", default=None,
                    help="extracted media root (with audio/, openface_face/, "
                         "video/). If unset, media paths are left null.")
    args = ap.parse_args()

    root = os.path.expanduser(args.data_root)
    out = os.path.expanduser(args.out_dir)
    train_csv = os.path.join(root, "track2_train_mercaptionplus.csv")
    eval_csv = os.path.join(root, "track2_train_human.csv")
    sub_csv = os.path.join(root, "subtitle_chieng.csv")
    desc_csv = os.path.join(root, args.desc_csv)

    print("=== loading CSVs + auto-detecting columns ===")
    df_train = pd.read_csv(train_csv)
    df_eval = pd.read_csv(eval_csv)
    train_label = detect_label_column(df_train, "mercaptionplus(TRAIN)")
    eval_label = detect_label_column(df_eval, "human(EVAL)")
    # (mercaptionplus has no description column -> this prints NONE, by design)
    detect_description_column(df_train, "mercaptionplus(TRAIN)")

    print("\n=== joining subtitles + descriptions ===")
    subs = load_subtitles(sub_csv)
    print(f"  subtitles loaded: {len(subs)} names")
    descs = load_descriptions(desc_csv, os.path.basename(desc_csv))
    print(f"  descriptions loaded from {os.path.basename(desc_csv)}: {len(descs)} names")

    # media indices (optional)
    mr = os.path.expanduser(args.media_root) if args.media_root else None
    if mr:
        audio_idx = index_files_by_stem(os.path.join(mr, "audio"), AUDIO_EXTS)
        face_idx = index_face_npy(os.path.join(mr, "openface_face"))
        video_idx = index_files_by_stem(os.path.join(mr, "video"), VIDEO_EXTS)
        print(f"\n  media indexed: audio={len(audio_idx)} face={len(face_idx)} "
              f"video={len(video_idx)}")
    else:
        audio_idx = face_idx = video_idx = {}
        print("\n  media paths SKIPPED (no --media-root; rerun after extraction)")

    # ---- enforce eval-as-held-out: remove eval names from train ----
    eval_names = set(df_eval[NAME_COL])
    before = len(df_train)
    df_train = df_train[~df_train[NAME_COL].isin(eval_names)].reset_index(drop=True)
    removed = before - len(df_train)
    print(f"\n=== leakage removal ===\n  removed {removed} eval names from train "
          f"({before} -> {len(df_train)})")

    print("\n=== building manifests ===")
    train_recs = build_split(df_train, train_label, "TRAIN", subs, descs,
                             audio_idx, face_idx, video_idx)
    eval_recs = build_split(df_eval, eval_label, "EVAL", subs, descs,
                            audio_idx, face_idx, video_idx)

    # ---- vocabulary from TRAIN labels only ----
    counter: Counter = Counter()
    for r in train_recs:
        counter.update(r["label_words"])
    vocab = sorted(counter)
    print(f"\n=== vocabulary (from TRAIN labels) ===\n  size={len(vocab)}")
    print("  top 15:", ", ".join(f"{w}({c})" for w, c in counter.most_common(15)))

    # ---- write outputs ----
    write_jsonl(os.path.join(out, "train.jsonl"), train_recs)
    write_jsonl(os.path.join(out, "eval.jsonl"), eval_recs)
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"size": len(vocab), "words": vocab,
                   "counts": dict(counter)}, f, ensure_ascii=False, indent=2)

    s_train = summarize("TRAIN", train_recs)
    s_eval = summarize("EVAL", eval_recs)

    # ---- separation check + examples ----
    train_names = set(r["name"] for r in train_recs)
    overlap = len(train_names & eval_names)
    print(f"\n=== separation check ===\n  TRAIN ∩ EVAL names = {overlap} "
          f"({'OK, separated' if overlap == 0 else 'LEAKAGE!'})")

    print("\n=== a few parsed examples ===")
    for tag, recs in (("TRAIN", train_recs), ("EVAL", eval_recs)):
        for r in recs[:2]:
            d = r["description"]
            d = (d[:120] + "…") if d else "None"
            print(f"  [{tag}] {r['name']}: labels={r['label_words']}")
            print(f"          zh={(r['text_zh'] or '')[:40]!r}  desc={d!r}")

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"train": s_train, "eval": s_eval, "vocab_size": len(vocab),
                   "train_label_col": train_label, "eval_label_col": eval_label,
                   "desc_source": os.path.basename(desc_csv),
                   "leakage_removed": removed, "train_eval_overlap": overlap},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote train.jsonl, eval.jsonl, vocab.json, summary.json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
