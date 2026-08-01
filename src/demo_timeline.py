"""DG-DQA timeline demo: emotions over a LONG video, window by window.

A 1-min clip is too long for one read (the model emits ONE emotion set per
input, so a whole minute averages into mush). This splits the video into short
windows (~6 s), runs the model on each, prints a per-window timeline, and then
COMBINES the windows (mean score across windows) into one overall emotion set
for the whole video.

Reuses demo.py's encoders + face/audio extraction + the saved linear head.

Subtitle (the model leans on text): four options, in order of preference
  --asr                  auto-transcribe each window with Whisper (best, self-contained)
  --sub-vtt clip.vtt     per-window text from a YouTube/SRT-style .vtt
  --subtitle-zh "line"   ONE line applied to every window (ok)
  (none)                 text modality zeroed -> audio+visual only (weakest)

--asr needs faster-whisper (CTranslate2 backend, does NOT touch torch):
  pip install faster-whisper

Run:
  python src/demo_timeline.py --video clip.mp4 --window 6 --asr
  python src/demo_timeline.py --video clip.mp4 --window 6 --sub-vtt demo_src.zh-Hans.vtt
  python src/demo_timeline.py --video clip.mp4 --window 6 --subtitle-zh "一句话"
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import (Encoders, audio_from_video, faces_from_video,  # noqa: E402
                  CACHE)


# ---------------------------------------------------------------------------
# media helpers
# ---------------------------------------------------------------------------

def video_duration(path):
    """seconds (float) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise SystemExit(f"ffprobe could not read duration of {path}:\n{out.stderr}")


def cut_segment(path, t0, t1):
    """re-encode a [t0,t1] window to a temp mp4 (re-encode so seeking is exact)."""
    seg = os.path.join(tempfile.gettempdir(), f"win_{int(t0*1000)}_{int(t1*1000)}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
         "-c:v", "libx264", "-c:a", "aac", seg],
        capture_output=True)
    return seg


# ---------------------------------------------------------------------------
# ASR (faster-whisper) — lazy singleton
# ---------------------------------------------------------------------------

_ASR = {}


def transcribe(wav, model_name="small", lang=None):
    """transcribe a wav with faster-whisper; '' on failure / no audio."""
    if not wav or not os.path.exists(wav):
        return ""
    if "m" not in _ASR:
        from faster_whisper import WhisperModel
        dev = "cuda" if os.environ.get("ASR_CPU") != "1" else "cpu"
        ct = "float16" if dev == "cuda" else "int8"
        print(f"loading Whisper ({model_name}, {dev})…", flush=True)
        _ASR["m"] = WhisperModel(model_name, device=dev, compute_type=ct)
    segs, _ = _ASR["m"].transcribe(wav, language=lang, beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()


# ---------------------------------------------------------------------------
# subtitle (.vtt) parsing
# ---------------------------------------------------------------------------

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path):
    """return list of (start_sec, end_sec, text) cues from a .vtt/.srt file."""
    cues = []
    if not path or not os.path.exists(path):
        return cues
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            ts = _TS.findall(lines[i])
            if len(ts) >= 2:
                t0, t1 = _sec(*ts[0]), _sec(*ts[1])
                i += 1
                buf = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    txt = re.sub(r"<[^>]+>", "", lines[i]).strip()   # strip tags
                    if txt:
                        buf.append(txt)
                    i += 1
                if buf:
                    cues.append((t0, t1, " ".join(buf)))
                continue
        i += 1
    return cues


def text_for_window(cues, t0, t1, fallback):
    """concat cue texts that overlap [t0,t1]; else the fallback line."""
    if not cues:
        return fallback
    hit = [c[2] for c in cues if c[1] > t0 and c[0] < t1]
    # dedupe consecutive repeats (auto-subs duplicate a lot)
    seen, out = set(), []
    for h in hit:
        if h not in seen:
            seen.add(h); out.append(h)
    return " ".join(out) if out else fallback


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def prob_vector(model, enc, wav, imgs, text):
    """full sigmoid prob over canon (no thresholding)."""
    feats = {"text": enc.enc_text(text), "audio": enc.enc_audio(wav),
             "visual": enc.enc_faces(imgs)}
    x = torch.cat([feats[k] for k in model["order"]]).unsqueeze(0)
    W = model["state_dict"]["weight"]; b = model["state_dict"]["bias"]
    return torch.sigmoid(x @ W.t() + b)[0]


def labels_from_prob(prob, canon, thr, topk=None):
    above = [(canon[i], float(prob[i])) for i in range(len(canon)) if prob[i] > thr]
    above.sort(key=lambda t: -t[1])
    if not above:
        j = int(prob.argmax()); above = [(canon[j], float(prob[j]))]
    return above if topk is None else above[:topk]


def main():
    ap = argparse.ArgumentParser(description="DG-DQA emotion timeline over a long video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--window", type=float, default=6.0, help="window length (s)")
    ap.add_argument("--hop", type=float, default=None,
                    help="step between windows (s); default = window (no overlap)")
    ap.add_argument("--sub-vtt", default=None, help=".vtt/.srt with timed subtitles")
    ap.add_argument("--subtitle-zh", default=None, help="one line applied to all windows")
    ap.add_argument("--asr", action="store_true", help="auto-transcribe each window (Whisper)")
    ap.add_argument("--asr-model", default="small", help="faster-whisper model size")
    ap.add_argument("--asr-lang", default=None, help="force ASR language e.g. zh (default: auto)")
    ap.add_argument("--model", default=os.path.join(CACHE, "demo_model.pt"))
    args = ap.parse_args()

    model = torch.load(args.model, weights_only=False)
    enc = Encoders(model)
    canon, thr = model["canon"], model["threshold"]

    dur = video_duration(args.video)
    hop = args.hop or args.window
    cues = parse_vtt(args.sub_vtt)
    starts = []
    t = 0.0
    while t < dur - 0.5:                       # skip a <0.5s tail
        starts.append(t); t += hop
    subsrc = ("asr" if args.asr else
              f"vtt({len(cues)} cues)" if cues else
              "one-line" if args.subtitle_zh else "NONE")
    print(f"\n=== VIDEO {os.path.basename(args.video)} | {dur:.1f}s | "
          f"{len(starts)} windows of {args.window:.0f}s | subs: {subsrc} ===\n")

    agg = torch.zeros(len(canon))
    nwin = 0
    for t0 in starts:
        t1 = min(t0 + args.window, dur)
        seg = cut_segment(args.video, t0, t1)
        imgs = faces_from_video(seg)
        wav = audio_from_video(seg)
        if args.asr:
            txt = transcribe(wav, args.asr_model, args.asr_lang)
        else:
            txt = text_for_window(cues, t0, t1, args.subtitle_zh)
        prob = prob_vector(model, enc, wav, imgs, txt)
        agg += prob; nwin += 1
        top = labels_from_prob(prob, canon, thr, topk=4)
        nf = len(imgs)
        tx = (txt[:24] + "…") if txt and len(txt) > 24 else (txt or "—")
        print(f"  {t0:5.1f}-{t1:4.1f}s  faces={nf:2d}  txt:{tx:<26s}  "
              + ", ".join(f"{w}({s:.2f})" for w, s in top))
        try:
            os.remove(seg)
        except OSError:
            pass

    # ---- combine windows: mean prob across windows -> overall emotion set ----
    mean = agg / max(nwin, 1)
    overall = labels_from_prob(mean, canon, thr)
    print("\n=== COMBINED (mean score across all windows) ===")
    print("OVERALL EMOTIONS (above threshold):")
    for w, s in overall:
        print(f"   {w:18s} {s:.2f}")
    print("\n(top-10 overall by mean score):")
    rank = sorted([(canon[i], float(mean[i])) for i in range(len(canon))],
                  key=lambda t: -t[1])[:10]
    print("   " + ", ".join(f"{w}({s:.2f})" for w, s in rank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
