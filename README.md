# DG-DQA — Open-Vocabulary Multimodal Emotion Recognition

A lightweight, discriminative model for **fine-grained, open-vocabulary, multi-label
emotion recognition** on short video clips (**text + audio + visual**). It predicts
any number of emotions per clip from a large open label space (~190+ categories),
not a fixed 6–7 class set. We train our own model — we do **not** prompt a
multimodal LLM for answers.

## Architecture (one line)

Frozen cached encoders (audio = emotion2vec, visual = CLIP-ViT-L / FER-ViT on
OpenFace faces, text = BGE-M3, Chinese-capable) → **DQF fusion body** (per-modality
Conv1D → Q-UA unimodal → Q-CA pairwise cross-modal → global [CLS] fusion) →
**open-vocab head** projecting the fused vector into the text-embedding space,
matched to emotion-word anchors by cosine similarity, trained with a multi-positive
contrastive (InfoNCE) loss.

## Key Results

| Setup | Data | Model | F_s |
|---|---|---|---|
| **Human-aligned 5-fold CV** ⭐ | 1,532 human clips | linear probe (frozen feats) | **44.6** |
| Machine-only 5-fold CV | 30,779 machine clips | full DQF (~20M) | 36.5 ± 0.5 |
| Machine-only single split | 30,779 machine clips | full DQF | 36.9 |
| DQF on human data | 1,532 human clips | full DQF | 34.7 |
| Cross-data (rejected) | machine → human | full DQF | ~28 |

**Findings:** the simple linear head beat the heavy DQF on the small clean set
(overfitting); systematic ablations rule out features, architecture, and label-space
coverage as the bottleneck — the ceiling is **auto-annotation label noise**. See
[REPORT.md](REPORT.md) for the full write-up and [CLAUDE.md](CLAUDE.md) for the
detailed project log (env, data contract, metric, decisions, gotchas).

## Current status & where to continue

**Where things stand:** the full pipeline (encoders → cache → DQF → open-vocab head)
is built, validated, and plotted. Best result **44.6 F_s** (human, linear probe);
DQF within-machine **36.5 ± 0.5**. The bottleneck has been diagnosed as **label
noise**, not the model.

**Already ruled out — don't re-spend effort here:**
- Bigger/heavier models (DQF and MLP heads *overfit*, lose to the linear probe).
- Better feature *coverage* of the label space (curated 193-label taxonomy → F_s flat).
- Encoder features are not the limiter *at the current label ceiling* (linear probe ≈ 44).

**Most promising next directions (in rough priority):**
1. **More clean human-labeled data** — the single biggest lever (human set is only 1,532).
2. **Fine-tune the encoders on emotion** (parameter-efficient / LoRA) or add richer
   visual (body/scene, not face-only) + stronger audio.
3. **Train a loss closer to F_s** (current train objective is contrastive; selection is F_s).
4. **Open-vocab zero-shot for the rare tail** — the AlignHead can emit unseen emotion
   words; evaluate/exploit this instead of training on noisy rare labels.

**Not recommended:** adding out-of-domain data (e.g. MUStARD) to "add" emotions like
sarcasm — analysis shows sarcasm is already present but fragmented/noisy, not missing.

## Repository layout

```
src/
  env_check.py          torch / CUDA / sm_120 sanity check
  metric.py             F_s emotion-wheel-grouped set metric
  data_prep.py          dataset parsing, label space, wheel groups -> manifests
  extract_features.py   frozen encoders -> cached features
  dataset.py            DataLoader over cached features
  models.py             DQF fusion body (+ granularities)
  train.py              open-vocab head + contrastive training (gate)
  train_machine_cv.py   within-machine train/val/test split + 5-fold CV (DQF)
  train_hybrid.py       pretrain on pool -> human-CV fine-tune
  linear_probe.py       feature-ceiling linear probe (frozen pooled feats)
  probe_new_features.py upgraded-encoder human-CV probe (the 44.6 recipe)
  train_demo_model.py   trains the saved demo model (data/cache/demo_model.pt)
  demo.py / demo_timeline.py   inference demos
  relabel_map.py        builds the curated repaired label map
  label_curation.py     human-reviewed label overrides (edit lists here)
  plot_training.py      training-curve plots (loss + val F_s)
  parse_logs.py         reconstruct history.json from old stdout logs
data/
  manifests/            train.jsonl, eval.jsonl, vocab.json, summary.json
  cache/                small artifacts: vocab_emb.pt, anchors_*.pt,
                        pooled_new_eval.pt, demo_model.pt, repaired_label_map.json,
                        *_meta.json   (the large feature cache is NOT included — see below)
runs/                   all training plots (plot.png) + history.json / result.json
logs/                   training stdout logs
third_party/AffectGPT/OV-MER/emotion_wheel/   F_s metric wheel files (format.csv,
                        synonym.xlsx, wheel{1..5}.xlsx) — from the AffectGPT repo
```

## Data availability

This project uses **MER2026 / MER-Caption+** (`MERChallenge/MER2026` on HuggingFace),
a **gated dataset** — you must be approved to download it. This repository therefore
does **not** include:

- the raw media (`data/MER2026/`, `data/MER2026_extracted/` — audio, face frames),
- the per-clip **feature cache** (`data/cache/train/`, `data/cache/eval/` — tens of GB).

The included `data/manifests/*.jsonl` describe the splits and labels. Small derived
artifacts needed to run the demo / linear probe (`vocab_emb.pt`, `pooled_new_eval.pt`,
`demo_model.pt`, anchors, the curated label map) **are** included.

## Reproduce

```bash
python src/env_check.py                          # verify torch / CUDA / GPU (expect (12,0))

# --- needs the gated dataset downloaded + extracted (see Data availability) ---
python src/data_prep.py --data-root data/MER2026 --out-dir data/manifests \
    --desc-csv data/MER2025/track3_train_mercaptionplus.csv \
    --media-root data/MER2026_extracted                     # build manifests
python src/extract_features.py --manifest data/manifests/eval.jsonl  --split eval  --out-dir data/cache
python src/extract_features.py --manifest data/manifests/train.jsonl --split train --out-dir data/cache --is-train

# --- these run from the included artifacts (no raw data needed) ---
python src/relabel_map.py                                    # build curated 193-label map
python src/probe_new_features.py                             # human-CV linear probe -> 44.6

# --- these need the feature cache from extract_features above ---
python src/train_machine_cv.py --mode cv --folds 5 --epochs 20        # DQF 5-fold -> 36.5
python src/train_machine_cv.py --mode split --epochs 20 \
    --label-map data/cache/repaired_label_map.json                    # curated split
python src/plot_training.py --run runs/machine_cv_zh_canon            # training plots
python src/parse_logs.py --log logs/gate_zh_s0.log --out runs/gate_zh_word   # plot old runs
```

Model selection uses **only** the F_s metric (never accuracy or loss). Encoders are
frozen; only the fusion body + head train.

