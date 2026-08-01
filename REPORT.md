# Open-Vocabulary Multimodal Emotion Recognition — Project Report

## 1. Objective

Build a **lightweight, discriminative model** that recognizes emotions in short video
clips using three modalities — **text, audio, and visual**. Unlike standard systems
that classify into a fixed 6–7 emotions, our model is:

- **Open-vocabulary** — predicts from a large space of ~190+ emotions, including
  subtle ones (shy, grateful, nervous, sarcastic).
- **Multi-label** — a clip can carry several emotions simultaneously.

We train our *own* model; we do not prompt a large language model for answers.

## 2. Data

- **Source:** MER2026 / MER-Caption+ — Chinese TV clips with English subtitles.
- **Training set:** 30,779 clips, **machine (auto) labeled** — plentiful but noisy.
- **Evaluation set:** 1,532 clips, **human labeled** — small but clean.
- **Per clip:** audio (WAV), cropped OpenFace face frames, Chinese/English subtitles,
  and a free-text description.
- **Label space:** 3,311 distinct raw emotion words → collapsed to **190 canonical
  emotions** using the OV-MER emotion wheel (merges synonyms, e.g. *happy/joyful → joy*).

## 3. Overall Approach

```
Frozen encoders → cached features → DQF fusion body → open-vocab head → cosine-match to emotion words → labels
```

The three encoders are **frozen** and run once, with outputs cached to disk. Only the
fusion body and head are trained. This makes training cheap and reproducible on a
single 16 GB GPU.

## 4. Frozen Encoders and Dimensions

| Modality | Encoder | Output dim |
|---|---|---|
| Audio | emotion2vec (plus_large) | 768 / 1024 |
| Visual | CLIP-ViT-L/14 (and FER-ViT) on face frames `[T,112,112,3]` | 768 |
| Text | BGE-M3 (Chinese-capable) | 1024 |

Features are stored as **sequences over time** (not averaged), so the fusion model
can use temporal structure.

## 5. Feature Cache (data contract)

Each clip is stored as:
- `X_audio [T_a≤512, 768]`, `X_visual [N_f≤16, 768]`, `X_text_zh [T_zh≤128, 1024]`
- `desc_emb [1024]` — description embedding (training only)
- Missing modality → zero placeholder + a mask flag (for robustness)

All 32,311 clips were encoded once and cached.

## 6. DQF Fusion Body (core architecture)

The DQF (Dual-Query Fusion) body combines the three streams in stages:

1. **Per-modality preprocessing:** `Linear → Conv1D` (project to width 256, smooth
   over time).
2. **Q-UA (unimodal):** 8 learnable query vectors summarize each modality on its own,
   followed by a 6-layer Transformer → per-modality features `t, a, v`.
3. **Q-CA (cross-modal):** for each pair {text-audio, text-visual, audio-visual}, a
   shared query checks how the two modalities agree or disagree → pair features
   `ta, tv, av`. *(This captures cross-modal mismatch, e.g. sarcasm.)*
4. **Global fusion:** `[CLS]` token + 3 pair features → 2-layer Transformer → one
   **fused clip vector `tav`**.

- Config: `d_model=256`, 4 heads, 6 layers, 8 queries.
- **Output dimension: 256.** Exposes 7 granularities `{t,a,v,ta,tv,av,tav}`.
- **≈ 19.8M trainable parameters.**

## 7. Prediction Heads

**(a) Open-vocab head (AlignHead) — main architecture.**
`Linear(256→512) → GELU → Linear(512→1024) → L2-normalize`. No fixed-class
classifier; prediction = **cosine similarity** between the clip vector and each
emotion-**word** embedding ("anchor"), thresholded. Adding a new emotion needs no
retraining — just add its word.

**(b) Linear probe — simple baseline.**
Mean-pool + concatenate features (`1024 + 1024 + 768 = 2,816`) → single
`Linear(2816 → 138)`, sigmoid + threshold. ~0.39M params, no hidden layer.

## 8. Loss Functions

**Open-vocab (DQF) model:**
- **Main:** Multi-positive InfoNCE (contrastive) — pull each clip toward *all* its true
  emotion words, push from the rest (temperature 0.07).
- **Auxiliary:** the same loss applied to each fusion granularity (weight 0.3).
- **Optional:** description-grounding contrastive (off in the main machine runs).

**Linear model:** binary cross-entropy over the fixed emotion classes.

Model selection uses **only F_s**, never the training loss.

## 9. Evaluation Metric (F_s)

F_s = emotion-wheel-grouped set metric (Lian et al., OV-MER, ICML 2025). Maps each
predicted/true word to its wheel group (synonyms count as matches), then measures set
overlap (precision/recall) as a corpus harmonic mean. Chosen because emotions overlap
and plain accuracy would mislead.

## 10. Methodology and Results

Following advisor guidance, we switched from **cross-data validation** (train machine,
test human — a hard ceiling from label disagreement) to **within-distribution
evaluation** — train/val/test split *and* 5-fold CV on each label set separately
(checkpoint + threshold selected on validation, reported on held-out test; no leakage).

| Setup | Data | Model | F_s |
|---|---|---|---|
| **Human-aligned 5-fold CV** ⭐ | 1,532 human clips | linear probe | **44.6** |
| Machine-only 5-fold CV | 30,779 machine clips | DQF | 36.5 ± 0.5 |
| Machine-only single split | 30,779 machine clips | DQF | 36.9 |
| DQF on human data | 1,532 human clips | DQF | 34.7 |
| Cross-data (rejected) | machine → human | DQF | ~28 |

## 11. Key Findings

1. **Best result: F_s = 44.6** — clean human data, far beyond usual 6–7-class systems.
2. **Simple beats complex on small data:** on the same human set, the ~0.39M-param
   linear probe (44.6) beat the ~20M-param DQF (34.7); an MLP head also did worse
   (41.1). On limited clean data, simpler generalizes better.
3. **The bottleneck is label noise, not the model.** We ruled out:
   - **Features** — a linear probe already reaches ~44.
   - **Architecture** — the heavy DQF did not beat the linear head.
   - **Label-space coverage** — a curated 193-emotion taxonomy left F_s flat
     (36.86 → 36.88).
4. **Label-space analysis:** the wheel silently drops 68% of words (incl. "sarcasm").
   We built a curated, coverage-complete 193-emotion taxonomy (added humor/neutral/
   serious, dropped non-emotion junk); re-scoring left F_s unchanged, confirming
   coverage was not the limiter — the ceiling is **structural auto-label noise**.
5. **Training is healthy:** loss decreases monotonically and validation F_s plateaus
   with no overfitting (see `runs/*/plot.png`).

## 12. Conclusions and Next Steps

- A working lightweight, open-vocabulary, multimodal emotion recognizer at **44.6 F_s**.
- A rigorous diagnosis: the limiter is **auto-annotation label quality**, not model
  capacity or features.
- A byproduct: a clean, curated open-vocabulary emotion taxonomy.
- **Next steps:** more clean human-labeled data (the main lever), fine-tune encoders on
  emotion / add richer visual + audio, train a loss closer to F_s, and use open-vocab
  zero-shot to reach the rare tail.
