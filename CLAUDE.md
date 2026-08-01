# CLAUDE.md — DG-DQA project memory

> Read this file FIRST at the start of every session. Update it at the end of
> every prompt/milestone. Sections below are load-bearing — keep them current.

---

## 1. Project goal + one-line architecture

**Goal:** A trained, discriminative, lightweight model for fine-grained,
open-vocabulary, multi-label emotion recognition on short video clips
(text + audio + visual). "Open-vocabulary" = predict any number of emotions
per clip from a large open label space (~236+ categories, incl. subtle ones
like shy / nervous / grateful), NOT a fixed 6–7 class set. We build our own
discriminative model — we do **not** prompt a multimodal LLM for answers.

**One-line architecture (DG-DQA = Description-Grounded Dual-Query Alignment):**
Frozen cached encoders (audio=emotion2vec, visual=CLIP-ViT-L on OpenFace faces,
text=Chinese-capable encoder) → DQF fusion body (per-modality Conv1D → Q-UA →
Q-CA pairwise cross-modal → MHSA global fusion + [CLS]) → open-vocab head
projecting [CLS] to an L2-normalized text-emotion embedding, supervised by a
symmetric multi-positive contrastive loss over per-clip **descriptions** + gold
**label words**; inference = cosine-match vs. emotion-word embeddings, emit
labels above a learned threshold / predicted cardinality.

---

## 2. Environment

- **GPU:** ONE RTX 5060 Ti, 16 GB, Blackwell **sm_120**, capability `(12, 0)`.
- **PyTorch:** `torch 2.11.0+cu128`, `torch.version.cuda = 12.8` — installed from
  the **cu128 wheel**. **NEVER reinstall torch.** Verified 2026-06-18: matmul on
  the GPU works (no "no kernel image" error). The sm_120 mismatch symptom would
  be a CUDA "no kernel image is available" error.
- **Conda env:** `mer`  (visible in shell prompt `(mer) [user10@...]`).
  - Activate: `conda activate mer`
- **Code home on server:** `~/SRIP/` (isolated; this is where we work).
- **Server:** `user10@172.16.121.34`. Claude has **no direct access**; the user
  SSHes in and runs commands manually. All runnable steps are emitted in a
  code block prefixed **"Run on server:"**, one dependent step at a time, and we
  wait for pasted-back output before continuing.

---

## 3. Directory layout

```
SRIP/
├── CLAUDE.md                # this file — project memory
├── README.md
└── src/
    ├── env_check.py         # [DONE] torch/CUDA/sm_120 sanity check + matmul
    ├── metric.py            # [stub] F_s emotion-wheel-grouped set metric
    ├── data_prep.py         # [stub] parse datasets, label space, wheel groups
    ├── extract_features.py  # [stub] frozen encoders -> cached features
    ├── dataset.py           # [stub] DataLoader over cached features
    ├── models.py            # [stub] DG-DQA fusion body + open-vocab head
    └── train.py             # [stub] training loop, select on F_s only
```

**Server-side data tree (`~/SRIP/` on user10@172.16.121.34):**
```
~/SRIP/
├── src/ … (synced from Windows via scp)
├── third_party/AffectGPT/   # cloned repo (emotion_wheel + ref metric code)
├── logs/                     # dl_media.log, extract.log
└── data/
    ├── MER2026/              # HF dataset: *.csv + audio_7z/ + openface_7z/
    ├── MER2026_extracted/    # audio/<name>.wav ; openface_face/<name>/<name>.npy
    ├── MER2025/              # track3_train_mercaptionplus.csv (descriptions); OV-MERD CSVs later
    ├── manifests/            # train.jsonl, eval.jsonl, vocab.json, summary.json
    └── cache/                # FEATURE CACHE (see §5): train/<name>.pt (30779),
                              #   eval/<name>.pt (1532), {train,eval}_meta.json, vocab_emb.pt
```
Local Windows repo: `c:\Users\anura\Desktop\SRIP` (edit here, scp src/ to server).
*(Update as files/dirs are added: feature cache dir, checkpoints, etc.)*

---

## 4. Data

- **Language note:** source is **Chinese** TV clips with **English** translations.
  Text encoder must be Chinese-capable.
- **Train pool:** MER-Caption+ (large, auto-annotated, **noisy** labels).
- **Eval sets:** OV-MERD / OV-MERD+ (small, human-annotated, clean).
- **Rule:** train on MER-Caption+; evaluate/select on OV-MERD(+). **Never**
  select on the tiny clean set's own training signal.
- **Dataset file roles + exact paths:** _TBD — to be filled from Prompt 3._
- **Dataset SOURCE (identified 2026-06-19):** HuggingFace gated dataset
  **`MERChallenge/MER2025`** (`repo_type=dataset`). Confirmed file tree (38
  files): label CSVs `track{1,2,3}_train_*.csv` (track2 ≈ open-set emotion
  words, track3 ≈ descriptions, track1 ≈ discrete+dimensional),
  `track2_train_ovmerd.csv` / `track3_train_ovmerd.csv` (OV-MERD),
  `track2_train_mercaptionplus.csv` / `track3_train_mercaptionplus.csv`
  (MER-Caption+), `track_all_candidates.csv` (candidate label space),
  `subtitle_chieng.csv` (Chinese+English transcripts). Raw modalities provided:
  `audio.zip` (→ emotion2vec), `openface_face/openface_face_split.*` (pre-cropped
  OpenFace faces → CLIP), `video/video_split.*` (raw videos). **NOTE: OpenFace
  faces + subtitles + audio are pre-provided — no OpenFace run needed.** Target
  layout on server: `~/SRIP/data/MER2025/`.
- **ACCESS RESOLVED → we use `MERChallenge/MER2026`, NOT MER2025.** Account
  `varun1235` is ACCEPTED for **MER2026** (not MER2025). Logged in via
  `hf auth login` device flow. CSVs downloaded to `~/SRIP/data/MER2026/`.

- **MER2026 schema (CONFIRMED 2026-06-19) + role mapping:**
  - `track2_train_mercaptionplus.csv` (31,327; cols `name,openset`; openset is a
    list-literal e.g. `[concern, pessimism]`) → **TRAIN** (MER-Caption+, noisy).
  - `track2_train_human.csv` (1,532; cols `name,openset`; comma-sep e.g.
    `shocked,care,serious`) → **EVAL** (human-annotated open-set; the MER2026
    analog of OV-MERD — there is NO literal "ovmerd" file in MER2026).
  - `track3_candidate.csv` (10,000; cols `name,a1,a2`) → **DG-DQA descriptions**
    (two long multimodal reasoning texts per clip).
  - `subtitle_chieng.csv` (132,171; cols `name,chinese,english`) → **TEXT**
    (english frequently NaN → Chinese is primary; matches Chinese-encoder note).
  - `track1_train.csv` (9,395; cols `name,discrete`) → discrete single-label set.
  - `track3_emoprefer.csv` (574) / `track3_emopreferv2.csv` (2,096) → EmoPrefer
    preference pairs (NOT used by DG-DQA).
  - `track1_track2_candidate.csv` (20,000; `name` only) → unlabeled candidate pool.
  - Raw modalities (gated, large, split zips under `audio_7z/`, `openface_7z/`,
    `video_7z/`, per-track subfolders w/ `chunk_map.tsv`+`sample_ids.txt`;
    `extract_mer2026_archives.sh` provided). OpenFace faces + audio pre-provided.
  - `name` ids look like `samplenew3_00000058`, `samplenew4_...`, `sample_...`.

- **JOIN/COVERAGE ANALYSIS (CONFIRMED 2026-06-19):**
  - subtitle coverage: **100%** of train (31327/31327) AND eval (1532/1532).
  - english subtitles present for 119019/132171 (~90%); **Chinese = primary text
    (100%)**, English auxiliary only.
  - DG-DQA descriptions (track3): only **2725/31327 (~8.7%)** of train clips have
    one; 573/1532 eval clips. → description-contrastive is SPARSE; gold-label-word
    contrastive applies to ALL clips. DG-DQA = "label-words everywhere +
    descriptions where available."
  - **Train/eval LEAKAGE: 548** human-eval names also in mercaptionplus train.

- **MANIFESTS BUILT (2026-06-19) → `~/SRIP/data/manifests/`** by `data_prep.py`.
  JSONL per split; each record: `name, label_words[list], description[str|null],
  text_zh, text_en, audio_path, face_path, video_path(null)`.
  - **TRAIN** (`train.jsonl`): n=**30,779**; median labels/clip=**3**; desc=
    **30,779 (100%)**; zh=100%, en=98.1%; audio/face=100%.
  - **EVAL** (`eval.jsonl`): n=**1,532**; median labels/clip=**5**; desc=548
    (35.8%, irrelevant — eval unused for DG); zh=98.4%, en=98.4%; audio/face=100%.
  - **VOCAB** (`vocab.json`, from TRAIN labels): **3,311** raw emotion words
    (UN-normalized — includes morph variants anger/angry, anxiety/anxious,
    tension/tense; the wheel collapses these at scoring time, NOT here). Top:
    anger, anxiety, frustration, dissatisfaction, anxious, positive, tense,
    concern, disappointment, tension…
  - TRAIN ∩ EVAL names = **0** (separation verified).
  - **DESCRIPTION SOURCE (RESOLVED 2026-06-19):** TRAIN descriptions come from
    **MER2025 `track3_train_mercaptionplus.csv`** (`reason` column, 31,327 rows,
    **100% name match** to MER2026 mercaptionplus) at
    `~/SRIP/data/MER2025/track3_train_mercaptionplus.csv`. Pass via
    `data_prep.py --desc-csv <that path>`. Gives **100% train desc coverage** (vs
    8.1% from MER2026 track3_candidate). This is the canonical desc source now.

- **SPLIT DECISIONS (FROZEN):**
  - **EVAL = `track2_train_human` (1,532, human open-set).** Clean benchmark
    (MER2026 analog of OV-MERD).
  - **TRAIN = `track2_train_mercaptionplus` MINUS the 548 names in the human
    set (≈30,779).** Removes leakage; the 548 clips' human labels go to eval only.
  - **TEXT = `subtitle_chieng.csv` Chinese** (English auxiliary where present).
  - Media needed: track2 audio + openface for mercaptionplus & human only
    (skip video — audio+faces+subtitles suffice). Splits live under
    `audio_7z/`, `openface_7z/` per-track subfolders.
  - **MEDIA DOWNLOAD COMPLETE & VERIFIED (2026-06-19):** all 10 archives,
    108.8/108.8 GB (100%, 0 bad) at `~/SRIP/data/MER2026/{audio_7z,openface_7z}/`.
    GOTCHA: the `hf` xet client hung overnight on a dropped connection (process
    alive, bytes frozen at 94 G); killing + re-running `hf download` resumed and
    finished. If it stalls again, use `HF_HUB_DISABLE_XET=1`. `7z` installed via
    conda-forge `p7zip` (binaries `7z`/`7za`/`7zr`).
  - **EXTRACTED (2026-06-19) to `~/SRIP/data/MER2026_extracted/`** via
    `extract_mer2026_archives.sh … track2_train_mercaptionplus track2_train_human`:
    - `audio/<name>.wav` — flat, one WAV per clip.
    - `openface_face/<name>/<name>.npy` — uint8 array **[T, 112, 112, 3]**, T
      aligned RGB face frames (T varies per clip, e.g. 67/88/248). This is the
      visual input for CLIP (NOT loose images, NOT AUs/landmarks).
    - **32,311** audio files AND 32,311 openface dirs = exactly the train∪eval
      union (31,327 + 1,532 − 548). Media coverage COMPLETE.
    - Names mix `sample_*`, `samplenew3_*`, `samplenew4_*` prefixes.

- **OV-MERD / check #3 status:** OV-MERD labels/clips NOT in the AffectGPT clone
  (only loader code `ovmerd_dataset.py`). Account `varun1235` NOW has MER2025
  access. **OV-MERD CONFIRMED present in `MERChallenge/MER2025`:**
  `track2_train_ovmerd.csv` (~19 kB, `name,openset` gold) +
  `track3_train_ovmerd.csv` (~437 kB, descriptions) + `track_all_candidates.csv`
  (~400 kB, label space). **Check #3 DEFERRED** (optional metric re-validation +
  bonus eval); metric already trusted via #1/#2 + 7386/1255 count match. NOTE for
  #3: still need each OV-MERD sample's ORIGINAL single discrete label (the
  "one-hot" prediction) — likely from `track1_*` / OV-MER repo gt, TBD when run.
- **Old MELD project at `~/mer-project/` (SUPERSEDED, do not confuse):** contains
  ~21 GB of MELD (`data/MELD.Raw` + `MELD.Raw.tar.gz`), MELD feature cache
  (`cache/{train,dev,test}.pt`), and an old pipeline (`src/compare.py`,
  `extract_features.py`, `models.py`, `env_check.py`). Wrong dataset + wrong
  label space for us — NOT reusable as cache, but the old `extract_features.py`
  is a useful *reference template* for the frozen-encoder→`.pt` pattern. Can be
  deleted later to reclaim ~21 GB; kept for now. Disk: 372 GB free.

---

## 5. Data contract (cache format) — **SACRED, never change silently**

The cache is the fixed interface between `extract_features.py`, `dataset.py`,
and `train.py`. Any change is a deliberate, documented version bump.

**Frozen encoders (run once, cached) + dims (CONFIRMED 2026-06-19 on GPU):**
- audio  = emotion2vec  `iic/emotion2vec_base` (FunASR), frame-level → **d=768**
- visual = CLIP-ViT-L/14 `openai/clip-vit-large-patch14`, projected image
  embedding per sampled face → **d=768** (vision pooler is 1024; we use the 768
  projected)
- text   = BGE-M3 `BAAI/bge-m3`, token states (sequence) **d=1024**, and pooled
  CLS (L2-normalized) for the description anchor + emotion-word targets.

**Representation = SEQUENCES (not pooled), so DQF Conv1D/Q-UA/Q-CA have a time
axis.** Pooled vectors are derivable later; never cache only pooled.

**Layout — PER-CLIP files (resumable; Dataset pads per batch):**
```
cache/<split>/<name>.pt        # torch.save(dict), tensors:
  name        : str
  X_audio     : float16 [T_a, 768]     emotion2vec frames (T_a≤512, uniform subsample)
  X_visual    : float16 [N_f, 768]     CLIP per-face   (N_f≤16, uniform sample)
  X_text_zh   : float16 [T_zh, 1024]   BGE-M3 tokens (Chinese subtitle, ≤128 tok)
  X_text_en   : float16 [T_en, 1024]   BGE-M3 tokens (English subtitle, ≤128 tok)
  desc_emb    : float32 [1024]          BGE-M3 pooled desc (L2-norm) — TRAIN ONLY
  present     : {audio,visual,zh,en[,desc] : bool}   # missing → zeros[1,d], flag False
  label_words : list[str]
cache/<split>_meta.json        # n, new, skipped, missing counts, dims, caps, model ids
cache/vocab_emb.pt             # {vocab: list[str](3311), word_emb: float32 [V,1024]}
                               #   open-vocab match target; L2-normed; built once
```
**Masks** are derived from each sequence's length (no padding stored; the
Dataset pads + builds masks per batch). **Missing modality** = a `[1,d]` zero
tensor + `present[...]=False` (no video fallback — video not downloaded; faces
are 100% present anyway). Caps are `extract_features.py` args
(`--max-audio-frames 512 --n-faces 16 --max-text-tokens 128 --max-desc-tokens 256`).

---

## 6. Metric — F_s (emotion-wheel-grouped set metric)

**THE ONLY model-selection metric.** Never select on accuracy, BCE loss, or
training loss. Implemented in `src/metric.py` (port of Lian et al., OV-MER
ICML 2025, §3). Public API: `score(pred_lists, gold_lists, wheel_path)`.

**Per-sample set scores** (after mapping each word to a group id, dedup to sets
gold `Y`, pred `P`):
- `precision_s = |Y∩P| / |P|`,  `recall_s = |Y∩P| / |Y|`.
- Per the official code (`calculate_openset_overlap_rate`): **skip the sample if
  `Y` is empty**; if `P` is empty after mapping, that sample's precision &
  recall = 0.

**AGGREGATION — corpus harmonic mean (NOT mean of per-sample f):**
- `P = mean over samples of precision_s`, `R = mean over samples of recall_s`.
- `F_s = 2·P·R / (P + R)`  (0 if P+R == 0).
This is why HM(92.2, 51.1) = 65.75 ≈ the 65.7 one-hot anchor. The earlier
"mean of per-sample f_s" reading was WRONG; the paper combines corpus means.
Single-sample unit tests are unaffected (corpus-HM == per-sample f for n=1):
exact → 1.0; disjoint → 0.0; pred `{a}` vs gold `{a,b}` → P 1.0 / R 0.5 / F 2/3.

**Grouping (word → group_id) — ported verbatim from AffectGPT
`my_affectgpt/evaluation/wheel.py` (`func_backward_case{1,2,3}`).** Two maps:
- `format_mapping` from `format.csv` (`read_format2raws`): surface form →
  list of base word-forms (level3→level2). Self-maps each base.
- `raw_mapping` from `synonym.xlsx` (`read_candidate_synonym_merge`, 8 runs
  merged): base word → list of canonical wheel labels (level2→level1). Synonym
  `s` of canonical `c` adds `s→c`; canonical self-maps `c→c`.
- `wheel_map` from `wheel{1..5}.xlsx` (`func_get_wheel_cluster`): canonical
  label → wheel cluster center (level1 or level2).
Backward (each picks `sorted(...)[0]` for determinism):
- **case1** (Level 1, word-form): word → smallest base form.
- **case2** (Level 2, +synonym): word → base form → smallest canonical label.
  *(This is what makes joyful→happy match; the level we SELECT on.)*
- **case3_wheelN_levelL** (Level 3, +wheel): word → all canonicals → first in
  `wheel_map` → its cluster. Reported as the mean over the 5 wheels.

**OOV handling — DECISION: report BOTH.**
- **PRIMARY (select on this):** drop OOV. `func_backward` returns `""` for any
  word not in `format_mapping`; `func_map_label_to_synonym` removes those from
  BOTH pred and gold (paper-faithful; required to reproduce the anchor).
- **SECONDARY (logged diagnostic):** singleton variant — keep OOV words as
  their own group (stricter open-vocab eval). Never used for selection.

**DECISION:** each eval reports Level-2 (case2, **selection metric**) and
Level-3 (case3 mean over 5 wheels, logged), each under drop (primary) and
singleton (diagnostic).

**Wheel-file path (server, CONFIRMED):**
`~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel/` (shallow git clone of
github.com/zeroQiaoba/AffectGPT). Contains `format.csv`, `synonym.xlsx`,
`wheel{1..5}.xlsx`. Reference metric code (ported from):
`OV-MER/my_affectgpt/evaluation/wheel.py`. Requires `openpyxl` in the `mer` env
(installed 2026-06-18; pure-python, does NOT touch torch).

**Port validation:** loader builds **7386 surface forms** and **1255 words**,
exactly matching AffectGPT's own counts → port is faithful.

**Acceptance:**
1. set-math unit tests — ✅ PASSED (`--selftest`).
2. synonym test: pred "joyful" vs gold "happy" — ✅ PASSED; both → canonical
   `joy` at case2; f_s=1.0.
3. one-hot upper bound on English OV-MERD: F_s ≈ 65.7, Precision_s ≈ 92.2,
   Recall_s ≈ 51.1 (±~1 pt) — ⏳ BLOCKED: OV-MERD gold openset + original single
   label are NOT in the repo (only `output/results-ovmerd/` model outputs). Need
   the OV-MERD dataset release. Reproduction wired but unrun.

---

## 7. Architecture decisions + rationale

- **Frozen encoders, features cached once:** 16 GB GPU can't fine-tune large
  encoders E2E; freezing + caching makes training cheap and reproducible.
- **Open-vocab contrastive head (Option B) over closed BCE head:** a fixed
  BCE classifier can't scale to a 236+ open label space or generalize to unseen
  emotion words; an L2-normalized text-emotion embedding + cosine matching can.
- **emotion2vec for audio:** speech-emotion-pretrained representation; audio is
  the weakest modality so we want the strongest available frozen features.
- **CLIP-ViT-Large (visual) on pre-cropped OpenFace faces:** face crops focus
  the encoder on expression; CLIP gives a text-aligned visual space.
- **Chinese-capable text encoder:** data is Chinese with English translations.
- **DQF fusion body:** dual-query (Q-UA unimodal refine + Q-CA cross-modal),
  multi-granularity aux losses, random per-modality masking for missing-modality
  robustness.

**DQF body — IMPLEMENTED `src/models.py` (Prompt 5, 2026-06-19):**
- Operates on per-modality SEQUENCES (the §5 cache), NOT pooled vectors —
  length-1 / missing modality is the degenerate case (a `[B,1,d]` zero token).
- `DQFBody(d_text=1024, d_audio=768, d_visual=768, d_model=256, n_heads=4,
  n_layers=6, n_query=8, global_layers=2)`. ~**19.8M trainable params**
  (frozen encoders excluded). Defaults = DQF "6 layers / 4 heads" optimum.
- Per modality: `Linear -> Conv1d(k3, temporal) -> QUA`. Padded positions zeroed
  before conv; attention uses `key_padding_mask`.
- **QUA**: `n_query` learnable queries cross-attend to features (MHA) + LayerNorm
  residual → broadcast-add summary to features → `n_layers` TransformerEncoder.
  Outputs contextual seq + masked-mean summary (the unimodal granularity feat).
- **QCA** per pair {t-a,t-v,a-v}: one shared learnable query attends to each
  modality (both directions) → mean-pool → concat → Linear+LN → pair feature.
- **Global**: `[CLS] + 3 pair feats` → `global_layers` TransformerEncoder →
  fused `[CLS]` (= `tav`).
- `forward(feats, pads, modality_mask=None)` returns granularity dict
  **{t,a,v, ta,tv,av, tav}** each `[B, d_model]` — hooks for Prompt 6 aux losses.
  `modality_mask [B,3]` zeros whole modalities; `random_modality_mask()` helper.
- INTERFACE NOTE: body is text-modality-agnostic — caller passes ONE text stream
  (zh by default; zh/en/merge chosen upstream). Open-vocab head (project `tav`
  → 1024 text-emotion space, L2-norm) + losses come in Prompt 6.
- GOTCHA: `enable_nested_tensor=False` on TransformerEncoders (norm_first=True
  warning). Use `torch.load(weights_only=False)` for the cache.

---

## 8. STATUS / PROGRESS LOG

- **2026-06-18 — scaffold created.** Directory layout + 7 src modules
  (env_check implemented, rest stubbed), CLAUDE.md (all 10 sections), README.md.
  **Check:** run `python src/env_check.py` on the server → expect `(12, 0)`,
  GPU name, and a matmul result with no error. **Next:** Prompt 2 (environment /
  encoder setup), then define data paths + data contract (Prompt 3).
- **2026-06-18 — server inventoried + code uploaded.** `~/SRIP/` now has
  `src/`, `CLAUDE.md`, `README.md`. Confirmed conda env = `mer`. Found old MELD
  project at `~/mer-project` (superseded; see §4). Confirmed MER-Caption+ /
  OV-MERD data is **not yet on the server** → dataset acquisition needed before
  extraction. `env_check.py` not yet run on GPU (pending). **Next:** run
  `python src/env_check.py`, then plan dataset download.
- **2026-06-18 — env_check PASSED (Prompt 1 complete).** `python src/env_check.py`
  on GPU: torch 2.11.0+cu128, CUDA 12.8, capability (12,0), RTX 5060 Ti, matmul
  OK, exit 0. Environment fully verified. **Next milestone:** acquire MER-Caption+
  (train) + OV-MERD/OV-MERD+ (eval) datasets onto `~/SRIP/data/`, then define the
  data contract (§5) and write data_prep.py / extract_features.py.
- **2026-06-18 — metric.py: set-math core done & verified (Prompt 2, in
  progress).** Implemented `src/metric.py` (OV-MER set metric). Set-math core
  passes `--selftest` locally (check #1 ✅). Researched the real AffectGPT
  `emotion_wheel/` layout: `format.csv` (word-form, confirmed), `synonym.xlsx`
  (synonyms), `wheel{1..5}.xlsx` (wheel groups). Wheel loader implemented with
  auto-detected xlsx columns — **pending validation against the real files**.
  Decision: report both Level-2 (select) + Level-3 (log). **Next:** clone repo
  on server, confirm synonym.xlsx/wheel1.xlsx schema, pass synonym test (#2),
  then wire one-hot OV-MERD reproduction (#3) once OV-MERD gold is available.
- **2026-06-19 — Prompt 4: extract_features.py written, smoke PASSED.** Encoders
  pinned + verified on GPU (emotion2vec 768, CLIP-ViT-L 768 proj, BGE-M3 1024).
  funasr installed (torch untouched). `--limit 16` eval smoke: 16 per-clip caches
  + `vocab_emb.pt [3311,1024]` written; shapes match §5 contract. Data contract
  frozen in §5. NOTE: `torch.load(..., weights_only=False)` required (cache dicts
  hold non-tensor fields). **Next:** full extraction (eval ~1.5K then train ~30K,
  resumable, background), then optional CLIP/text batching if throughput slow.
- **2026-06-20 — Prompt 6 GATE: FAIL at ~28 F_s. Root cause = TRAIN/EVAL LABEL
  MISMATCH, not features.** Implemented `dataset.py`, `train.py` (AlignHead +
  multi-positive InfoNCE over emotion-word anchors + DQF multi-granularity aux +
  random modality masking; threshold + checkpoint selected on a held-out TRAIN
  slice, OV-MERD used ONLY for the final number). Also `linear_probe.py`.
  Results (OV-MERD = `track2_train_human`, 1,532, F_s case2/drop):
  - word-level anchors (3,311): OV F_s **28.3**; canonical anchors (190): **28.1**.
  - DIAGNOSTIC CHAIN: label-space oracle ceiling ~**100**; dumb-prior ~**15**;
    linear probe on frozen pooled feats (trained on AUTO labels) **27.8**
    (text 24.5 / visual 21.9 / audio **17.8** ≈ prior → audio nearly dead);
    **auto-vs-human label agreement F_s = 27.1** (P35/R22 on the 548 shared
    clips); **human-only 5-fold CV linear probe = 41.7 ± 1.1**.
  - INTERPRETATION: our model already SATURATES the auto→human label-transfer
    ceiling (~27). Features are fine (support ~42 under consistent labels). The
    cap is that MER-Caption+ auto openset labels ≠ human OV-MERD labels (only 27
    F_s agreement; spot-checks show genuinely different/ richer human labels).
  - Per the gate, did NOT add DG description-grounding. NEXT = fix the TRAINING
    SIGNAL (not encoders): see options in the chat / GOTCHAS. Threshold/cardinality
    is NOT the issue (OV oracle-thr ≈ val-thr).
- **2026-06-20 — DECISION POINT: paused to rethink scope with advisor.** Full
  pipeline built + validated (Prompts 1–6). Gate did not clear 50; rigorous
  diagnosis shows WHY (see chain below). Best result = **44.6 F_s human-CV**
  (up from the 28 gate). OPEN STRATEGIC QUESTIONS for advisor:
  1. **Eval protocol** — "train auto MER-Caption+ / eval human OV-MERD" has a
     hard ceiling of **27** (auto↔human label agreement). Keep it (accept ~28),
     or switch to human-aligned training (human-CV ≈ 44.6, but only 1,532 clips)?
  2. **Discriminative vs generative** — published ~50–65 use generative MLLMs on
     DESCRIPTIONS; we chose lightweight discriminative on frozen feats (≈44.6).
     Is the goal still a lightweight discriminative model?
  3. **Data** — is more human-labeled open-vocab MER data available beyond 1,532?
     That's the only real lever left toward 50.
  4. **DG contribution** — description-grounding did NOT help (descriptions are
     auto-derived → same distribution; zero-shot stayed ~27). Rethink "DG", or is
     the contribution now the analysis + the human-aligned recipe?
  5. **Deliverable** — SOTA number, a method, or the finding itself (auto-openset
     labels are a poor training target; ceiling=27)? The analysis is publishable.
  All code/cache/results reproducible. Resume after advisor input.
- **2026-06-20 — Feature upgrade validated (human-CV): 41.7 → 44.6.** Re-encoded
  the 1,532 human clips with **emotion2vec_plus_large** (audio, d=1024) +
  **FER-ViT** backbones (`trpakov`/`dima806`/`motheecreator`, d=768 CLS) via
  `probe_new_features.py`; reused BGE text. Human-CV linear probe: text 37.0,
  **audio(plus_large) 38.7** (the base-audio "17.8 dead" was an AUTO-label
  artifact — under human labels big audio is the STRONGEST single modality!),
  visual(FER) ~34.5, **combined linear 44.6**. Head tweaks ALL WORSE: MLP 41.1,
  L2-norm 32, 3-FER-ensemble 34 → classic small-data overfit. CONCLUSION: the
  binding constraint is now the **1,532-clip human set size**, not features/head.
  Path to ~50 = MORE human-aligned data (e.g. add MER2025 OV-MERD/OV-MERD+),
  not more model. Upgraded encoders (plus_large + a FER) are worth adopting.
  NOTE: full 32K re-extraction only needed if we resume pool-pretraining (which
  HURT); human-CV uses only eval clips, already cached in `pooled_new_eval.pt`.
- **2026-06-20 — Hybrid (DG-pretrain pool → human-CV finetune): 34.7 ± 0.6.**
  `train_hybrid.py`: pretrain DQF on 30K pool (InfoNCE over 190 canonical anchors
  + multi-granularity aux + DG description-contrastive on `desc_emb`), then 5-fold
  fine-tune+eval on human. Pretrain zero-shot on human ~27 (DG did NOT lift past
  the gate — descriptions are auto-derived, same distribution). Full fine-tune →
  **34.71 ± 0.62**, which is BELOW the **41.7 human-CV linear probe**. CONCLUSION:
  the auto-pretrained 256-d DQF `tav` bottleneck loses human-relevant info + the
  20M body overfits ~980 fine-tune clips. The heavy fusion does NOT beat a linear
  probe on frozen pooled features. Current best = **41.7 (linear, frozen feats,
  human-CV)**. Path to ~50 is better FEATURES (audio≈dead) or more human data,
  NOT a heavier head. **NEXT decision: see chat.**
- **2026-06-19 — Prompt 5 COMPLETE: DQF fusion body.** `src/models.py` —
  `DQFBody` over per-modality sequences (Q-UA → Q-CA pairs → global [CLS]),
  ~19.8M trainable params, returns {t,a,v,ta,tv,av,tav}. Self-check passes
  (full + drop-1 + drop-2 + random-mask forward, all `[B,256]`, no NaN). Details
  in §7. **Reconciled Prompt 5's "length-1" framing with the §5 sequence cache:
  body uses sequences (length-1 is the degenerate case).** **Next: Prompt 6 —
  open-vocab head (project tav→1024 L2-norm) + DG-DQA losses** (symmetric
  multi-positive contrastive on desc_emb + vocab_emb, multi-granularity aux,
  BCE/cardinality) using `vocab_emb.pt` + `desc_emb`.
- **2026-06-19 — Prompt 4 COMPLETE: feature cache built + validated.** EVAL
  1,532 (missing zh23/en24) + TRAIN 30,779 (missing en588 only; audio/visual/zh/
  desc all 0) cached at `cache/{eval,train}/<name>.pt`; `vocab_emb.pt [3311,1024]`
  built. Validation pass: **0/32,311 corrupt** (load + shape check). Train clips
  carry `desc_emb` (100% desc coverage). GOTCHA logged: train launch fired TWICE
  (two bg processes raced ~22 min); killed the dup; validation confirmed no
  corruption. Throughput ~4 clips/sec (emotion2vec unbatched = floor). **Next:
  Prompt 5 — Dataset/DataLoader** over the cache (ragged→padded batches, per-mod
  masks, random missing-modality masking). Raw `*_7z` zips (~105 GB) now
  redundant → can delete to reclaim space (extracted media kept for re-extract).
- **2026-06-19 — Prompt 3 COMPLETE: data_prep + media ready.** MER2026 media
  (108.8 GB audio+openface) downloaded, verified (100%), extracted to
  `data/MER2026_extracted/` (audio WAV + openface `[T,112,112,3]` uint8 face npy;
  32,311 clips = train∪eval). `data_prep.py` built manifests: TRAIN 30,779 (median
  3 labels), EVAL 1,532 (median 5), vocab 3,311, audio/face/zh coverage ~100%,
  TRAIN∩EVAL=0. OPEN: description coverage only 8.1% on train (consider MER2025
  full descriptions). **Next:** define DATA CONTRACT (§5) + write
  `extract_features.py` (emotion2vec audio, CLIP-ViT-L on faces, Chinese text).
- **2026-06-18 — metric.py grouping ported & validated (Prompt 2, checks 1&2
  done).** Rewrote `src/metric.py` as a faithful standalone port of AffectGPT
  `wheel.py` (`func_backward_case{1,2,3}`, corpus-HM aggregation). `WheelGrouper`
  loads format.csv + synonym.xlsx + wheel*.xlsx once. Verified on server:
  set-math ✅, synonym joyful↔happy→`joy` ✅ (f_s=1.0); loader counts 7386/1255
  match the paper. API: `score()` (case2/drop = selection), `score_report()`
  (case2+case3 × drop+singleton). **Only check #3 (one-hot reproduction)
  remains, BLOCKED on OV-MERD gold labels (not in repo).** **Next:** get OV-MERD
  + MER-Caption+ data onto server; then run #3 and freeze §6; then Prompt 3
  (data contract + data_prep).
- **2026-06-25 — ADVISOR PIVOT: drop cross-data eval; do WITHIN-machine CV +
  train/val/test + DQF + TRAINING PLOTS.** Advisor rejected the "train auto
  MER-Caption+ / eval human OV-MERD" protocol (= cross-data validation; hard cap
  ~28 from auto↔human label disagreement). New ask: evaluate WITHIN the machine
  label distribution — same 5-fold CV we used on the human set, but on the 30,779
  machine clips — plus a proper train/val/test split, using the heavy DQF (plenty
  of data → no small-data overfit). Also: training plots for all runs.
  WROTE 3 scripts (local repo, PENDING server validation):
  - `src/train_machine_cv.py` — DQFBody+AlignHead FROM SCRATCH on machine clips;
    `--mode split` (80/10/10) or `--mode cv --folds 5`; selects ckpt+thr on VAL
    F_s only, reports TEST F_s mean±std; saves `runs/<name>/{history,result}.json`.
  - `src/plot_training.py` — loss + val-F_s curves → PNG (Agg/headless); single
    run, CV fold-overlay, and `--compare` across runs.
  - `src/parse_logs.py` — reconstruct `history.json` from OLD train.py /
    train_hybrid.py STDOUT logs (they never saved history) so prior runs plot
    w/o re-running. Regexes unit-tested locally; namespaces fold series by kind.
  CAVEAT TO STATE: machine-CV measures "predict the AUTO labels" (within-dist),
  NOT human emotions (still human-CV ≈44.6). Report both side by side.
  Needs matplotlib in `mer` (conda-forge; does NOT touch torch). New dir:
  `~/SRIP/runs/`. NEXT: scp 3 files, check `logs/` for old stdout + matplotlib,
  smoke `train_machine_cv.py --mode split --epochs 3 --limit 2000`, then real runs.
- **2026-06-28 — Machine within-dist DQF runs DONE + all prior runs plotted.**
  matplotlib installed in `mer`. Old stdout logs existed in `~/SRIP/logs/`
  (`gate_zh_s0.log`, `gate_zh_canon_s0.log`, `hybrid_zh.log`) → `parse_logs.py`
  reconstructed them into `runs/{gate_zh_word,gate_zh_canon,hybrid_zh}/` + PNGs
  (regexes matched the real log format exactly). New DQF results (from scratch,
  30,779 machine clips, canonical anchors, zh, 20 ep, select ckpt+thr on VAL F_s):
  - **`machine_split` (80/10/10): TEST F_s = 36.86** (best_val 36.69 @ep12).
  - **`machine_cv` (5-fold): TEST F_s = 36.51 ± 0.53** `[36.8,36.9,35.7,36.0,37.1]`.
  COMPARISON SET for advisor: cross-data (old, rejected) ~28 → within-machine
  split 36.9 / 5-fold CV 36.5±0.5 → human-CV 44.6 (reference, different question).
  Within-machine ≈ +9 over cross-data = the payoff of consistent labels (the whole
  point of the pivot). Plots in `runs/<name>/plot.png`. Curves healthy (loss
  monotone, val F_s plateaus ~36, no overfit). VERIFIED human-CV split hygiene:
  `probe_new_features.py` cv_probe does clean fit/val/test per fold (thr on val,
  score on test, no leakage) → 44.6 is leakage-free. CAVEAT: demo model's 47.87
  (`train_demo_model.py`) is VAL-ONLY (trains on all 1,532, thr on 20% held-out) —
  a deployable artifact, NOT a clean test number; cite 44.6 as the human eval.
  - **NO trained model checkpoints are saved anywhere** — train.py/train_hybrid.py/
    train_machine_cv.py all keep best-by-val state in MEMORY then discard on exit.
    The ONLY saved model on disk is `data/cache/demo_model.pt` (linear 2816→138
    demo head, thr=0.2, from `train_demo_model.py`; demo.py/demo_timeline.py load
    it). To persist a DQF checkpoint, must ADD torch.save to train_machine_cv.py
    (was drafted then reverted per user — re-add deliberately when keeping a model).
  - LOCAL BACKUP at `c:\Users\anura\Desktop\SRIP\server_backup\`: `runs/` (all
    plots+json) pulled; `demo_model.pt` pulled. GOTCHA: cache is at `data/cache/`
    NOT `cache/` — earlier essentials-tar used `cache/...` so it MISSED
    vocab_emb/anchors/pooled/meta (harmless for now; fix path if re-backing-up).

---

## 9. GOTCHAS

- **sm_120 torch:** must use the cu128 wheel; a "no kernel image" RuntimeError
  means a build mismatch. **Never reinstall torch** to "fix" it blindly.
- **MER-Caption+ labels are noisy** (auto-annotated) — expect label noise; do
  not over-trust per-clip gold on the train pool.
- **Audio is the weakest modality** — confirmed 2026-06-20: emotion2vec_base
  linear probe ≈ 17.8 F_s, barely above the 15 dumb-prior. Visual (CLIP-on-faces)
  21.9, text (BGE/zh subtitle) 24.5. If we later need stronger features, audio
  (emotion2vec_plus_large) and visual (a face-specific FER encoder vs CLIP) are
  the upgrades — but per the gate diagnosis features are NOT the current blocker.
- **KEY CONSTRAINT (2026-06-20): MER-Caption+ auto openset labels are a POOR
  training target for human OV-MERD** — they agree at only **F_s≈27**. Training a
  discriminative model on the auto openset labels caps OV-MERD F_s at ~27–28 no
  matter the encoder/fusion quality. Human-only CV linear probe reaches 41.7, so
  the path to higher F_s is a HUMAN-ALIGNED training signal (fine-tune on human
  folds; and/or use the richer `reason` DESCRIPTIONS, which motivates DG-DQA),
  NOT bigger encoders. Re-think the Prompt-0 premise "train auto / eval human".
- **OpenFace face paths** — visual features come from pre-cropped face images;
  path/availability per clip must be verified during feature extraction.
- **`7z` needed for media extraction** (the dataset's `extract_mer2026_archives.sh`
  uses `7z x` on the `*_split*.zip` chunks). Not present by default in `mer`;
  install via `conda install -c conda-forge p7zip` (standalone CLI, does NOT
  touch torch). Confirm binary name (`7z` vs `7za`).
- **Harmless `conda activate mer` warning:** `RequestsDependencyWarning: urllib3
  (2.6.3) or chardet/charset_normalizer doesn't match a supported version`.
  Cosmetic (requests version skew), unrelated to torch — ignore it.

---

## 10. Conventions

- **Select checkpoints ONLY on F_s.** Never on accuracy / BCE / training loss.
- **Never reinstall torch** (cu128 wheel is intended).
- **The cache is sacred** — the data contract (§5) never changes silently.
- **Train on MER-Caption+, evaluate on OV-MERD(+).** Never select on clean set.
- Code style: **argparse only** (no config frameworks, no web UI), type hints,
  short docstrings, incremental builds with a runnable self-check per module.
