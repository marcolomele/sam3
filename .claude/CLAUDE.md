# CLAUDE.md — Developer Context for Claude Code

This file provides persistent context for Claude Code sessions. Keep it updated after every session.

---

## Project Overview

VLM + SAM3 object correspondence pipeline. Given a source ego/exo frame with an annotated object, find and segment the same object in a destination frame from a different viewpoint (ego↔exo).

**Pipeline**: Source frame(s) → VLM (qwen3-vl) → text description or bbox → SAM3 → segmentation mask → IoU vs ground truth

---

## Repository Structure

```
reluminati-research/
├── README.md
├── CLAUDE.md                          ← this file
├── notebooks/
│   ├── baseline-overview.ipynb        ← dataset stats (Marco)
│   ├── experiment-analysis.ipynb      ← Marco's analysis notebook
│   ├── results_analysis.ipynb         ← generic/old analysis
│   ├── results_analysis_463119.ipynb  ← job 463119 analysis (gio)
│   └── VLM_SAM3_Experiments.ipynb     ← old experiments
└── src/scripts/
    ├── data/                          ← dataset processing scripts
    └── experiments/vlm-sam3/
        ├── experiment.py              ← MAIN PIPELINE (~1050 lines)
        ├── config.json                ← active config (skip-worktree, local)
        ├── config.box.json            ← box-only smoke config (skip-worktree)
        ├── config.multiframe.json     ← multiframe ablation config (skip-worktree)
        ├── config.cloud_only.json     ← cloud-only config (skip-worktree)
        ├── smoke.sbatch               ← main SLURM script (skip-worktree)
        ├── smoke_box.sbatch           ← box experiments SLURM script (skip-worktree)
        ├── smoke_multiframe.sbatch    ← multiframe SLURM script (skip-worktree)
        ├── create_multiframe_csv.py   ← generates baseline_multiframe_w10.csv
        └── export_to_gsheets.py       ← export results to Google Sheets
```

**Note**: All `config*.json` and `smoke*.sbatch` files are git skip-worktree (local-only, not pushed). They contain API keys.

---

## HPC Access

```bash
ssh <your_username>@lnode01-da.hpc.unibocconi.it
```
Each team member has their own username (3164542 = Giovanni, others differ). Always use `/data/video_datasets/reluminati/` for **shared outputs** — avoids manual cp between users.

## Data Paths

| Path | Description |
|------|-------------|
| `/data/video_datasets/3321908/output_dir_all/` | Dataset root: 180 takes, 11GB |
| `/data/video_datasets/reluminati/` | **Shared team folder** — save results here |
| `/data/video_datasets/reluminati/evaluation/` | Shared evaluation outputs |
| `/data/video_datasets/reluminati/LM-EEC/` | LM-EEC project data |
| `/data/video_datasets/3164542/baseline_multiframe_w10.csv` | Current baseline: 191 streams × 10 frames = 1910 rows |
| `/data/video_datasets/3164542/lm_eec_pairs_from_vlm460850.csv` | Old baseline (253 pairs) |
| `/data/video_datasets/3164542/results_partial.csv` | Partial results accumulation (Giovanni only) |

**Data layout**: `root/{take_uid}/{camera}/{frame}.jpg` — flat JPEGs directly in camera folder (not nested under object_name/rgb/)

**Image sizes**: 448×448 (aria/egocentric), 796×448 (exo) — heavily downsampled from original EgoExo4D 1404×1404

**GT mask resolution**: masks in annotation.json are stored at 1408×1408 (original resolution). Always resize to image size before indexing:
```python
if mask.shape != (H, W):
    mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST)).astype(bool)
```

**Baseline**: 50 balanced takes, 10 per scenario (basketball, bike repair, cooking, health, music)

---

## Baseline Comparison (LM-EEC)

| Direction | IoU |
|-----------|-----|
| ego→exo | 0.569 |
| exo→ego | 0.660 |

---

## HPC / SLURM

- **Partition**: `long_gpu` (14h limit) — use this, NOT `gpu`
- **Resources**: `--gres=gpu:1 --cpus-per-task=4 --mem=16G`
- **Submit**: `cd src/scripts/experiments/vlm-sam3 && sbatch smoke.sbatch`
- **Monitor**: `squeue -u 3164542`, `sacct -u 3164542 --format=JobID,JobName,State,Elapsed,ExitCode`
- **Logs**: `smoke_{JOBID}.out`, `smoke_{JOBID}.err` (in vlm-sam3 directory)
- **Output CSV**: `/data/video_datasets/3164542/results_{JOB_ID}.csv`

---

## Experiment Design

### Completed Jobs

| Job ID | Config | Status | Notes |
|--------|--------|--------|-------|
| 460850 | old/manual | COMPLETED | 253 pairs, old code |
| 462563 | config.json (10 exps) | COMPLETED | 1100 pairs, 11000 rows |
| 463119 | config.json (4 exps: C/cot/ctx/ctx-cot) | FAILED at 75% | Crash at .mean() due to resume dtype bug |
| 465254 | config.json (4 exps resume) | COMPLETED | Finishes remaining 477 pairs → 2493 total per exp |
| 465270 | config.box.json | CANCELLED | Broken 1000-scale bbox bug, results discarded |
| 465278 | config.multiframe.json | CANCELLED | Contaminated with bad box rows, results discarded |
| 465289/93/94 | config.box.json | CANCELLED | Still broken 1000-scale, results discarded |
| 465301-303 | config.json / box / mf | CANCELLED | Monitor false-positive killed at pair 371 / 126 |
| **465388** | config.box.json | **COMPLETED** | EXP-C-box/cot, 1433 pairs, 3h25m |
| **465389** | config.multiframe.json | **COMPLETED** | EXP-F-mf1/3/5/10, 1433 pairs, 2h02m |

### Final Results (all jobs deduplicated, 2026-03-09)

| Experiment | N | iou>0% | mean IoU (pos) | iou>0.5% | iou>0.75% |
|---|---|---|---|---|---|
| **LM-EEC baseline (exo→ego)** | — | — | **0.660** | — | — |
| EXP-C (cloud, text) | 2493 | 31.7% | 0.686 | 78.1% | 66.7% |
| EXP-C-cot (instruct, text) | 2493 | 41.1% | 0.580 | 65.0% | 55.6% |
| EXP-C-ctx (cloud, ctx) | 1433* | 18.6% | 0.617 | 71.5% | 61.0% |
| EXP-C-ctx-cot (instruct, ctx) | 1433* | 24.8% | 0.571 | 64.8% | 54.6% |
| **EXP-C-box (cloud, bbox)** | 1433* | **49.4%** | 0.357 | 35.6% | 21.3% |
| EXP-C-box-cot (instruct, bbox) | 1433* | 30.4% | 0.320 | 31.7% | 18.3% |
| EXP-F-mf1 (1 frame) | 1433* | 33.1% | 0.527 | 58.6% | 49.4% |
| EXP-F-mf3 (3 frames) | 1433* | 23.4% | 0.620 | 69.9% | 57.9% |
| EXP-F-mf5 (5 frames) | 2493 | 17.6% | 0.589 | 66.4% | 56.8% |
| EXP-F-mf10 (10 frames) | 1433* | 21.7% | 0.572 | 65.0% | 55.3% |

*partial (~75% of full 1910 pairs)

### Current Experiment IDs (config.json)

| ID | Model | Images | Prompt | Notes |
|----|-------|--------|--------|-------|
| EXP-C | 235b-cloud | src_clean, src_overlay, dst | clean-overlay-dst | 31.7% success |
| EXP-C-cot | 235b-instruct | src_clean, src_overlay, dst | clean-overlay-dst | **41.1% success**, 0% missing — best text |
| EXP-C-ctx | 235b-cloud | src_clean, src_overlay, dst | neighbor-context | 18.6% — WORSE (SAM3 token truncation) |
| EXP-C-ctx-cot | 235b-instruct | src_clean, src_overlay, dst | neighbor-context-cot | 24.8% — worse than cot |
| EXP-C-box | 235b-cloud | src_clean, src_overlay, dst | localize-bbox | **49.4% success — highest localization rate!** |
| EXP-C-box-cot | 235b-instruct | src_clean, src_overlay, dst | localize-bbox-cot | 30.4% — CoT hurts bbox |

### Multiframe Ablation (config.multiframe.json)

| ID | num_frames | iou>0% | mean_pos | Notes |
|----|-----------|--------|----------|-------|
| EXP-F-mf1 | 1 | 33.1% | 0.527 | best localization |
| EXP-F-mf3 | 3 | 23.4% | **0.620** | best quality when found |
| EXP-F-mf5 | 5 | 17.6% | 0.589 | |
| EXP-F-mf10 | 10 | 21.7% | 0.572 | |

**Surprise**: fewer frames = better localization. More context images confuse the VLM.

---

## Key Findings So Far

1. **EXP-C-box has highest localization (49.4%)**: VLM→bbox→SAM3 geometry prompt finds objects more often than text approach. But IoU quality is lower (0.357 vs 0.686) — SAM3 box prompts are less precise than text prompts when text works.
2. **EXP-C (text, plain) is highest quality when it works**: mean IoU=0.686 on success, 78.1% have IoU>0.5 — **beats LM-EEC (0.660)**!
3. **Main failure mode is VLM localization**: 50-68% of pairs fail because VLM can't describe/find the object, not because SAM3 segments badly.
4. **CoT hurts bbox**: EXP-C-box-cot (30.4%) << EXP-C-box (49.4%). Reasoning causes second-guessing of coordinates.
5. **Neighbor context hurts**: -13pp vs EXP-C. More tokens → SAM3 truncation kills description quality.
6. **Multiframe: 1 frame best for localization, 3 frames best for quality**: More images confuse VLM localization. When 3 frames do succeed, mean IoU=0.620 (best across mf ablation).
7. **SAM3 hard 28-token limit**: max_position_embeddings=32, ~28 usable tokens. Long CoT outputs get truncated.
8. **Per-scenario EXP-C-box**: Music 96.9% (piano/guitar easy), Basketball 71.2%, Bike 57.6%, Health 46.3%, Cooking 44.7%.
9. **Qwen3-VL always uses 0-1000 space**: bbox coordinates MUST be divided by 1000 and scaled by img dimension. Critical bug fixed in commit ece420a.

---

## Critical Bugs Found and Fixed

### 1. Resume dtype contamination (job 463119 crash)
**Error**: `TypeError: Could not convert string '0.009...' to numeric` at `.mean()`
**Cause**: Old CSV loaded as `dtype=str`, new rows as float. Pandas can't `.mean()` mixed string/float.
**Fix**: Add before summary:
```python
for _c in ["iou", "ba", "ca", "le"]:
    if _c in df.columns:
        df[_c] = pd.to_numeric(df[_c], errors="coerce")
```

### 2. SAM3 token overflow (sequence length error)
**Error**: `Sequence length N exceeds model maximum (32)`
**Cause**: CoT outputs 100-200 tokens fed directly to SAM3 text encoder
**Fix**: `_prepare_sam3_text()` — extract final answer from CoT ("final answer:" marker), truncate to 28 tokens

### 3. sample_source_frames temporal bug
**Error**: linspace selected random frames across full video
**Cause**: `np.linspace(0, len(all_frame_ids)-1, n_frames, dtype=int)` ignores current_frame position
**Fix**: `current_frame` parameter — select N most recent annotated frames before current_frame

### 4. DOS line endings in sbatch
**Error**: `sbatch: error: Batch script contains DOS line breaks (\r\n)`
**Fix**: `sed -i 's/\r//' smoke_box.sbatch`

### 5. num_predict=32 missing outputs
**Error**: 20-25% of VLM outputs are empty/truncated (cloud model)
**Cause**: num_predict=32 too short for "color object" answers
**Fix**: num_predict=64 for cloud, 512 for instruct. But 64 still gives 25.5% missing! Use 128-256 for cloud.

---

## Key Architecture Decisions

### VLM as Localizer (EXP-C-box) — NEW
Instead of VLM → text description → SAM3 text prompt, use:
**VLM → bbox coordinates (x1,y1,x2,y2) → SAM3 box prompt**

This bypasses SAM3's 28-token limit entirely.

```python
# SAM3 with box prompt (no text encoder)
inputs = proc(
    images=img,
    input_boxes=[[[x1, y1, x2, y2]]],  # batch × n_boxes × 4
    return_tensors="pt",
)
```

Key functions: `parse_vlm_bbox()`, `sam3_segment_box()`
Routing: `exp_cfg.get("vlm_output_type", "text")` — set `"bbox"` for box experiments.

### Config-driven Experiments
All experiments defined in `config.json`. Add new experiment by adding entry to `"experiments": [...]`.
Key fields: `id`, `images`, `prompt`, `vlm_model`, `num_predict`, `num_frames` (optional), `vlm_output_type` (optional, "text"|"bbox")

### Resume Logic
Script saves to `results_{JOB_ID}.csv` every 50 pairs. On restart, loads previous CSV and skips already-processed (pair_id, experiment) tuples. **Warning**: old CSV may have different dtype — always apply `pd.to_numeric()` after loading.

---

## API Keys

API keys live **only** in `src/scripts/experiments/vlm-sam3/config.json` (skip-worktree, never committed).
- Ollama keys: 3 keys in `vlm-api-keys` array (hyphens, not underscores)
- HuggingFace key: in `huggingface-api-keys` array
- Ollama base URL: `https://ollama.com` — hardcoded in experiment.py, **not** in config
- Key names use hyphens throughout: `vlm-api-keys`, `vlm-model`, `huggingface-api-keys`, `huggingface-model`

---

## Git Setup

- Branch: `gio/config-driven-experiments`
- Remote: `https://github.com/marcolomele/reluminati-research.git`
- Push requires PAT: `git push https://GioviManto:<PAT>@github.com/marcolomele/reluminati-research.git gio/config-driven-experiments`
- Config/sbatch files: `git update-index --skip-worktree <file>` (local-only, not pushed)

---

## TODO / Next Steps

- [ ] Run EXP-C-box / EXP-C-cot on full 1910 pairs (current runs cover ~1433 = 75%)
- [ ] Fix EXP-C num_predict: 64→128 to reduce 25.5% missing VLM outputs for cloud model
- [ ] Investigate WHY VLM fails 50-68% of pairs (qualitative error analysis)
- [ ] Try hybrid: EXP-C-box bbox → refine with SAM3 text prompt around bbox region
- [ ] Download high-res EgoExo4D images for local model testing
- [ ] Add results to Google Sheet (export_to_gsheets.py)
- [ ] Run `notebooks/results_analysis_465388_465389.ipynb` for team visualization
