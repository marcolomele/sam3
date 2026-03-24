# SAM 3 Mask Propagation — Technical Deep Dive

> **Context**: This document explains the internal mechanics behind the `tracker.propagate_in_video()` call in `examples/run_egoexo_propagation.py`. It covers every layer of the stack, from the public API down to the transformer decoder and memory bank.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Key Files](#2-key-files)
3. [Step 0 — Model Construction](#3-step-0--model-construction)
4. [Step 1 — Inference State Initialisation](#4-step-1--inference-state-initialisation)
5. [Step 2 — Encoding the Reference Mask (`add_new_mask`)](#5-step-2--encoding-the-reference-mask-add_new_mask)
6. [Step 3 — Propagation Preflight](#6-step-3--propagation-preflight)
7. [Step 4 — Frame-by-Frame Propagation](#7-step-4--frame-by-frame-propagation)
8. [Step 5 — Core Per-Frame Inference (`_run_single_frame_inference`)](#8-step-5--core-per-frame-inference-_run_single_frame_inference)
9. [Step 6 — The Tracking Step (`track_step`)](#9-step-6--the-tracking-step-track_step)
10. [Step 7 — Memory Encoding](#10-step-7--memory-encoding)
11. [Step 8 — Multi-GPU Distribution (`_det_track_one_frame`)](#11-step-8--multi-gpu-distribution-_det_track_one_frame)
12. [Data Structures Reference](#12-data-structures-reference)
13. [Transformer Architecture](#13-transformer-architecture)
14. [Bi-directional Propagation in Practice](#14-bi-directional-propagation-in-practice)
15. [End-to-End Flow Diagram](#15-end-to-end-flow-diagram)

---

## 1. High-Level Overview

SAM 3 video propagation is a **memory-augmented, transformer-based, temporally-recursive** segmentation pipeline. Given a single binary mask on a reference frame, it propagates that mask across a video by:

1. **Encoding** the reference mask into a compact feature representation (the *memory*).
2. **Decoding** a new mask at each subsequent frame by cross-attending to the stored memory features.
3. **Updating** the memory bank with each newly decoded mask, so each frame benefits from the most recent segmentation history.
4. Repeating both **forward** (future frames) and **backward** (past frames) from the reference frame.

In `examples/run_egoexo_propagation.py` this appears as:

```python
# Forward pass
inf_state_fwd = tracker.init_state(video_path=tmp_video_dir)
tracker.add_new_mask(inference_state=inf_state_fwd, frame_idx=ref_seq_idx, obj_id=1, mask=mask_tensor)
sam3_masks_fwd = collect_propagation_masks(tracker, inf_state_fwd, ref_seq_idx, reverse=False, n_frames=n_frames)

# Backward pass
inf_state_bwd = tracker.init_state(video_path=tmp_video_dir)
tracker.add_new_mask(inference_state=inf_state_bwd, frame_idx=ref_seq_idx, obj_id=1, mask=mask_tensor)
sam3_masks_bwd = collect_propagation_masks(tracker, inf_state_bwd, ref_seq_idx, reverse=True, n_frames=n_frames)
```

where `collect_propagation_masks` calls `tracker.propagate_in_video(...)` and collects `(vrm[0] > 0.0)` as the binary mask per frame.

---

## 2. Key Files


| File                                    | Purpose                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `sam3/model_builder.py`                 | Factory: `build_sam3_video_model()`, `_create_tracker_transformer()`                                                     |
| `sam3/model/sam3_video_inference.py`    | Public API wrapper: `init_state()`, `propagate_in_video()`, `_run_single_frame_inference()`                              |
| `sam3/model/sam3_tracking_predictor.py` | SAM2-style tracker predictor: `init_state()`, `add_new_mask()`, `propagate_in_video()`, `propagate_in_video_preflight()` |
| `sam3/model/sam3_tracker_base.py`       | Core tracker: `track_step()`, memory bank management                                                                     |
| `sam3/model/sam3_video_base.py`         | Multi-GPU orchestration: `_det_track_one_frame()`, `run_tracker_propagation()`                                           |
| `sam3/model/sam3_video_predictor.py`    | Session management wrapper                                                                                               |
| `sam3/model/memory.py`                  | Memory modules: `SimpleMaskEncoder`, `SimpleMaskDownSampler`                                                             |


---

## 3. Step 0 — Model Construction

**File**: `sam3/model_builder.py` · `build_sam3_video_model()` (lines 675–816)

The video model is a composite of two sub-systems:

```
Sam3VideoModel
├── detector          ← image-level SAM3 (backbone + mask decoder)
│   └── backbone     ← shared ViT encoder
└── tracker           ← Sam3TrackerBase
    ├── transformer   ← 4-layer RoPE transformer decoder
    ├── memory_encoder← SimpleMaskEncoder
    └── memory_attention ← cross-attention to memory bank
```

The tracker is wired to share the **same backbone** as the detector:

```python
# run_egoexo_propagation.py line 512
tracker.backbone = sam3_model.detector.backbone
```

This is intentional: backbone features are computed once per frame and reused by both the detector (for detection) and the tracker (for propagation).

The tracker transformer is built at `_create_tracker_transformer()` (lines 369–431) with:

- `d_model = 256`
- `num_heads = 1`
- `num_layers = 4`
- `RoPEAttention` (Rotary Position Embeddings) for spatial awareness
- `kv_in_dim = 64` — cross-attention to compact 64-dim memory features

---

## 4. Step 1 — Inference State Initialisation

**File**: `sam3/model/sam3_video_inference.py` · `init_state()` (lines 55–89)

```python
inf_state = tracker.init_state(video_path=tmp_video_dir)
```

`init_state` loads the video from disk (or an in-memory source), runs a lightweight scan to determine frame count, and returns an **inference state dictionary**:

```python
inference_state = {
    "image_size": 1008,                   # model's normalised input size
    "num_frames": N,                       # total frames in the clip
    "orig_height": H,                      # original frame height
    "orig_width":  W,                      # original frame width
    "input_batch": BatchedDatapoint(...),  # batched raw frames
    "tracker_inference_states": [...],     # one entry per GPU shard
    "tracker_metadata": {...},             # obj IDs, scores, GPU routing
    "feature_cache": {},                   # backbone features cached per frame
    "cached_frame_outputs": {},            # per-frame masks (for display/resume)
    # per-object temporary outputs (populated by add_new_mask)
    "temp_output_dict_per_obj": {},
    "output_dict": {
        "cond_frame_outputs":     {},      # frames with user prompts
        "non_cond_frame_outputs": {},      # propagated frames
    },
    "output_dict_per_obj": {},
    "consolidated_frame_inds": {
        "cond_frame_outputs":     set(),
        "non_cond_frame_outputs": set(),
    },
    "obj_ids":       [],
    "obj_id_to_idx": {},
}
```

Two separate `inference_state` objects are created in `run_egoexo_propagation.py` — one for the forward pass, one for the backward pass — so they do not share memory state.

---

## 5. Step 2 — Encoding the Reference Mask (`add_new_mask`)

**File**: `sam3/model/sam3_tracking_predictor.py` · `add_new_mask()` (lines 343–459)

```python
tracker.add_new_mask(
    inference_state=inf_state_fwd,
    frame_idx=ref_seq_idx,   # e.g. frame 47 of 200
    obj_id=1,
    mask=mask_tensor,        # bool tensor (H, W)
)
```

### 5.1 Object Registration

The `obj_id` is mapped to an internal zero-based `obj_idx`. A new slot is allocated in `output_dict_per_obj` and `temp_output_dict_per_obj`.

### 5.2 Mask Resizing

The user-supplied mask is resized to the model's `input_mask_size` (typically 256×256 for the tracker):

```python
mask_resized = F.interpolate(mask[None, None].float(), size=input_mask_size, mode="nearest")
```

### 5.3 Single-Frame Inference on the Reference Frame

`_run_single_frame_inference()` is called with `mask_inputs=mask_resized` and `run_mem_encoder=False`. At this point the model:

- Extracts backbone features for the reference frame (or retrieves from `feature_cache`).
- Feeds the **mask directly** (not a point or box prompt) into the decoder as a conditioning signal — this is SAM 3's "mask prompt" path.
- Returns `current_out` containing:
  - `pred_masks` — predicted logits at low resolution (72×72)
  - `obj_ptr` — a 256-dim object pointer embedding summarising the object's appearance
  - `object_score_logits` — objectness confidence

### 5.4 Output Storage

The output is stored under `temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"][frame_idx]`. The `NO_OBJ_SCORE` sentinel values ensure non-overlapping constraints between objects:

- **Background** (where mask = 0): logit = `NO_OBJ_SCORE` (large negative)
- **Foreground** (where mask = 1): logit = `-NO_OBJ_SCORE` (large positive)

### 5.5 Cross-Object Consolidation

`_consolidate_temp_output_across_obj()` merges all per-object outputs into a single unified frame output. This enforces the hard constraint that no two objects can occupy the same pixel.

Memory encoding is **deferred** at this stage (`run_mem_encoder=False`). It runs during the preflight step.

---

## 6. Step 3 — Propagation Preflight

**File**: `sam3/model/sam3_tracking_predictor.py` · `propagate_in_video_preflight()` (called internally by `propagate_in_video`)

Before the main propagation loop, the preflight step:

1. **Runs the memory encoder** on all conditioned (prompt) frames. This converts the low-res mask logits into compact `maskmem_features` (B, 64, 72, 72) and positional encodings `maskmem_pos_enc`.
2. Moves outputs from `temp_output_dict_per_obj` into the stable `output_dict["cond_frame_outputs"]`.
3. Marks the reference frame in `consolidated_frame_inds["cond_frame_outputs"]`.

After this step, the memory bank contains one entry: the encoded reference frame mask. The propagation loop can begin.

---

## 7. Step 4 — Frame-by-Frame Propagation

**File**: `sam3/model/sam3_video_inference.py` · `propagate_in_video()` (lines 251–356)

```python
for frame_idx, _, _, vrm, _ in tracker.propagate_in_video(
    inf_state,
    start_frame_idx=start_frame_idx,
    max_frame_num_to_track=n_frames,
    reverse=reverse,
    propagate_preflight=True,
):
    masks[frame_idx] = (vrm[0] > 0.0).cpu().numpy().squeeze()
```

The outer loop iterates over frames in order (forward or backward). For each frame it:

1. Checks whether this frame already has a computed output (from `cached_frame_outputs` or `output_dict`). If so, yields it immediately — **no re-computation**.
2. Otherwise, calls `_run_single_frame_inference()` for the current frame — this is where the actual propagation happens.
3. Stores the result in `output_dict["non_cond_frame_outputs"][frame_idx]`.
4. Yields `(frame_idx, obj_ids, video_res_masks, low_res_masks, ious)`.

The caller (`collect_propagation_masks`) extracts `vrm[0] > 0.0` — the thresholded low-res mask (or video-resolution mask, depending on the return format).

**Processing order**:

- `reverse=False`: frames `[start_frame_idx, start_frame_idx+1, ..., N-1]`
- `reverse=True`: frames `[start_frame_idx, start_frame_idx-1, ..., 0]`

---

## 8. Step 5 — Core Per-Frame Inference (`_run_single_frame_inference`)

**File**: `sam3/model/sam3_video_inference.py` · `_run_single_frame_inference()` (lines ~175–250)

For a frame that has **no user prompt** (all frames except the reference), this function:

1. **Retrieves backbone features** for the frame. If already cached in `feature_cache`, reuse them. Otherwise, run the shared ViT backbone:
  ```
   frame_rgb (3, 1008, 1008) → backbone → features (256, 72, 72)
  ```
2. **Gathers memory features** from previous frames. The memory bank contains up to `num_maskmem=7` past frames' `maskmem_features`. For the first propagation step, this is just the reference frame.
3. **Calls the tracker's `track_step()`** (or equivalently, dispatches through `_det_track_one_frame()` in multi-GPU mode), passing:
  - Current frame backbone features
  - Concatenated memory features from the bank
  - Object pointers from the reference frame
4. **Stores the output** in `non_cond_frame_outputs[frame_idx]` and updates the memory bank.

---

## 9. Step 6 — The Tracking Step (`track_step`)

**File**: `sam3/model/sam3_tracker_base.py` · `track_step()` (line ~928+)

This is the heart of the propagation. It runs one step of the tracker decoder for a single frame:

### 9.1 Input Preparation

- `current_vision_feats`: backbone features for the current frame, shape `(HW, B, 256)` — flattened spatial positions as a sequence.
- `memory_bank`: a list of up to 7 `maskmem_features` tensors, each shape `(B, 64, 72, 72)`, flattened and concatenated into `(N_memory × HW, B, 64)`.
- `obj_ptr`: object pointer from the reference (or most recent) frame, shape `(B, 256)`.

### 9.2 Memory Cross-Attention

The transformer decoder's cross-attention layers attend the **object queries** to the **memory bank**:

```
Q: learned object queries + positional embeddings  (shape: n_queries × B × 256)
K, V: flattened memory features from past frames   (shape: N_mem_tokens × B × 64 → projected to 256)
```

The `kv_in_dim=64` memory features are projected to `d_model=256` before the attention operation. The result is a set of updated object embeddings informed by where the object was in the past.

### 9.3 Spatial Cross-Attention to Current Frame

After attending to memory, the queries cross-attend to the **current frame's backbone features**:

```
Q: memory-conditioned object queries  (n_queries × B × 256)
K, V: current frame features          (HW × B × 256)
```

This allows the decoder to localise the object in the current frame, guided by where it was in memory.

### 9.4 Mask Prediction

The updated object queries are passed through a lightweight mask prediction head (MLP + upsampling) to produce:

- `pred_masks`: logits at 72×72 resolution
- `iou_score`: predicted IoU of the mask
- `obj_ptr`: updated object pointer embedding (for future frames)

---

## 10. Step 7 — Memory Encoding

**File**: `sam3/model/memory.py` · `SimpleMaskEncoder`, `SimpleMaskDownSampler`

After each frame's mask is predicted, it is encoded back into the memory bank for use by subsequent frames.

### 10.1 SimpleMaskDownSampler

A convolutional stack that compresses the mask spatially:

```
Input:  (B, 1, 1152, 1152)   ← upsampled mask at high resolution
→ Conv 3×3, stride 2          (B, 4,   576,  576)
→ Conv 3×3, stride 2          (B, 8,   288,  288)
→ Conv 3×3, stride 2          (B, 16,  144,  144)
→ Conv 3×3, stride 2          (B, 32,   72,   72)
→ Conv 3×3, stride 2          (B, 64,   72,   72)  ← output
```

(Exact intermediate channels depend on `mask_dim` config; final output is always 64-dim at 72×72.)

### 10.2 SimpleMaskEncoder (Fuser)

After downsampling, a **CXBlock** (ConvNeXt-style block) fuses the mask features with the backbone features:

```python
fused = mask_features + vision_features   # residual addition
fused = CXBlock(fused)                    # depth-wise conv + LayerNorm + MLP
```

Output: `maskmem_features` of shape `(B, 64, 72, 72)`.

### 10.3 Memory Bank Update

`maskmem_features` and `maskmem_pos_enc` are stored in the frame's output dict. The memory bank slides: when more than `num_maskmem=7` frames are stored, the oldest non-conditioned frame is evicted. Conditioned frames (user prompts) are **never evicted**.

---

## 11. Step 8 — Multi-GPU Distribution (`_det_track_one_frame`)

**File**: `sam3/model/sam3_video_base.py` · `_det_track_one_frame()` (lines 152–292)

In multi-GPU mode (single-process multi-GPU / SPMD), this function orchestrates a **5-phase** pipeline per frame. In single-GPU mode (as in `run_egoexo_propagation.py`) all phases run on one device, but the same code path executes.


| Phase | Function                             | Description                                                                             |
| ----- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| 1     | `run_backbone_and_detection`         | ViT backbone + detector decoder → bounding boxes, masks, scores for *new* objects       |
| 2     | `run_tracker_propagation`            | Propagate existing tracked objects forward 1 frame via `track_step()`                   |
| 3     | `run_tracker_update_planning_phase`  | GPU 0: match detections vs. propagated masks (IoU matching), decide which to add/remove |
| 4     | `run_tracker_update_execution_phase` | Execute add/remove decisions; add new detections to tracker                             |
| 5     | `build_outputs`                      | Merge masks from propagation + new detections; upsample to video resolution             |


For video propagation of a **single pre-specified object** (as in the example), Phase 1 (detection of new objects) is largely irrelevant. The critical path is Phase 2 → Phase 5.

### Phase 2 Detail: `run_tracker_propagation`

```python
# Simplified from sam3_video_base.py lines 402–452
propagated_masks = []
for gpu_shard in tracker_shards:
    local_masks = _propogate_tracker_one_frame_local_gpu(gpu_shard, frame_features)
    propagated_masks.append(local_masks)
# All-gather across GPUs
global_masks = torch.cat(all_gather(propagated_masks), dim=0)
```

`_propogate_tracker_one_frame_local_gpu()` (lines 1098–1159) calls `track_step()` for each object in the local GPU's shard and returns its low-res mask output.

---

## 12. Data Structures Reference

### Inference State (top level)

```python
inference_state = {
    "image_size": int,                         # model input size (1008)
    "num_frames": int,                         # total frames
    "orig_height": int, "orig_width": int,     # original resolution
    "input_batch": BatchedDatapoint,           # raw frames
    "tracker_inference_states": List[Any],     # per-GPU tracker states
    "tracker_metadata": Dict,                  # obj ID routing
    "feature_cache": Dict[int, Tensor],        # frame_idx → backbone features
    "output_dict": {
        "cond_frame_outputs":     Dict[int, FrameOut],  # prompt frames
        "non_cond_frame_outputs": Dict[int, FrameOut],  # propagated frames
    },
    "output_dict_per_obj": Dict[int, Dict],   # obj_idx → same structure
    "temp_output_dict_per_obj": Dict,          # staging area before consolidation
    "consolidated_frame_inds": {
        "cond_frame_outputs":     Set[int],
        "non_cond_frame_outputs": Set[int],
    },
    "obj_ids":       List[int],                # user-facing IDs
    "obj_id_to_idx": Dict[int, int],           # user ID → internal idx
}
```

### Frame Output (`FrameOut`)

```python
current_out = {
    "pred_masks":            Tensor(B, 1, 72,   72),    # low-res logits
    "pred_masks_video_res":  Tensor(B, 1, H,    W),     # full-res binary
    "maskmem_features":      Tensor(B, 64, 72,  72),    # encoded memory
    "maskmem_pos_enc":       List[Tensor],               # position embeddings
    "obj_ptr":               Tensor(B, 256),             # object appearance embedding
    "object_score_logits":   Tensor(B, 1),               # objectness score
    "iou_score":             Tensor(B, 1),               # optional IoU estimate
}
```

---

## 13. Transformer Architecture

**File**: `sam3/model_builder.py` · `_create_tracker_transformer()` (lines 369–431)

```
Input: object queries (n_obj × B × 256) + frame features (HW × B × 256)

Layer 1–4 (repeated):
  ┌─────────────────────────────────────────────────────────────┐
  │ Self-Attention (RoPEAttention)                              │
  │   Q, K, V = object queries                                  │
  │   RoPE: encodes spatial position via rotary embeddings      │
  │   Objects attend to each other (for non-overlap awareness)  │
  ├─────────────────────────────────────────────────────────────┤
  │ Cross-Attention to Memory (RoPEAttention)                   │
  │   Q = object queries (256-dim)                              │
  │   K, V = maskmem_features from past frames (64→256-dim)     │
  │   Temporal: where was the object in the last N frames?      │
  ├─────────────────────────────────────────────────────────────┤
  │ Cross-Attention to Current Frame (RoPEAttention)            │
  │   Q = memory-conditioned queries (256-dim)                  │
  │   K, V = backbone features of current frame (256-dim)       │
  │   Spatial: where is the object now?                         │
  ├─────────────────────────────────────────────────────────────┤
  │ FFN (2-layer MLP, d_ff = 4 × 256 = 1024)                   │
  └─────────────────────────────────────────────────────────────┘

Output: updated queries → Mask MLP → pred_masks (72×72 logits)
                        → IOU MLP  → iou_score
                        → OBJ PTR  → obj_ptr (256-dim)
```

**RoPEAttention** uses rotary position embeddings instead of learned absolute positions. This allows the model to generalise to different spatial positions and frame orderings without retraining, which is critical for **reverse propagation** (the model was not necessarily trained on reversed sequences — RoPE handles this more gracefully than absolute positional embeddings).

---

## 14. Bi-directional Propagation in Practice

`run_egoexo_propagation.py` runs two separate inference states:

```
Reference frame (seq_idx = R)

Forward:   R → R+1 → R+2 → ... → N-1
Backward:  R → R-1 → R-2 → ... → 0

Final assignment:
  frame i < R  →  use backward mask
  frame i >= R →  use forward mask
```

```python
sam3_pred_masks = {
    i: (sam3_masks_bwd if i < ref_seq_idx else sam3_masks_fwd).get(i, empty_mask)
    for i in range(n_frames)
}
```

**Why two separate states?** The memory bank is causal — it propagates information in one direction. A single inference state cannot simultaneously propagate forward and backward. Two separate states each initialise from the reference mask and propagate in their respective direction, ensuring the memory bank always reflects the temporally "nearest" past.

**Memory bank at the reference frame**: Both states start with exactly one entry in their memory bank — the encoded reference mask. This is stored in `cond_frame_outputs` and is never evicted.

**Memory bank growth during propagation**: After each frame is processed, its `maskmem_features` are added to the bank. When the bank exceeds `num_maskmem=7` non-conditioned frames, the oldest one is dropped. The conditioning frame (reference) is always retained regardless.

---

## 15. End-to-End Flow Diagram

```
run_egoexo_propagation.py
│
├── tracker.init_state(video_path)
│   └── Loads frames, returns empty inference_state dict
│
├── tracker.add_new_mask(inf_state, frame_idx=R, mask=GT_mask)
│   ├── Resize mask → 256×256
│   ├── _run_single_frame_inference(frame_idx=R, mask_inputs=mask)
│   │   ├── backbone(frame_R) → features_R (256, 72, 72)         [cached]
│   │   └── decoder(queries, mask_input=mask) → pred_masks, obj_ptr
│   └── Store in temp_output_dict["cond_frame_outputs"][R]
│
├── propagate_in_video_preflight()
│   └── memory_encoder(pred_masks_R) → maskmem_features_R (64, 72, 72)
│       Store in output_dict["cond_frame_outputs"][R]
│
└── tracker.propagate_in_video(inf_state, start=R, reverse=False)
    │
    For frame t = R, R+1, R+2, ..., N-1:
    │
    ├── _run_single_frame_inference(frame_idx=t, no mask_inputs)
    │   │
    │   ├── backbone(frame_t) → features_t                        [cached]
    │   │
    │   ├── Gather memory bank:
    │   │   [maskmem_features_{t-7}, ..., maskmem_features_{t-1}]
    │   │   (plus cond_frame maskmem_features_R if within window)
    │   │
    │   └── track_step(features_t, memory_bank)
    │       ├── Self-attention (queries ↔ queries)
    │       ├── Cross-attention (queries ↔ memory_bank)   [WHERE was it?]
    │       ├── Cross-attention (queries ↔ features_t)    [WHERE is it now?]
    │       ├── FFN
    │       └── Mask MLP → pred_masks_t (72×72 logits)
    │           IOU  MLP → iou_score_t
    │           OBJ  PTR → obj_ptr_t
    │
    ├── memory_encoder(pred_masks_t) → maskmem_features_t
    │   ├── SimpleMaskDownSampler: 1152×1152 → 64×72×72
    │   └── CXBlock fuser: fuse mask features + vision features
    │
    ├── Store maskmem_features_t in memory bank
    │   (evict oldest non-cond frame if bank > 7 entries)
    │
    └── yield (t, obj_ids, video_res_masks, low_res_masks, ious)
        └── caller: masks[t] = (low_res_masks[0] > 0.0).numpy()
```

---

## Summary

SAM 3 mask propagation is, at its core, a **sliding-window temporal cross-attention** mechanism:

- The **reference mask** is compressed into a 64-dim spatial feature map via a convolutional encoder.
- Each subsequent frame's mask is predicted by a **4-layer transformer** that cross-attends to the last ≤7 frames' encoded masks and the current frame's backbone features.
- The predicted mask is immediately re-encoded and added to the rolling memory bank.
- **Backward propagation** is identical in structure — the frame processing order is simply reversed, and a fresh inference state ensures the memory bank grows backward in time.
- The **shared ViT backbone** computes frame features once and caches them, making repeated passes (forward + backward) computationally cheap.

