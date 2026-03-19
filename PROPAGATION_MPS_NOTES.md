# SAM3 on Apple Silicon (MPS) — Notes

## 1. Changes Made for MPS Compatibility

All changes replace hardcoded CUDA assumptions with device-aware logic. No algorithmic changes.

### `sam3/model_builder.py`
- `build_sam3_video_model`: default `device` parameter changed from `"cuda" if torch.cuda.is_available() else "cpu"` → `None`, resolved at runtime via `_get_default_device()` (MPS → CUDA → CPU priority).

### `sam3/model/sam3_video_predictor.py`
- `Sam3VideoPredictor.__init__`: removed `.cuda()` after model build; now uses `_get_default_device()`.
- `_get_torch_and_gpu_properties`: added MPS/CPU branches to avoid unconditional `torch.cuda.current_device()` call.

### `sam3/model/io_utils.py`
- Added `_get_compute_device()` helper (MPS → CUDA → CPU).
- Replaced all 6 hardcoded `.cuda()` calls on frame/mean/std tensors with `.to(_get_compute_device())`.

### `sam3/model/sam3_tracking_predictor.py`
- `init_state`: `inference_state["storage_device"]` changed from `torch.device("cuda")` → `self.device`.
- Frame fetch in `_run_single_frame_inference`: `.cuda()` → `.to(inference_state["device"])`.

### `sam3/model/sam3_tracker_base.py`
- `maskmem_features` and `maskmem_pos_enc` loads: `.cuda()` → `.to(device)` (device already derived from backbone tensor).
- Two `pin_memory()` calls guarded with `if device.type == "cuda"` — pinned memory is a CUDA-only optimization.

### `sam3/model/sam3_video_inference.py`
- `_postprocess_output`: `keep_idx.pin_memory()` guarded by `device.type == "cuda"`.

### `sam3/perflib/connected_components.py`
- `connected_components_cpu`: added empty-batch guard (`batch_size == 0` returns zero tensors immediately instead of crashing on `torch.stack([])`).
- Input moved to CPU before `skimage` processing; result moved back to original device (handles MPS tensors transparently).

### `sam3/sam/transformer.py` — `RoPEAttention` (critical fix)
- `__init__`: `freqs_cis` always computed on CPU (avoids `torch.polar` on MPS); `.real` and `.imag` always extracted.
- `forward`: recomputes freq buffers when shape **or device** mismatches. Moves `freqs_cis_real`/`freqs_cis_imag` to `q.device` separately (real tensors, fully supported on MPS). Forces `apply_rotary_enc_real` path when `q.device.type == "mps"`.

### `sam3/sam/rope.py`
- `apply_rotary_enc`: `freqs_cis.repeat()` on complex tensors not supported on MPS. Replaced with `view_as_real` → `repeat` → `view_as_complex` (`.contiguous()` required).

### `examples/sam3_video_predictor_example.ipynb`
- Cell 6 (`!nvidia-smi`): commented out.
- Cell 8 (setup): replaced `gpus_to_use = range(torch.cuda.device_count())` with MPS/CUDA/CPU detection, `torch.autocast`, and `torch.inference_mode` enter.
- Cell 9 (build predictor): replaced `build_sam3_video_predictor(gpus_to_use=...)` (CUDA multi-GPU class) with `Sam3VideoPredictor()` (single-device, now MPS-aware).

---

## 2. How SAM3 Propagates Masks Through a Video

### Overview

Propagation is an **autoregressive, frame-by-frame loop**. Given a prompted frame (text, point, or box), it produces a mask on that frame and then iterates through the remaining frames one at a time, using a **memory bank** to carry information across time.

### Per-frame computation (`_run_single_frame_inference`)

```
Current frame pixels
        │
        ▼
  ViT-L backbone          ← 32-layer transformer, ~80% of per-frame cost
  (visual encoding)
        │
        ▼
  Memory fusion            ← cross-attention over memory bank (4 transformer layers)
  (tracker)                   memory bank: up to 7 past frames + all prompted frames
        │
        ▼
  Mask decoder             ← produces mask logits at low resolution
        │
        ▼
  Memory encoder           ← compresses mask + features → stored for future frames
        │
        ▼
  Output mask (upsampled to video resolution)
```

### Memory bank structure

- `num_maskmem = 7` slots total.
- **6 most recent non-conditioning frames** (temporally strided if video is long).
- **All conditioning frames** (frames where user gave a prompt) — always kept, regardless of distance.
- Each slot stores compressed mask embeddings (`maskmem_features`, shape `[B, 64, H/16, W/16]`) plus positional encodings.
- Temporal position embeddings encode the distance from current frame to each memory frame.

### Propagation directions

All three modes are supported (`propagation_direction` parameter):

| Mode | Behaviour |
|------|-----------|
| `"forward"` | Start frame → end of video |
| `"backward"` | Start frame → beginning of video |
| `"both"` (default in notebook) | Forward pass first, then backward pass |

Backward propagation flips the temporal position sign (`tpos_sign_mul = -1`) and reads the memory bank from future frames rather than past frames. The memory cross-attention logic is otherwise identical.

---

## 3. Per-Frame Efficiency Bottlenecks

### On MPS (measured: ~4.65 s/frame)

| Step | Cost | Notes |
|------|------|-------|
| ViT-L backbone | ~3.5–4 s | 32 transformer layers; no Flash Attention on MPS |
| RoPE encoding | Previously slow (now fixed) | `view_as_complex` not implemented on MPS → was silently falling back to CPU on every attention layer. Fixed by forcing `apply_rotary_enc_real` (real arithmetic only) |
| Memory cross-attention | ~0.5 s | 4-layer transformer over 7 frames |
| Connected components | ~0.05 s | `skimage` on CPU; blocking MPS↔CPU transfer |
| `pin_memory()` | eliminated | was crashing; now skipped on MPS |

**Root cause of original slowness**: `aten::view_as_complex` is not implemented on MPS. Without `PYTORCH_ENABLE_MPS_FALLBACK=1`, it raised an error. With fallback, it silently moved tensors to CPU for every RoPE call in every attention layer (~36 layers × 2 RoPE calls = ~72 CPU round trips per frame).

### On CUDA (estimated)

With Flash Attention (FA2 via `scaled_dot_product_attention`) and `torch.bfloat16` autocast:

| Bottleneck | CUDA mitigation |
|------------|-----------------|
| ViT-L attention | Flash Attention: 2–4× faster than standard attention |
| Full model | `torch.compile` (set `compile=True` in `Sam3VideoPredictor`): ~1.5–2× |
| bfloat16 throughput | Ampere+ has native bf16: ~312 TFLOPS on A100 vs ~20 TFLOPS on M-series |

---

## 4. CUDA Acceleration — Practical Notes

### Estimated speedup vs MPS (4.65 s/frame baseline)

| GPU | Est. time/frame | Speedup |
|-----|-----------------|---------|
| A100 80GB | ~35–60 ms | ~80–130× |
| V100 32GB | ~80–150 ms | ~30–60× |
| RTX 4090 | ~40–70 ms | ~65–115× |
| RTX 3090 | ~70–120 ms | ~40–65× |

For the 270-frame example video:
- MPS: ~21 minutes
- A100: ~14 seconds

### How to enable acceleration

```python
# In Sam3VideoPredictor / build_sam3_video_model:
predictor = Sam3VideoPredictor(compile=True)
# or
model = build_sam3_video_model(device="cuda", compile=True)
```

`compile=True` triggers `torch.compile` on the ViT backbone and pixel decoder (set via `compile_mode="default"` in `model_builder.py`). Flash Attention is used automatically on CUDA when `torch.nn.functional.scaled_dot_product_attention` dispatches to the Flash kernel (requires PyTorch ≥ 2.0 and GPU with sm_80+).

### SLURM (Bocconi HPC)

```bash
#SBATCH --partition=long_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
```

The multi-GPU path (`Sam3VideoPredictorMultiGPU`) requires NCCL and distributes the ViT backbone across GPUs. Not needed for single-video inference; single GPU is sufficient.

---

## 5. Sessions — What They Are, How to Clean Up

### Background: GPU memory is not like RAM

On CPU, Python's garbage collector reclaims memory automatically when an object has no more references. GPU memory (VRAM on CUDA, unified memory on MPS) follows the same Python reference-counting rule, but with two important differences:

1. **The allocator keeps a cache.** PyTorch does not immediately return freed GPU memory to the OS; it holds it in an internal pool so that future allocations are fast. This means `del tensor` may not visibly reduce the reported GPU memory usage — PyTorch still owns the block. Only `torch.cuda.empty_cache()` (CUDA) or explicitly ending the process releases the pool.
2. **OOM errors are silent and abrupt.** If you allocate too much, the kernel raises `RuntimeError: out of memory` mid-computation with no warning. Unlike RAM, there is no swap.

Consequence: if you open many sessions in a loop without closing them, memory grows monotonically until the process crashes.

### What a SAM3 session is

A *session* in SAM3 is a Python `dict` (`inference_state`) that holds all the data needed to track objects through a specific video. It is created by `init_state` (inside `Sam3VideoPredictor.start_session`) and stored in the class-level dict `_ALL_INFERENCE_STATES`, keyed by a UUID string.

**What lives in GPU memory for a 270-frame, 1024×1024 video (float16):**

| Field | What it holds | Approx. size |
|-------|--------------|-------------|
| `images` | All video frames, normalised to `[B, 3, 1024, 1024]` float16 | ~1.6 GB |
| `cached_features` | ViT backbone features for recently visited frames (up to a few frames cached) | ~100–400 MB |
| `output_dict["cond_frame_outputs"]` | Mask embeddings + low-res logits for each prompted frame | ~tens of MB |
| `output_dict["non_cond_frame_outputs"]` | Same, for all non-prompted frames that have been propagated through | grows with propagation |
| `output_dict_per_obj` | Per-object slices of the above (shared memory, not a copy) | — |
| `temp_output_dict_per_obj` | Temporary scratch space during interactive prompting | small |

The `images` tensor is the dominant cost. `offload_video_to_cpu=True` keeps frames in CPU RAM instead (lower VRAM, slightly slower — documented 3 fps drop from 27 → 24 fps in the original code).

### Lifecycle

```
predictor.start_session(video_path)   →  allocates images tensor + state dict in GPU memory
                                          returns session_id (UUID string)

predictor.add_prompt(session_id, ...)  →  adds prompts to state (small)
predictor.propagate_in_video(...)      →  fills output_dict as frames are processed

predictor.reset_session(session_id)   →  clears all prompts and propagation outputs
                                          keeps the images tensor loaded
                                          use when re-running with different prompts on same video

predictor.close_session(session_id)   →  removes state from _ALL_INFERENCE_STATES
                                          calls del session + gc.collect()
                                          releases Python references; allocator pool may still hold blocks

predictor.shutdown()                  →  clears ALL sessions at once (no gc.collect per session)
```

### How to close correctly

```python
# After you are done with a video:
predictor.close_session(session_id)

# On CUDA, optionally release the allocator pool back to the OS:
torch.cuda.empty_cache()
```

`close_session` is **idempotent** — calling it twice on the same `session_id` is safe (logs a warning, does not raise).

On MPS there is no equivalent of `torch.cuda.empty_cache()`; the Metal runtime manages the pool internally and reclaims memory when Python's GC runs or when the process ends.

### Why `del` alone is not enough (on CUDA)

```python
state = predictor._ALL_INFERENCE_STATES[session_id]
del state          # only removes YOUR local reference
                   # _ALL_INFERENCE_STATES still holds a reference → no memory freed
```

`close_session` does the right thing: it calls `_ALL_INFERENCE_STATES.pop(session_id)`, which removes the last strong reference, then `gc.collect()` to ensure Python's cycle collector runs immediately. The VRAM blocks are returned to PyTorch's pool (not the OS) until `empty_cache()` is called.

### Common mistake: session leak in a loop

```python
# BAD — opens N sessions, each holding a full video in GPU memory
for video in videos:
    result = predictor.start_session(video)
    run_propagation(result["session_id"])
    # forgot to close!

# GOOD
for video in videos:
    result = predictor.start_session(video)
    try:
        run_propagation(result["session_id"])
    finally:
        predictor.close_session(result["session_id"])
```

The `finally` block ensures cleanup even if propagation raises an exception mid-way.
