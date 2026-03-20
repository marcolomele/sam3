#!/usr/bin/env python3
"""
SAM 3 Video Propagation vs. Naive Heuristic — EgoExo4D Batch Runner
====================================================================
Runs the full SAM 3 bi-directional propagation + one-shot centroid pipeline
for a user-specified list of (take, object, text prompt) triples.

Per-take behaviour:
  - Camera      : first camera containing "aria" in its name (ego view)
  - Reference   : frame with the LARGEST GT mask for the given object

Outputs per take (saved to {out_root}/{uid[:5]}_results/):
  0_salient_frame.jpeg            — reference frame + GT mask
  0_iou_across_frames.jpeg        — per-frame IoU curve for both methods
  side_by_side_frame_{n}.jpeg     — overlay panels for every VIS_STRIDE-th frame
  gt_masks.npz                    — ground-truth boolean masks  {seq_idx: mask}
  sam3_masks.npz                  — SAM 3 propagated masks      {seq_idx: mask}
  naive_masks.npz                 — heuristic propagated masks  {seq_idx: mask}

Input CSV format (--input_csv):
  take_uid,object_name,text_prompt
  8acb2112-...,black saucepan _0,kitchen utensil
  3f7d9a01-...,basketball,ball

Usage (SLURM-friendly):
  python run_egoexo_propagation.py \\
      --data_dir /path/to/data \\
      --input_csv takes.csv \\
      [--vis_stride 50] \\
      [--confidence_threshold 0.3] \\
      [--out_root /path/to/output_root]
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for SLURM / headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch SAM 3 propagation on EgoExo4D takes."
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Root directory containing one sub-folder per take UID.",
    )
    p.add_argument(
        "--input_csv", required=True,
        help=(
            "CSV file with columns: take_uid,object_name,text_prompt. "
            "One row per take to process."
        ),
    )
    p.add_argument(
        "--vis_stride", type=int, default=50,
        help="Save a side-by-side overlay every N frames (default 50).",
    )
    p.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Confidence threshold for SAM 3 image processor (default 0.3).",
    )
    p.add_argument(
        "--out_root", default=None,
        help=(
            "Root directory for output folders. "
            "Defaults to DATA_DIR if not set."
        ),
    )
    p.add_argument(
        "--bpe_path", default=None,
        help=(
            "Explicit path to bpe_simple_vocab_16e6.txt.gz. "
            "Use this when pkg_resources resolves to the wrong location "
            "(e.g. pip install missing assets/). "
            "Example: /home/user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        ) > 127
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def decode_mask(rle, H: int, W: int) -> np.ndarray:
    m = mask_util.decode(rle).astype(bool)
    if m.shape != (H, W):
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        ) > 127
    return m


def render_overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                   alpha: float = 0.45) -> np.ndarray:
    img = rgb.copy().astype(np.float32)
    if pred.shape != (rgb.shape[0], rgb.shape[1]):
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST
            )
        ) > 127
    overlap   = pred & gt
    pred_only = pred & ~overlap
    gt_only   = gt  & ~overlap
    for mask, colour in [
        (gt_only,   np.array([0,   200,   0], dtype=np.float32)),
        (pred_only, np.array([220,   0,   0], dtype=np.float32)),
        (overlap,   np.array([255, 220,   0], dtype=np.float32)),
    ]:
        img[mask] = img[mask] * (1 - alpha) + colour * alpha
    return img.clip(0, 255).astype(np.uint8)


def collect_propagation_masks(tracker, inf_state, start_frame_idx, reverse, n_frames):
    masks = {}
    for frame_idx, _, _, vrm, _ in tracker.propagate_in_video(
        inf_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=n_frames,
        reverse=reverse,
        propagate_preflight=True,
    ):
        masks[frame_idx] = (vrm[0] > 0.0).cpu().numpy().squeeze()
    return masks


# ---------------------------------------------------------------------------
# Per-take processing
# ---------------------------------------------------------------------------

def process_take(take_uid, object_name, text_prompt, data_dir, out_root,
                 vis_stride, confidence_threshold, tracker, image_processor,
                 device):
    print(f"\n{'='*70}")
    print(f"  Take  : {take_uid}")
    print(f"  Object: {object_name}")
    print(f"  Prompt: {text_prompt}")
    print(f"{'='*70}")

    # ── Load annotation ────────────────────────────────────────────────────
    ann_path = os.path.join(data_dir, take_uid, "annotation.json")
    if not os.path.exists(ann_path):
        print(f"  [SKIP] annotation.json not found at {ann_path}")
        return

    with open(ann_path) as f:
        ann = json.load(f)
    masks_by_obj = ann["masks"]  # {object: {camera: {frame_str: rle}}}

    # ── Validate object ────────────────────────────────────────────────────
    if object_name not in masks_by_obj:
        print(
            f"  [SKIP] Object \"{object_name}\" not found in annotation. "
            f"Available: {list(masks_by_obj.keys())}"
        )
        return

    # ── Select aria camera ─────────────────────────────────────────────────
    selected_cam = None
    selected_frame_strs = None
    for cam, frame_dict in masks_by_obj[object_name].items():
        if "aria" in cam.lower():
            selected_cam = cam
            selected_frame_strs = sorted(frame_dict.keys(), key=int)
            break

    if selected_cam is None:
        print(
            f"  [SKIP] No aria camera found for object \"{object_name}\" "
            f"in take {take_uid}. Available cameras: "
            f"{list(masks_by_obj[object_name].keys())}"
        )
        return

    selected_obj = object_name
    frame_nums = [int(f) for f in selected_frame_strs]
    print(f"  Camera: {selected_cam}  |  {len(frame_nums)} annotated frames")

    # ── Determine image size ───────────────────────────────────────────────
    img0_path = os.path.join(data_dir, take_uid, selected_cam,
                             f"{selected_frame_strs[0]}.jpg")
    img0 = Image.open(img0_path)
    W, H = img0.width, img0.height
    empty_mask = np.zeros((H, W), dtype=bool)

    # ── Load all frames + GT masks ─────────────────────────────────────────
    print(f"  Loading {len(frame_nums)} frames and GT masks...")
    frames_rgb = {}
    gt_masks   = {}
    for i, (fstr, fn) in enumerate(zip(selected_frame_strs, frame_nums)):
        img_path = os.path.join(data_dir, take_uid, selected_cam, f"{fstr}.jpg")
        frames_rgb[fn] = np.array(Image.open(img_path).convert("RGB"))
        gt_masks[fn] = decode_mask(
            masks_by_obj[selected_obj][selected_cam][fstr], H, W
        )
        if (i + 1) % 100 == 0 or (i + 1) == len(frame_nums):
            print(f"    loaded {i+1}/{len(frame_nums)} frames")

    # ── Select reference frame: largest GT mask ────────────────────────────
    ref_fn = max(frame_nums, key=lambda fn: gt_masks[fn].sum())
    ref_seq_idx = frame_nums.index(ref_fn)
    ref_gt_mask = gt_masks[ref_fn]
    print(f"  Reference frame: {ref_fn}  (seq {ref_seq_idx})  "
          f"mask={ref_gt_mask.sum():,} px")

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = os.path.join(out_root, f"{take_uid[:5]}_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    # ── Plot 0: salient reference frame ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(frames_rgb[ref_fn])
    axes[0].set_title(f"Reference Frame: {ref_fn}  (seq {ref_seq_idx})")
    axes[0].axis("off")
    axes[1].imshow(frames_rgb[ref_fn])
    axes[1].imshow(ref_gt_mask, alpha=0.5, cmap="jet")
    axes[1].set_title(f"GT Mask — {ref_gt_mask.sum():,} px  (largest)")
    axes[1].axis("off")
    fig.suptitle(f"{take_uid}  |  {selected_obj}  |  {selected_cam}", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "0_salient_frame.jpeg"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Copy frames to temp video dir for SAM 3 tracker ───────────────────
    n_frames = len(frame_nums)
    print(f"  Copying {n_frames} frames to temp dir...")
    tmp_video_dir = tempfile.mkdtemp(prefix="sam3_egoexo_")
    try:
        for seq_idx, fstr in enumerate(selected_frame_strs):
            src = os.path.join(data_dir, take_uid, selected_cam, f"{fstr}.jpg")
            dst = os.path.join(tmp_video_dir, f"{seq_idx:05d}.jpg")
            shutil.copy(src, dst)
        print(f"  Temp dir ready: {tmp_video_dir}")

        mask_tensor = torch.tensor(ref_gt_mask.astype(np.float32))

        # ── Method 1: SAM 3 bi-directional propagation ────────────────────
        print(f"  [1/2] SAM 3 propagation — forward ({n_frames - ref_seq_idx} frames)...")
        inf_state_fwd = tracker.init_state(video_path=tmp_video_dir)
        tracker.add_new_mask(
            inference_state=inf_state_fwd,
            frame_idx=ref_seq_idx, obj_id=1, mask=mask_tensor,
        )
        sam3_masks_fwd = collect_propagation_masks(
            tracker, inf_state_fwd, ref_seq_idx, reverse=False,
            n_frames=n_frames,
        )
        print(f"    forward done — {len(sam3_masks_fwd)} frames propagated")

        print(f"  [1/2] SAM 3 propagation — backward ({ref_seq_idx} frames)...")
        inf_state_bwd = tracker.init_state(video_path=tmp_video_dir)
        tracker.add_new_mask(
            inference_state=inf_state_bwd,
            frame_idx=ref_seq_idx, obj_id=1, mask=mask_tensor,
        )
        sam3_masks_bwd = collect_propagation_masks(
            tracker, inf_state_bwd, ref_seq_idx, reverse=True,
            n_frames=n_frames,
        )
        print(f"    backward done — {len(sam3_masks_bwd)} frames propagated")

        sam3_pred_masks = {
            i: (sam3_masks_bwd if i < ref_seq_idx else sam3_masks_fwd).get(
                i, empty_mask
            )
            for i in range(n_frames)
        }
        del inf_state_fwd, inf_state_bwd
        print(f"  SAM 3 propagation complete — {len(sam3_pred_masks)} total frames")

    finally:
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
        print(f"  Temp dir removed")

    # ── Method 2: One-shot SAM 3 + centroid heuristic ─────────────────────
    print(f"  [2/2] One-shot SAM 3 + centroid heuristic — {n_frames} frames...")
    ref_centroid = mask_centroid(ref_gt_mask)
    naive_pred_masks = {}
    naive_no_candidate = 0

    for seq_idx, (fstr, fn) in enumerate(zip(selected_frame_strs, frame_nums)):
        img_pil = Image.open(
            os.path.join(data_dir, take_uid, selected_cam, f"{fstr}.jpg")
        ).convert("RGB")

        state = image_processor.set_image(img_pil)
        state = image_processor.set_text_prompt(text_prompt, state)

        candidate_masks  = state.get("masks")
        candidate_scores = state.get("scores")

        if candidate_masks is None or candidate_masks.shape[0] == 0:
            naive_pred_masks[seq_idx] = empty_mask
            naive_no_candidate += 1
            continue

        best_mask = None
        best_dist = float("inf")
        for k in range(candidate_masks.shape[0]):
            m = candidate_masks[k, 0].cpu().numpy()
            c = mask_centroid(m)
            if c is None:
                continue
            dist = (
                (c[0] - ref_centroid[0]) ** 2 + (c[1] - ref_centroid[1]) ** 2
            ) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_mask = m

        naive_pred_masks[seq_idx] = best_mask if best_mask is not None else empty_mask

        if (seq_idx + 1) % 50 == 0 or (seq_idx + 1) == n_frames:
            print(f"    frame {seq_idx+1}/{n_frames}  |  no-candidate so far: {naive_no_candidate}")

    print(f"  One-shot heuristic complete — {naive_no_candidate}/{n_frames} frames had no candidates")

    # ── IoU computation ────────────────────────────────────────────────────
    print("  Computing IoU...")
    sam3_iou_per_frame  = {}
    naive_iou_per_frame = {}

    for seq_idx, fn in enumerate(frame_nums):
        gt = gt_masks[fn]
        sam3_iou_per_frame[fn]  = compute_iou(sam3_pred_masks[seq_idx],  gt)
        naive_iou_per_frame[fn] = compute_iou(naive_pred_masks[seq_idx], gt)

    sam3_mean_iou  = float(np.mean(list(sam3_iou_per_frame.values())))
    naive_mean_iou = float(np.mean(list(naive_iou_per_frame.values())))
    print(f"  SAM 3 mean IoU  : {sam3_mean_iou:.4f}")
    print(f"  Naive mean IoU  : {naive_mean_iou:.4f}")

    # ── Plot 1: IoU across frames ──────────────────────────────────────────
    print("  Saving IoU plot...")
    fig, ax = plt.subplots(figsize=(12, 3))
    xs = list(range(len(frame_nums)))
    ax.plot(xs, [sam3_iou_per_frame[fn] for fn in frame_nums],
            label=f"SAM 3 Propagation  (mean = {sam3_mean_iou:.3f})",
            color="steelblue", linewidth=1.5)
    ax.plot(xs, [naive_iou_per_frame[fn] for fn in frame_nums],
            label=f"One-shot + Centroids  (mean = {naive_mean_iou:.3f})",
            color="tomato", linestyle="--", linewidth=1.5)
    ax.axvline(ref_seq_idx, color="gray", linestyle=":", linewidth=1.2,
               label=f"Reference frame (seq {ref_seq_idx}, frame {ref_fn})")
    ax.set_xlabel("Frame sequence index")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f'Per-frame IoU — "{selected_obj}"  |  camera: {selected_cam}'
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "0_iou_across_frames.jpeg"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: side-by-side overlays ──────────────────────────────────────
    print(f"  Saving side-by-side overlays (stride={vis_stride})...")
    for seq_idx in range(0, len(frame_nums), vis_stride):
        fn = frame_nums[seq_idx]
        sam3_pred  = sam3_pred_masks[seq_idx]
        naive_pred = naive_pred_masks[seq_idx]
        gt         = gt_masks[fn]
        rgb        = frames_rgb[fn]
        sam3_iou   = sam3_iou_per_frame[fn]
        naive_iou  = naive_iou_per_frame[fn]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, pred, iou, title in [
            (axes[0], sam3_pred,  sam3_iou,  "SAM 3 Propagation"),
            (axes[1], naive_pred, naive_iou, "One-shot + Centroids"),
        ]:
            ax.imshow(render_overlay(rgb, pred, gt))
            ax.set_title(f"{title}  (IoU = {iou:.3f})")
            ax.axis("off")
        axes[1].legend(
            handles=[
                mpatches.Patch(color=(0, 200 / 255, 0),  label="GT only"),
                mpatches.Patch(color=(220 / 255, 0, 0),   label="Pred only"),
                mpatches.Patch(color=(1.0, 220 / 255, 0), label="Overlap"),
            ],
            loc="lower right", fontsize=8,
        )
        marker = "  ← reference" if seq_idx == ref_seq_idx else ""
        fig.suptitle(
            f"Frame {fn}  (seq {seq_idx}){marker}", y=1.01
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"side_by_side_frame_{fn}.jpeg"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    # ── Save masks as .npz ─────────────────────────────────────────────────
    print("  Saving masks (.npz)...  gt / sam3 / naive")
    np.savez_compressed(
        os.path.join(out_dir, "gt_masks.npz"),
        **{str(i): gt_masks[fn] for i, fn in enumerate(frame_nums)},
    )
    np.savez_compressed(
        os.path.join(out_dir, "sam3_masks.npz"),
        **{str(i): sam3_pred_masks[i] for i in range(len(frame_nums))},
    )
    np.savez_compressed(
        os.path.join(out_dir, "naive_masks.npz"),
        **{str(i): naive_pred_masks[i] for i in range(len(frame_nums))},
    )

    print(f"  Done — results written to {out_dir}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Load input CSV ─────────────────────────────────────────────────────
    # Expected columns: take_uid, object_name, text_prompt
    import csv
    takes = []
    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            takes.append((
                row["take_uid"].strip(),
                row["object_name"].strip(),
                row["text_prompt"].strip(),
            ))

    if not takes:
        print("No rows found in input CSV. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(takes)} take(s) from {args.input_csv}")
    out_root = args.out_root if args.out_root else args.data_dir

    # ── Device ────────────────────────────────────────────────────────────
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.autocast("mps", dtype=torch.bfloat16).__enter__()
        print("Using MPS (Metal Performance Shaders).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        print("Using CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    torch.inference_mode().__enter__()

    # ── Build models (once, shared across all takes) ───────────────────────
    print("Loading SAM 3 model...")
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = args.bpe_path
    if bpe_path is None:
        import pkg_resources
        bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    print(f"Using bpe_path: {bpe_path}")
    sam3_model = build_sam3_video_model(bpe_path=bpe_path)
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone
    image_processor = Sam3Processor(
        sam3_model.detector,
        confidence_threshold=args.confidence_threshold,
        device=device,
    )
    print("Model ready.\n")

    # ── Process each take ─────────────────────────────────────────────────
    errors = []
    for take_uid, object_name, text_prompt in takes:
        try:
            process_take(
                take_uid=take_uid,
                object_name=object_name,
                text_prompt=text_prompt,
                data_dir=args.data_dir,
                out_root=out_root,
                vis_stride=args.vis_stride,
                confidence_threshold=args.confidence_threshold,
                tracker=tracker,
                image_processor=image_processor,
                device=device,
            )
        except Exception as exc:
            print(f"\n  [ERROR] Take {take_uid} failed: {exc}")
            import traceback
            traceback.print_exc()
            errors.append((take_uid, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Processed {len(takes) - len(errors)}/{len(takes)} takes successfully.")
    if errors:
        print("  Failed takes:")
        for uid, msg in errors:
            print(f"    {uid}: {msg}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
