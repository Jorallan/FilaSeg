"""Generate all schematic figures + GIF for docs/pipeline_overview.md.

Outputs go into docs/figures/. Re-run anytime the example outputs change.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from skimage.morphology import skeletonize


# Per-frame display time in milliseconds. PIL/Pillow uses this directly in the
# GIF Graphic Control Extension; viewers honor it more consistently than the
# imageio `duration` argument.
FRAME_MS = 2500  # 2.5 seconds per frame


def write_gif(path: Path, frames: list[np.ndarray], frame_ms: int = FRAME_MS) -> None:
    """Write an animated GIF using Pillow with explicit per-frame duration.

    Pillow stores duration as the centisecond value `duration` in the GIF's
    Graphic Control Extension. Most browsers and image viewers honour it
    consistently (the imageio v2 `duration=` API does not always).
    """
    if not frames:
        return
    pil_frames = [Image.fromarray(f) for f in frames]
    # `disposal=2` clears each frame before showing the next so caption strips
    # don't ghost across frames.
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(frame_ms),
        loop=0,
        disposal=2,
        optimize=False,
    )

ROOT = Path(r"C:\Repos\filaments_quantification")
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Example run to source figures from (stringart_tiles.py, the pipeline's only
# stage-1 method).
EXAMPLE_RUN = ROOT / "output" / "full_pipeline" / "sem_full_00000_1p66_crop512_b"
INPUT_DIR = ROOT / "input" / "sem_full_00000_1p66_crop512"
MASK_PATH = INPUT_DIR / "mask.png"
SEM_PATH = INPUT_DIR / "sem.png"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def colorize_labels(lbl: np.ndarray) -> np.ndarray:
    """Distinct random color per non-zero label."""
    h, w = lbl.shape
    rng = np.random.default_rng(42)
    ids = np.unique(lbl)
    palette = {int(i): rng.integers(40, 255, size=3, dtype=np.uint8) for i in ids if i != 0}
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in palette.items():
        out[lbl == i] = color
    return out


def overlay_on_sem(sem_gray: np.ndarray, mask: np.ndarray, color, alpha=0.6) -> np.ndarray:
    rgb = cv2.cvtColor(sem_gray, cv2.COLOR_GRAY2RGB)
    if mask.dtype != bool:
        mask = mask > 0
    out = rgb.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * rgb[mask]).astype(np.uint8)
    return out


# ----------------------------------------------------------------------
# Figure 1: end-to-end block diagram
# ----------------------------------------------------------------------

def fig01_block_diagram():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stages = [
        ("Input", "#FFEFD5",
         "mask.png\nsem.png"),
        ("1. stringart", "#B0E0E6",
         "skeletonize\nsplit at junctions\nbin by tangent angle\nper-angle branches"),
        ("2. preprocess", "#DDA0DD",
         "oriented close\ndrop tiny CCs\ndominant path\nB-spline smooth"),
        ("3. reconnect", "#90EE90",
         "stage_clear\nstage_strict\nstage_relaxed\nsmooth passes"),
        ("4. postprocess", "#F0E68C",
         "skeletonize\nsmooth path\nsmart-width render\nabsorb / trim"),
        ("Final", "#FFB6C1",
         "labels.tif\ncolor preview\noverlay PNG\nDEM JSON"),
    ]

    box_w, box_h = 2.05, 3.0
    gap = 0.15
    y_center = 3.0
    x = 0.1
    centers = []
    for title, color, desc in stages:
        rect = FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.08",
                              linewidth=1.5, edgecolor="#333333",
                              facecolor=color)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_center + 1.05, title,
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + box_w / 2, y_center - 0.3, desc,
                ha="center", va="center", fontsize=8.5, linespacing=1.4)
        centers.append((x + box_w / 2, y_center))
        x += box_w + gap

    # arrows between boxes
    for i in range(len(centers) - 1):
        a = centers[i]
        b = centers[i + 1]
        ax.add_patch(FancyArrowPatch((a[0] + box_w / 2 - 0.02, a[1]),
                                     (b[0] - box_w / 2 + 0.02, b[1]),
                                     mutation_scale=18, lw=2, color="#222222"))

    ax.set_title("Filament-quantification pipeline (FilaSeg)", fontsize=14, pad=12)
    out = FIG_DIR / "01_block_diagram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 2: stage 1 — mask → branches reconstruction
# ----------------------------------------------------------------------

def fig02_stage1():
    mask = read_gray(MASK_PATH)
    recon = read_gray(EXAMPLE_RUN / "1.stringart" / "reconstructed.png")
    # Build a colored-by-bin composite from the per-branch PNGs
    br_dir = EXAMPLE_RUN / "1.stringart" / "branches"
    branch_files = sorted(br_dir.glob("mask_branch_*.png"),
                          key=lambda p: int(p.stem.split("_")[-1]))
    n_b = len(branch_files)
    cmap = plt.get_cmap("hsv")
    composite = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, bf in enumerate(branch_files):
        b = read_gray(bf) > 127
        color = (np.array(cmap(i / max(1, n_b - 1))[:3]) * 255).astype(np.uint8)
        composite[b] = color

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Input mask")
    axes[1].imshow(composite)
    axes[1].set_title(f"Stage 1: per-angle branches ({n_b} bins, color = bin index)")
    axes[2].imshow(recon, cmap="gray")
    axes[2].set_title("Stage 1: reconstructed (union of all branches)")
    for a in axes:
        a.axis("off")
    out = FIG_DIR / "02_stage1_branches.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 3: stage 2 — preprocess clean-up of a single branch
# ----------------------------------------------------------------------

def fig03_stage2():
    # Compare one raw branch to its cleaned version
    raw = read_gray(EXAMPLE_RUN / "1.stringart" / "branches" / "mask_branch_3.png")
    cleaned = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / "mask_branch_3.png")
    raw_merge = read_gray(EXAMPLE_RUN / "1.stringart" / "branches" / "mask_branches_merge.png")
    cleaned_merge = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / "mask_branches_merge.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(raw, cmap="gray")
    axes[0, 0].set_title("Stage 1 branch 3 (raw, before preprocess)")
    axes[0, 1].imshow(cleaned, cmap="gray")
    axes[0, 1].set_title("Stage 2 branch 3 (after oriented close + dominant-path + spline)")
    axes[1, 0].imshow(raw_merge, cmap="gray")
    axes[1, 0].set_title("Stage 1 all branches merged")
    axes[1, 1].imshow(cleaned_merge, cmap="gray")
    axes[1, 1].set_title("Stage 2 all branches merged")
    for a in axes.flat:
        a.axis("off")
    out = FIG_DIR / "03_stage2_clean.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 4: stage 3 — reconnect tip merging (preprocess merged vs reconnect overlap)
# ----------------------------------------------------------------------

def fig04_stage3():
    sem = read_gray(SEM_PATH)
    preprocess_merge = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / "mask_branches_merge.png") > 127
    reconnect_lbl = tifffile.imread(EXAMPLE_RUN / "3.reconnect" / "mask_reconnect_labels_dilated.tif").astype(np.int32)

    pre_overlay = overlay_on_sem(sem, preprocess_merge, (0, 220, 220), alpha=0.55)
    rec_color = colorize_labels(reconnect_lbl)
    rec_overlay = cv2.addWeighted(cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB), 0.45, rec_color, 0.55, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(pre_overlay)
    axes[0].set_title("Preprocess output (cleaned fragments on SEM)")
    axes[1].imshow(rec_overlay)
    axes[1].set_title("Stage 3: reconnected bundles (one color per ID)")
    for a in axes:
        a.axis("off")
    out = FIG_DIR / "04_stage3_reconnect.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 5: stage 4 — postprocess final smart-width rendering
# ----------------------------------------------------------------------

def fig05_stage4():
    sem = read_gray(SEM_PATH)
    reconnect_lbl = tifffile.imread(EXAMPLE_RUN / "3.reconnect" / "mask_reconnect_labels_dilated.tif").astype(np.int32)
    post_lbl = tifffile.imread(EXAMPLE_RUN / "4.postprocess" / "mask_post_labels.tif").astype(np.int32)

    rec_color = colorize_labels(reconnect_lbl)
    post_color = colorize_labels(post_lbl)
    rec_overlay = cv2.addWeighted(cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB), 0.45, rec_color, 0.55, 0)
    post_overlay = cv2.addWeighted(cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB), 0.45, post_color, 0.55, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(rec_overlay)
    axes[0].set_title(f"Stage 3 output ({len(np.unique(reconnect_lbl))-1} reconnect IDs)")
    axes[1].imshow(post_overlay)
    axes[1].set_title(f"Stage 4 final ({len(np.unique(post_lbl))-1} smart-width bundles)")
    for a in axes:
        a.axis("off")
    out = FIG_DIR / "05_stage4_postprocess.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 6: animated GIF of pipeline progression
# ----------------------------------------------------------------------

def fig06_gif():
    mask = read_gray(MASK_PATH)
    sem = read_gray(SEM_PATH)
    recon = read_gray(EXAMPLE_RUN / "1.stringart" / "reconstructed.png")
    cleaned_merge = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / "mask_branches_merge.png")
    reconnect_lbl = tifffile.imread(EXAMPLE_RUN / "3.reconnect" / "mask_reconnect_labels_dilated.tif").astype(np.int32)
    post_lbl = tifffile.imread(EXAMPLE_RUN / "4.postprocess" / "mask_post_labels.tif").astype(np.int32)
    sem_rgb = cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB)

    def caption(img: np.ndarray, text: str) -> np.ndarray:
        # add white caption strip at the top
        h, w = img.shape[:2]
        strip_h = max(28, h // 18)
        canvas = np.full((h + strip_h, w, 3), 235, dtype=np.uint8)
        if img.ndim == 2:
            canvas[strip_h:, :, :] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            canvas[strip_h:, :, :] = img
        cv2.putText(canvas, text, (10, strip_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        return canvas

    frames = []
    frames.append(caption(mask, "Input mask"))
    frames.append(caption(sem, "SEM image"))
    frames.append(caption(overlay_on_sem(sem, mask > 127, (0, 220, 220), 0.55),
                          "Mask on SEM"))
    frames.append(caption(recon, "Stage 1: reconstructed (union of per-angle branches)"))
    frames.append(caption(cleaned_merge, "Stage 2: cleaned + spline-smoothed branches"))
    rec_color = colorize_labels(reconnect_lbl)
    frames.append(caption(cv2.addWeighted(sem_rgb, 0.45, rec_color, 0.55, 0),
                          f"Stage 3: reconnect ({len(np.unique(reconnect_lbl))-1} IDs)"))
    post_color = colorize_labels(post_lbl)
    frames.append(caption(cv2.addWeighted(sem_rgb, 0.45, post_color, 0.55, 0),
                          f"Stage 4: final smart-width bundles ({len(np.unique(post_lbl))-1} IDs)"))

    out = FIG_DIR / "06_pipeline_progression.gif"
    write_gif(out, frames)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------
# Figure 6b: bundle-tracking GIF — follow one bundle through every stage
# ----------------------------------------------------------------------

def _load_multilabel_layer(npz_path: Path, target_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_layer_bool, all_layers_union_bool) from a sparse multilabel npz.

    Using the layered representation rather than the flat label image means
    overlap-absorb / first-write-wins don't truncate the bundle - we always get
    the bundle's *original* pixel footprint as written by stage 3 or stage 4.
    """
    data = np.load(npz_path)
    shape = tuple(int(v) for v in data["shape"])
    ids = data["ids"].astype(np.int64)
    indptr = data["indptr"].astype(np.int64)
    indices = data["indices"].astype(np.int64)
    union = np.zeros(shape[0] * shape[1], dtype=bool)
    target = np.zeros(shape[0] * shape[1], dtype=bool)
    for i, cid in enumerate(ids):
        seg = indices[indptr[i]:indptr[i + 1]]
        union[seg] = True
        if int(cid) == int(target_id):
            target[seg] = True
    return target.reshape(shape), union.reshape(shape)


def fig06b_bundle_track(target_id: int = 6):
    """Animate one bundle's journey through all stages.

    Stages shown:
        1. Input mask (bundle region highlighted)
        2. SEM with bundle area outlined
        3. Stage 1: per-angle branches (its dominant branch lit, others dim)
        4. Stage 1: just the dominant branch in isolation
        5. Stage 2: cleaned dominant branch
        6. Stage 3: reconnect IDs (only target ID in color, rest grey) +
                    endpoints of the target's fragments before merging
        7. Stage 4: final smart-width (only target ID in color, rest grey)
    """
    mask = read_gray(MASK_PATH)
    sem = read_gray(SEM_PATH)
    sem_rgb = cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB)

    # Stage 4 bundle from the multilabel (full layer mask, not overlap-absorbed)
    post_mb_path = EXAMPLE_RUN / "4.postprocess" / "mask_post_multilabel.npz"
    m_target, post_union = _load_multilabel_layer(post_mb_path, target_id)
    if not m_target.any():
        print(f"target id{target_id} not in post multilabel - skipping bundle GIF")
        return

    # Find dominant stage-1 branch by overlap with the post bundle
    br_dir = EXAMPLE_RUN / "1.stringart" / "branches"
    branch_files = sorted(br_dir.glob("mask_branch_*.png"),
                          key=lambda p: int(p.stem.split("_")[-1]))
    branch_imgs = {}
    overlaps = {}
    for bf in branch_files:
        idx = int(bf.stem.split("_")[-1])
        bimg = read_gray(bf) > 127
        branch_imgs[idx] = bimg
        overlaps[idx] = int((bimg & m_target).sum())
    dom_idx = max(overlaps, key=overlaps.get)
    n_b = len(branch_imgs)

    # Stage 3 bundle from the reconnect multilabel (sparse npz). Most reconnect
    # IDs preserve their numbering into postprocess, so try same id first; if
    # not present (or empty overlap), fall back to the dominant-overlap layer.
    rec_mb_path = EXAMPLE_RUN / "3.reconnect" / "mask_reconnect_multilabel.npz"
    rec_target_mask, rec_union = _load_multilabel_layer(rec_mb_path, target_id)
    rec_target_id = target_id
    if int(rec_target_mask.sum()) == 0:
        # Fallback: scan layers for one with most overlap with the post bundle
        data = np.load(rec_mb_path)
        shape = tuple(int(v) for v in data["shape"])
        ids = data["ids"].astype(np.int64)
        indptr = data["indptr"].astype(np.int64)
        indices = data["indices"].astype(np.int64)
        best_ov = 0
        for i, rid in enumerate(ids):
            seg = indices[indptr[i]:indptr[i + 1]]
            layer = np.zeros(shape[0] * shape[1], dtype=bool)
            layer[seg] = True
            layer = layer.reshape(shape)
            ov = int((layer & m_target).sum())
            if ov > best_ov:
                best_ov = ov
                rec_target_id = int(rid)
                rec_target_mask = layer

    HIGHLIGHT = (255, 50, 50)  # red

    def caption(img: np.ndarray, text: str) -> np.ndarray:
        h, w = img.shape[:2]
        strip_h = max(30, h // 18)
        out = np.full((h + strip_h, w, 3), 235, dtype=np.uint8)
        if img.ndim == 2:
            out[strip_h:, :, :] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            out[strip_h:, :, :] = img
        cv2.putText(out, text, (10, strip_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        return out

    def draw_outline(canvas: np.ndarray, mask_bool: np.ndarray, color, thickness=2):
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            cv2.drawContours(canvas, [c], -1, color, thickness, lineType=cv2.LINE_AA)
        return canvas

    frames = []

    # Frame 1: input mask, bundle region highlighted on mask
    f1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # dim non-bundle mask pixels
    bundle_mask = mask > 127
    f1[bundle_mask & ~m_target] = (110, 110, 110)
    f1[m_target] = HIGHLIGHT
    frames.append(caption(f1, f"Input mask: highlighting target bundle (final id={target_id}, {int(m_target.sum())}px)"))

    # Frame 2: SEM with bundle outline
    f2 = sem_rgb.copy()
    f2 = draw_outline(f2, m_target, HIGHLIGHT, thickness=2)
    frames.append(caption(f2, "SEM image: target bundle outlined in red"))

    # Frame 3: per-angle stringart branches in colored palette, with dominant branch fully bright and others dimmed
    cmap = plt.get_cmap("hsv")
    f3 = np.full((mask.shape[0], mask.shape[1], 3), 30, dtype=np.uint8)
    for i, bf in enumerate(branch_files):
        idx = int(bf.stem.split("_")[-1])
        color = (np.array(cmap((idx - 1) / max(1, n_b - 1))[:3]) * 255).astype(np.uint8)
        b = branch_imgs[idx]
        if idx == dom_idx:
            f3[b] = color  # full color
        else:
            f3[b] = (0.4 * color).astype(np.uint8)  # dimmed
    ang_a = (dom_idx - 1) * 15
    ang_b = dom_idx * 15
    frames.append(caption(f3, f"Stage 1: 12 per-angle branches; this bundle's dominant branch is {dom_idx} ({ang_a}-{ang_b} deg)"))

    # Frame 4: just dominant branch alone, with target bundle pixels in red
    dom_mask = branch_imgs[dom_idx]
    f4 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    f4[dom_mask] = (200, 200, 200)
    f4[dom_mask & m_target] = HIGHLIGHT
    frames.append(caption(f4, f"Stage 1: just branch {dom_idx} (red = pixels that end up in target bundle)"))

    # Frame 5: stage-2 cleaned dominant branch
    cleaned = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / f"mask_branch_{dom_idx}.png") > 127
    f5 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    f5[cleaned] = (180, 180, 220)
    f5[cleaned & m_target] = HIGHLIGHT
    frames.append(caption(f5, f"Stage 2: branch {dom_idx} after preprocess (drop specks, dominant path, B-spline)"))

    # Frame 6: stage-3 reconnect bundle from the multilabel layer (full
    # footprint, not the overlap-truncated flat label). Skeleton endpoints of
    # the target's reconnect mask are highlighted so the role of tip-tip
    # geometry in stage 3 is visible.
    f6 = sem_rgb.copy().astype(np.float32)
    rec_others = rec_union & ~rec_target_mask
    f6[rec_others] = 0.55 * f6[rec_others] + 0.45 * np.array([150, 150, 150])
    f6[rec_target_mask] = 0.4 * f6[rec_target_mask] + 0.6 * np.array(HIGHLIGHT)
    f6 = np.clip(f6, 0, 255).astype(np.uint8)
    # Skeletonise the target's reconnect mask and find endpoints (deg-1
    # skeleton nodes). Reconnect's three stages all fire on these tips.
    rec_skel = skeletonize(rec_target_mask)
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = cv2.filter2D(rec_skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) * rec_skel.astype(np.uint8)
    endpoints = np.argwhere((rec_skel > 0) & (nb == 1))
    for r, c in endpoints:
        cv2.drawMarker(f6, (int(c), int(r)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(f6, (int(c), int(r)), 7, (255, 255, 0), 1, cv2.LINE_AA)
    rec_label = f"id={rec_target_id}" if rec_target_id is not None else "?"
    frames.append(caption(f6,
        f"Stage 3: reconnect bundles; target = rec_{rec_label} (red), endpoints in yellow (tip-tip merge inputs)"))

    # Frame 7: stage-4 final bundle from the post multilabel
    f7 = sem_rgb.copy().astype(np.float32)
    other_post = post_union & ~m_target
    f7[other_post] = 0.55 * f7[other_post] + 0.45 * np.array([150, 150, 150])
    f7[m_target] = 0.30 * f7[m_target] + 0.70 * np.array(HIGHLIGHT)
    f7 = np.clip(f7, 0, 255).astype(np.uint8)
    frames.append(caption(f7, f"Stage 4: final smart-width bundles; target = id {target_id} (red), others dim"))

    out = FIG_DIR / f"06b_bundle_track_id{target_id}.gif"
    write_gif(out, frames)
    print(f"wrote {out.name} (target final id={target_id}, dom branch={dom_idx}, rec_id={rec_target_id})")


# ----------------------------------------------------------------------
# Figure 9: reconnect gate schematic — endpoints + three stages
# ----------------------------------------------------------------------

def fig09_reconnect_gates():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Three example scenarios. Each panel:
    #   - Two fragments (a long horizontal trunk + a 2nd candidate).
    #   - Tip endpoints drawn as yellow crosses.
    #   - Gap labelled with dist, opposition, forward cos.
    #   - Title shows which stage accepts/rejects it.

    def draw_fragment(ax, path_xy, color, label):
        xs, ys = zip(*path_xy)
        ax.plot(xs, ys, color=color, linewidth=10, solid_capstyle="round", alpha=0.85)
        ax.plot(xs[0], ys[0], marker="x", color="gold", markersize=14,
                markeredgewidth=3, zorder=5)
        ax.plot(xs[-1], ys[-1], marker="x", color="gold", markersize=14,
                markeredgewidth=3, zorder=5)
        ax.text(np.mean(xs), np.mean(ys) + 0.35, label, fontsize=9, ha="center",
                color=color, fontweight="bold")

    # Panel 1: stage_clear ACCEPT
    ax = axes[0]
    ax.set_xlim(-5, 5); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal"); ax.axis("off")
    draw_fragment(ax, [(-4.5, 0), (-0.6, 0)], "tab:red", "fragment A")
    draw_fragment(ax, [(0.6, 0), (4.5, 0)], "tab:blue", "fragment B")
    ax.annotate("", xy=(0.55, 0), xytext=(-0.55, 0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(0, 0.55, "dist = 1.2 px", fontsize=9, ha="center")
    ax.text(0, -2.2,
            "stage_clear ACCEPT\n"
            "dist ≤ 16, residual ≤ 6,\n"
            "opp ≥ 0.6  (here ≈ 1.0)",
            fontsize=10, ha="center", color="green", fontweight="bold")

    # Panel 2: stage_strict ACCEPT (slightly larger gap, still aligned)
    ax = axes[1]
    ax.set_xlim(-5, 5); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal"); ax.axis("off")
    draw_fragment(ax, [(-4.5, 0.3), (-1.5, 0.1)], "tab:red", "fragment A")
    draw_fragment(ax, [(1.5, -0.1), (4.5, -0.3)], "tab:blue", "fragment B")
    ax.annotate("", xy=(1.45, -0.1), xytext=(-1.45, 0.1),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(0, 0.65, "dist = 6.0 px", fontsize=9, ha="center")
    ax.text(0, -2.2,
            "stage_strict ACCEPT\n"
            "dist ≤ 15, fwd ≥ 0.85,\n"
            "opp ≥ 0.55, residual ≤ 6",
            fontsize=10, ha="center", color="green", fontweight="bold")

    # Panel 3: stage_relaxed FAIL on opposition (perpendicular T)
    ax = axes[2]
    ax.set_xlim(-5, 5); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal"); ax.axis("off")
    draw_fragment(ax, [(-4.5, 0), (-0.4, 0)], "tab:red", "fragment A")
    draw_fragment(ax, [(0.0, -0.4), (0.0, -2.8)], "tab:blue", "fragment B")
    # gap line from A tip to B tip
    ax.annotate("", xy=(0.0, -0.4), xytext=(-0.4, 0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(0.0, 0.6, "dist = 0.6 px,  opp ≈ 0", fontsize=9, ha="center")
    ax.text(0, -3.4,
            "REJECT all stages\n"
            "tip-tip distance fine,\n"
            "but inward tangents are\n"
            "perpendicular (opp < 0.4)",
            fontsize=10, ha="center", color="firebrick", fontweight="bold")

    fig.suptitle(
        "Stage 3 reconnect: tip-tip merge gates.  Yellow crosses = skeleton endpoints (tips).",
        fontsize=12,
    )
    out = FIG_DIR / "09_reconnect_gates.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ----------------------------------------------------------------------

def main():
    fig01_block_diagram()
    fig02_stage1()
    fig03_stage2()
    fig04_stage3()
    fig05_stage4()
    fig06_gif()
    # Bundle-tracking GIFs for the 5 biggest bundles in the example run
    for tid in (1, 6, 7, 8, 10):
        fig06b_bundle_track(target_id=tid)
    fig09_reconnect_gates()


if __name__ == "__main__":
    main()
