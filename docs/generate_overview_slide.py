"""Generate a single rich umbrella slide that summarizes the whole FilaSeg
pipeline at a glance. Designed to fit cleanly into a 16:9 PowerPoint slide.

Output:
    docs/figures/00_umbrella_overview.png  (1920x1080-ish, ready for slide use)

Layout:
    +----------------------------------------------------------------------+
    | TITLE BANNER                                                         |
    +----------------------------------------------------------------------+
    |                                                                      |
    | [Input]  →  [Stage 1]  →  [Stage 2]  →  [Stage 3]  →  [Stage 4]  → [Final] |
    | mask/sem    stringart      preprocess    reconnect    postprocess     labels |
    | thumb       thumb          thumb         thumb        thumb           thumb  |
    | bullets     bullets        bullets       bullets      bullets         bullets|
    |                                                                      |
    +----------------------------------------------------------------------+
    | FOOTER: outputs, tooling, key files                                  |
    +----------------------------------------------------------------------+
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
FIG_DIR = DOCS / "figures"
EXAMPLE_RUN = ROOT / "output" / "full_pipeline" / "sem_full_00000_1p66_crop512_b"
INPUT_DIR = ROOT / "input" / "sem_full_00000_1p66_crop512"

# Stage colors (match the block diagram)
COL_INPUT = "#FFEFD5"
COL_STR = "#B0E0E6"
COL_PRE = "#DDA0DD"
COL_REC = "#90EE90"
COL_POST = "#F0E68C"
COL_OUT = "#FFB6C1"
DARK = "#1f1f1f"
GREY = "#555555"
ACCENT = "#1976d2"


def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def colorize_labels(lbl: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    ids = np.unique(lbl)
    palette = {int(i): rng.integers(50, 255, size=3, dtype=np.uint8)
               for i in ids if i != 0}
    out = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    for i, c in palette.items():
        out[lbl == i] = c
    return out


def build_thumbnails():
    """Return (input_mask, branches_recon, preprocessed_merge, reconnect_color,
    final_color, final_overlay) as RGB uint8 arrays."""
    mask = read_gray(INPUT_DIR / "mask.png")
    sem = read_gray(INPUT_DIR / "sem.png")
    recon = read_gray(EXAMPLE_RUN / "1.stringart" / "reconstructed.png")
    pre_merge = read_gray(EXAMPLE_RUN / "2.preprocess" / "branches" / "mask_branches_merge.png")
    reconnect_lbl = tifffile.imread(EXAMPLE_RUN / "3.reconnect" / "mask_reconnect_labels_dilated.tif").astype(np.int32)
    post_lbl = tifffile.imread(EXAMPLE_RUN / "4.postprocess" / "mask_post_labels.tif").astype(np.int32)

    sem_rgb = cv2.cvtColor(sem, cv2.COLOR_GRAY2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    recon_rgb = cv2.cvtColor(recon, cv2.COLOR_GRAY2RGB)
    pre_rgb = cv2.cvtColor(pre_merge, cv2.COLOR_GRAY2RGB)
    rec_color = colorize_labels(reconnect_lbl)
    rec_overlay = cv2.addWeighted(sem_rgb, 0.45, rec_color, 0.55, 0)
    post_color = colorize_labels(post_lbl)
    post_overlay = cv2.addWeighted(sem_rgb, 0.45, post_color, 0.55, 0)

    # Input column shows mask + SEM stacked
    h, w = mask.shape
    stacked = np.zeros((h * 2 + 4, w, 3), dtype=np.uint8)
    stacked[:h, :, :] = mask_rgb
    stacked[h:h + 4, :, :] = 240
    stacked[h + 4:, :, :] = sem_rgb

    return {
        "input": stacked,
        "stringart": recon_rgb,
        "preprocess": pre_rgb,
        "reconnect": rec_overlay,
        "postprocess": post_overlay,
        "final": post_overlay,
        "post_lbl_count": int(len(np.unique(post_lbl)) - 1),
        "rec_lbl_count": int(len(np.unique(reconnect_lbl)) - 1),
    }


def stage_panel(ax, title, color, thumb_rgb, bullets, badge_text=None,
                tag_color=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Card background
    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.012",
                          linewidth=2.0,
                          edgecolor=tag_color or DARK,
                          facecolor=color, alpha=0.32,
                          transform=ax.transAxes)
    ax.add_patch(card)

    # Title strip
    title_strip = FancyBboxPatch(
        (0.02, 0.84), 0.96, 0.13,
        boxstyle="round,pad=0.008",
        linewidth=0, facecolor=tag_color or DARK, alpha=0.95,
        transform=ax.transAxes,
    )
    ax.add_patch(title_strip)
    ax.text(0.5, 0.905, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color="white",
            transform=ax.transAxes)
    if badge_text:
        ax.text(0.5, 0.86, badge_text, ha="center", va="center",
                fontsize=9.5, color="#f0f0f0", style="italic",
                transform=ax.transAxes)

    # Thumbnail
    ax_thumb = ax.inset_axes([0.07, 0.40, 0.86, 0.40])
    ax_thumb.imshow(thumb_rgb)
    ax_thumb.set_xticks([]); ax_thumb.set_yticks([])
    for spine in ax_thumb.spines.values():
        spine.set_edgecolor("#888"); spine.set_linewidth(0.8)

    # Bullets
    y = 0.34
    for b in bullets:
        ax.text(0.07, y, "• " + b, ha="left", va="top",
                fontsize=10.5, color=DARK, transform=ax.transAxes,
                wrap=True)
        y -= 0.075


def main():
    thumbs = build_thumbnails()

    fig = plt.figure(figsize=(20, 11), facecolor="white")
    gs = GridSpec(
        nrows=3, ncols=11,
        # title row | spacer between input | 4 stage cards | spacer | output | spacer
        height_ratios=[0.18, 1.0, 0.16],
        width_ratios=[
            0.95,    # input
            0.18,    # arrow
            1.0,     # stage 1
            0.18,    # arrow
            1.0,     # stage 2
            0.18,    # arrow
            1.0,     # stage 3
            0.18,    # arrow
            1.0,     # stage 4
            0.18,    # arrow
            0.95,    # final
        ],
        hspace=0.06, wspace=0.05,
    )

    # ─── Title banner ─────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_title.transAxes,
                                       facecolor=ACCENT))
    ax_title.text(0.012, 0.62, "FilaSeg",
                  ha="left", va="center", fontsize=34, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    ax_title.text(0.012, 0.22,
                  "Binary mask → vectorize → clean → reconnect → smart-width render → instance labels",
                  ha="left", va="center", fontsize=16, color="white",
                  transform=ax_title.transAxes)
    ax_title.text(0.985, 0.5,
                  "input/sem_full_00000_1p66_crop512  →  "
                  f"{thumbs['post_lbl_count']} bundle instances",
                  ha="right", va="center", fontsize=13, color="#e3f0ff",
                  transform=ax_title.transAxes, style="italic")

    # ─── 6 columns: input + 4 stages + final ──────────────────────────────
    panels = [
        (gs[1, 0],  "Input",        COL_INPUT, thumbs["input"],
         ["binary mask + SEM image",
          "from microscopy / segmentation",
          "1.66 µm/px default scale"],
         "raw data", "#a07d2e"),
        (gs[1, 2],  "1. stringart", COL_STR,   thumbs["stringart"],
         ["tiled probabilistic Hough",
          "two gates: new-pix + density",
          "12 angle bins (15° wide)",
          "multi-grid voting"],
         "stringart_tiles.py", "#1e6091"),
        (gs[1, 4],  "2. preprocess", COL_PRE,  thumbs["preprocess"],
         ["per-branch threshold",
          "drop tiny CCs",
          "oriented morph close",
          "dominant 2-tip path + B-spline"],
         "preprocess_stringart_branches.py", "#7a3f7a"),
        (gs[1, 6],  "3. reconnect", COL_REC,   thumbs["reconnect"],
         ["staged tip-tip merges",
          "stage_clear → strict → relaxed",
          "smooth passes until fixpoint",
          f"output: {thumbs['rec_lbl_count']} reconnect IDs"],
         "reconnect_run.py (staged)", "#2e7d32"),
        (gs[1, 8],  "4. postprocess", COL_POST, thumbs["postprocess"],
         ["re-skeletonize each layer",
          "moving-window smooth",
          "smart-width from SEM edges",
          "overlap absorb + occlusion trim"],
         "post_process_reconnect.py", "#a07d2e"),
        (gs[1, 10], "Final",         COL_OUT,  thumbs["final"],
         ["labels.tif (instance per ID)",
          "color preview + overlay",
          "DEM JSON (centerlines, widths)",
          f"{thumbs['post_lbl_count']} bundles in this run"],
         "output/.../final", "#b04a5a"),
    ]
    for spec, *args in panels:
        ax = fig.add_subplot(spec)
        stage_panel(ax, *args)

    # ─── Connecting arrows (in their thin column subplots) ────────────────
    for col in (1, 3, 5, 7, 9):
        ax_arrow = fig.add_subplot(gs[1, col])
        ax_arrow.axis("off")
        ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
        ax_arrow.add_patch(FancyArrowPatch(
            (0.05, 0.5), (0.95, 0.5),
            mutation_scale=24, lw=2.5, color="#333",
            arrowstyle="->",
        ))

    # ─── Footer: utilities, regenerators, troubleshooting ─────────────────
    ax_foot = fig.add_subplot(gs[2, :])
    ax_foot.axis("off")
    ax_foot.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_foot.transAxes,
                                      facecolor="#f5f5f5"))
    # Three footer columns
    cols = [
        ("End-to-end runner",
         "Tools/run_full_sem_pipeline.py  —  resolves µm/px, scales all pixel params, "
         "rewrites reconnect_config.yaml per-run, copies outputs to final/."),
        ("Troubleshooting CLI",
         "Tools/troubleshoot_reconnect.py  —  diagnose-missed, compare-followups, "
         "check-coords, trace-component, visualize-pairs, trace-evolution."),
        ("Interactive review",
         "Tools/visualize_ids.py  —  click bundles, paint corrections, lock/select IDs. "
         "Tools/manualID_correct.py  —  per-ID manual cleanup."),
    ]
    x0 = 0.02
    col_w = 0.31
    for i, (label, body) in enumerate(cols):
        x = x0 + i * (col_w + 0.013)
        ax_foot.text(x, 0.74, label, ha="left", va="top",
                     fontsize=11, fontweight="bold", color=ACCENT,
                     transform=ax_foot.transAxes)
        ax_foot.text(x, 0.50, body, ha="left", va="top",
                     fontsize=9.5, color=DARK,
                     transform=ax_foot.transAxes, wrap=True)

    out = FIG_DIR / "00_umbrella_overview.png"
    # 110 dpi keeps the image under ~2000px wide while still looking crisp on a
    # widescreen slide (the slide projector typically renders at ~1920×1080).
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
