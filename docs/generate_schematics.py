"""Generate per-step schematic PNGs explaining each pipeline stage.

Output structure:
    docs/figures/schematics/
        01_stringart/    *.png
        02_preprocess/   *.png
        03_reconnect/    *.png
        04_postprocess/  *.png

Each PNG is a focused matplotlib figure showing one specific concept (e.g. one
reconnect stage gate, or one preprocess step). They are intentionally
hand-drawn schematics, not screenshots of real data, so the geometry is clean
and a viewer can see exactly what each step does.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                 Polygon, Rectangle)
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
FIG_BASE = ROOT / "figures" / "schematics"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

ACCENT = "#1976d2"
RED = "#d62728"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
GREY = "#9e9e9e"
DARK = "#212121"


def setup_axes(ax, lim=4.0, title=None, subtitle=None):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0, lim - 0.4, subtitle, ha="center", va="top",
                fontsize=9.5, color="#555")


def save_fig(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIG_BASE.parent.parent)}")


def draw_filament(ax, path_xy, color=ACCENT, lw=10, alpha=0.9, **kw):
    xs, ys = zip(*path_xy)
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round",
            alpha=alpha, **kw)


def draw_tip(ax, xy, color="gold", size=14):
    ax.plot(xy[0], xy[1], marker="x", color=color, markersize=size,
            markeredgewidth=3, zorder=10)
    ax.add_patch(Circle(xy, size / 30, fill=False, color=color, lw=1.5, zorder=10))


def annotate_arrow(ax, p0, p1, label=None, color="black", lw=1.2,
                   text_offset=(0, 0.25), text_color=None):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + text_offset[0], my + text_offset[1], label,
                ha="center", va="bottom", fontsize=9,
                color=text_color or color)


# ===========================================================================
# 01. STRINGART (traditional tiled Hough path)
# ===========================================================================

STR_DIR = FIG_BASE / "01_stringart"


def stringart_tile_grid():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    setup_axes(ax, lim=5,
               title="Stringart step 1: tile the mask",
               subtitle="Each tile is processed independently by probabilistic Hough.")

    # background mask: a few diagonal "filaments"
    for (x0, y0, x1, y1) in [(-4.6, 4.1, 4.5, 1.0), (-4.6, -3.0, 3.5, 4.0),
                              (-3.2, 4.5, 1.5, -4.5), (-1.5, 4.4, -1.2, -4.5),
                              (-4.5, 0.6, 4.5, -1.8)]:
        ax.plot([x0, x1], [y0, y1], color="#bbbbbb", linewidth=2.5, alpha=0.85, zorder=1)

    # tile grid (4x4)
    tile = 2.2
    n = 4
    for i in range(n + 1):
        coord = -4.4 + i * tile
        ax.plot([-4.4, -4.4 + n * tile], [coord, coord], color=ACCENT, lw=1)
        ax.plot([coord, coord], [-4.4, -4.4 + n * tile], color=ACCENT, lw=1)

    # highlight one tile
    ax.add_patch(Rectangle((-4.4 + tile, -4.4 + 2 * tile), tile, tile,
                            fill=False, edgecolor=RED, lw=2.5, zorder=3))
    ax.text(-4.4 + 1.5 * tile, -4.4 + 2 * tile + tile + 0.25,
            "one tile", color=RED, fontsize=10, ha="center", fontweight="bold")

    save_fig(fig, STR_DIR / "01_tile_grid.png")


def stringart_hough_in_tile():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    setup_axes(ax, lim=4,
               title="Stringart step 2: probabilistic Hough inside one tile",
               subtitle="Candidate segments are sampled from the mask pixels.")

    # tile boundary
    ax.add_patch(Rectangle((-3.6, -3.6), 7.2, 7.2, fill=False, edgecolor=ACCENT, lw=2.5))
    # mask pixels (a noisy diagonal)
    rng = np.random.default_rng(2)
    xs0 = np.linspace(-3.2, 3.2, 80)
    ys0 = 0.6 * xs0 + rng.normal(0, 0.18, len(xs0))
    ax.scatter(xs0, ys0, s=14, color="#777")
    # specks
    sxs = rng.uniform(-3.2, 3.2, 25)
    sys_ = rng.uniform(-3.2, 3.2, 25)
    ax.scatter(sxs, sys_, s=8, color="#aaa", alpha=0.5)

    # 3 candidate Hough segments
    cands = [
        ((-3.0, -1.4), (3.0, 2.4), "candidate 1"),
        ((-2.7, -1.6), (2.5, 1.2), "candidate 2"),
        ((-1.5, 2.8), (1.8, -2.5), "candidate 3 (off mask)"),
    ]
    for (a, b, label), color in zip(cands, [RED, ORANGE, GREEN]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2.5, alpha=0.9)
        ax.text(b[0] + 0.2, b[1] + 0.1, label, fontsize=8, color=color)

    save_fig(fig, STR_DIR / "02_hough_in_tile.png")


def stringart_gate_newpix():
    fig, ax = plt.subplots(figsize=(7, 5))
    setup_axes(ax, lim=4.2,
               title="Stringart gate A: new-pixels added",
               subtitle="A candidate must add ≥ min_accept_newpix previously-unclaimed mask pixels.")

    # already-claimed (green) skeleton + remaining (grey) mask pixels
    rng = np.random.default_rng(7)
    claimed = np.array([[x, 0.5 * x] for x in np.linspace(-3.5, -0.2, 14)])
    remaining = np.array([[x, 0.5 * x + rng.normal(0, 0.15)] for x in np.linspace(0.1, 3.5, 14)])

    ax.scatter(claimed[:, 0], claimed[:, 1], s=80, color=GREEN, edgecolor="black",
               label="already covered by accepted lines", zorder=3)
    ax.scatter(remaining[:, 0], remaining[:, 1], s=80, color="#999", edgecolor="black",
               label="mask pixels still uncovered", zorder=3)

    # candidate line
    ax.plot([-3.7, 3.7], [-1.85, 1.85], color=RED, lw=2.5, label="candidate Hough line")
    ax.text(3.0, 2.4, "12 new px added\n(threshold = 6) → PASS",
            color=GREEN, fontsize=10, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaffea", edgecolor=GREEN))
    ax.legend(loc="lower left", fontsize=8)
    save_fig(fig, STR_DIR / "03_gate_newpix.png")


def stringart_gate_density():
    fig, ax = plt.subplots(figsize=(7, 5))
    setup_axes(ax, lim=4.2,
               title="Stringart gate B: support density",
               subtitle="Rejects long fake chords that mostly cross background.")

    # a real filament (dense pixels along a short segment)
    rng = np.random.default_rng(11)
    xs = np.linspace(-3.5, -0.5, 40)
    ys = 0.15 + 0.4 * xs + rng.normal(0, 0.05, len(xs))
    ax.scatter(xs, ys, s=20, color="#888")

    # accepted candidate hugs the filament
    ax.plot([-3.6, -0.4], [0.15 - 1.4, 0.15 - 0.2], color=GREEN, lw=2.5)
    ax.text(-2.0, -1.7, "density 0.85 ≥ 0.45 → PASS",
            color=GREEN, fontsize=9, ha="center", fontweight="bold")

    # rejected candidate cuts across the whole tile, barely touches the mask
    ax.plot([-3.6, 3.6], [3.5, -3.5], color=RED, lw=2.5, ls="--")
    ax.text(2.5, 2.7,
            "density 0.08 < 0.45 → REJECT\n(fake long chord across background)",
            color=RED, fontsize=9, ha="center", fontweight="bold")

    save_fig(fig, STR_DIR / "04_gate_density.png")


def stringart_angle_binning():
    """Clean two-panel layout:
        Top  : fan of 12 short lines at each bin's center angle, color-coded;
               huge bin-id label by every line.
        Bot  : horizontal "legend bar" - 12 colored cells in a row showing
               the range each bin covers, plus one worked example callout.

    The previous wheel-style figure crammed all 12 wedges + tick marks +
    example line in tiny space and was unreadable. This layout gives each
    element its own large area and avoids overlapping text.
    """
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.35)
    ax_fan = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    cmap = plt.get_cmap("hsv")
    n_bins = 12
    step = 15

    # ─── Top: fan of 12 oriented lines ────────────────────────────────────
    ax_fan.set_xlim(-1.5, 1.5)
    ax_fan.set_ylim(-0.05, 1.35)
    ax_fan.set_aspect("equal")
    ax_fan.axis("off")
    ax_fan.set_title("Every accepted Hough line falls into ONE of 12 angle bins (15° wide each).",
                     fontsize=13, fontweight="bold", pad=10)

    L = 1.05
    for i in range(n_bins):
        mid_deg = i * step + step / 2
        rad = math.radians(mid_deg)
        x = L * math.cos(rad); y = L * math.sin(rad)
        color = cmap(i / n_bins)
        # Half-line from origin to the +tip (the visible end on top half)
        ax_fan.plot([0, x], [0, y], color=color, linewidth=7,
                    solid_capstyle="round")
        # Bin-id badge on the tip
        bx = (L + 0.12) * math.cos(rad)
        by = (L + 0.12) * math.sin(rad)
        ax_fan.add_patch(Circle((bx, by), 0.085,
                                facecolor="white", edgecolor=color, lw=2,
                                zorder=10))
        ax_fan.text(bx, by, str(i + 1), ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold", zorder=11)
    # baseline
    ax_fan.plot([-1.15, 1.15], [0, 0], color="#888", lw=1)
    # degree markers below the baseline
    for deg_label, x_pos in [("0°", 1.18), ("90°", 0), ("180°", -1.18)]:
        ax_fan.text(x_pos, -0.04, deg_label, ha="center", va="top",
                    fontsize=10, color="#555")

    # ─── Bottom: horizontal legend bar ────────────────────────────────────
    ax_bar.set_xlim(0, n_bins)
    ax_bar.set_ylim(-1.3, 1.4)
    ax_bar.axis("off")

    cell_h = 1.0
    for i in range(n_bins):
        color = cmap(i / n_bins)
        rect = Rectangle((i, 0), 1, cell_h,
                         facecolor=color, edgecolor="white", linewidth=2)
        ax_bar.add_patch(rect)
        # bin number (white, centered)
        ax_bar.text(i + 0.5, 0.7, f"bin {i + 1}",
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="white")
        # angle range (white, centered, below number)
        ax_bar.text(i + 0.5, 0.3, f"{i*step}° – {(i+1)*step}°",
                    ha="center", va="center", fontsize=9, color="white")

    ax_bar.text(n_bins / 2, 1.25,
                "Legend: each colored cell is one per-angle output image",
                ha="center", fontsize=10.5, color=DARK, fontweight="bold")

    # Worked example below the bar: a single accepted Hough line at 37°
    # falls into bin 3 (30-45°). Draw the line + arrow to the right cell.
    example_angle = 37
    color_ex = cmap(2 / n_bins)  # bin index 2 = bin 3
    # Mini line at center-bottom
    line_x = 2.0
    line_y = -0.75
    line_L = 0.9
    a_rad = math.radians(example_angle)
    ax_bar.plot([line_x - line_L * math.cos(a_rad),
                 line_x + line_L * math.cos(a_rad)],
                [line_y - line_L * math.sin(a_rad),
                 line_y + line_L * math.sin(a_rad)],
                color=color_ex, lw=6, solid_capstyle="round")
    ax_bar.text(line_x, line_y - 0.55, f"example line, θ = {example_angle}°",
                ha="center", fontsize=10, color=DARK)
    # arrow from the example to cell 3
    ax_bar.annotate("", xy=(2.5, -0.05), xytext=(2.5, line_y + 0.4),
                    arrowprops=dict(arrowstyle="->", color=color_ex, lw=2))
    ax_bar.text(3.4, line_y, "→  goes into bin 3", fontsize=11,
                color=color_ex, fontweight="bold", va="center")

    fig.suptitle("Stringart step 4: assign each accepted line to one angle bin",
                 fontsize=14, fontweight="bold", y=0.98)
    save_fig(fig, STR_DIR / "05_angle_binning.png")


def stringart_multi_grid_voting():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    grids = [(0, 0), (8, 0), (0, 8), (8, 8)]
    for i, (ox, oy) in enumerate(grids):
        ax = axes[i]
        ax.set_xlim(-1, 32); ax.set_ylim(-1, 32); ax.set_aspect("equal"); ax.axis("off")
        # background filaments
        ax.plot([2, 30], [4, 26], color="#bbb", lw=2.5)
        ax.plot([0, 25], [22, 5], color="#bbb", lw=2.5)
        # grid lines
        for g in range(0, 33, 16):
            ax.plot([ox + g - 16, ox + g - 16], [-1, 32], color=ACCENT, lw=0.7, alpha=0.6)
            ax.plot([-1, 32], [oy + g - 16, oy + g - 16], color=ACCENT, lw=0.7, alpha=0.6)
        ax.set_title(f"grid origin\noffset ({ox},{oy})", fontsize=9)

    fig.suptitle(
        "Stringart multi-grid voting: same input run at N grid origins; "
        "keep only pixels that appear in ≥ tile_grid_vote_min outputs.",
        fontsize=11,
    )
    save_fig(fig, STR_DIR / "06_multi_grid_voting.png")


def gen_stringart():
    print("stringart...")
    stringart_tile_grid()
    stringart_hough_in_tile()
    stringart_gate_newpix()
    stringart_gate_density()
    stringart_angle_binning()
    stringart_multi_grid_voting()


# ===========================================================================
# 02. PREPROCESS
# ===========================================================================

PRE_DIR = FIG_BASE / "02_preprocess"


def preprocess_threshold():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    # left: grayscale-ish input
    rng = np.random.default_rng(3)
    xs = np.linspace(-3.5, 3.5, 80)
    ys = 0.3 * xs
    img = rng.uniform(0, 0.4, (50, 50))
    for x, y in zip(xs, ys):
        cx = int(25 + x * 5); cy = int(25 + y * 5)
        if 0 <= cy < 50 and 0 <= cx < 50:
            img[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.85 + rng.uniform(0, 0.15)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input: grayscale branch PNG\n(stringart per-angle output)", fontsize=11)
    axes[0].axis("off")

    binar = (img > 0.5).astype(np.uint8) * 255
    axes[1].imshow(binar, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("After threshold (--pre-bin-threshold = 127)\nbinary mask only", fontsize=11)
    axes[1].axis("off")

    fig.suptitle("Preprocess step 1: threshold the grayscale branch", fontsize=12, fontweight="bold")
    save_fig(fig, PRE_DIR / "01_threshold.png")


def preprocess_drop_specks():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax in axes:
        setup_axes(ax, lim=4)

    # before: filaments + specks
    draw_filament(axes[0], [(-3.4, 1.1), (3.0, -2.3)], color=DARK, lw=12, alpha=0.95)
    draw_filament(axes[0], [(-2.4, -2.0), (3.5, 2.0)], color=DARK, lw=12, alpha=0.95)
    rng = np.random.default_rng(13)
    for _ in range(18):
        x, y = rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 3.5)
        sz = rng.uniform(0.05, 0.18)
        axes[0].add_patch(Circle((x, y), sz, color=DARK))
    axes[0].set_title("Before: filaments + specks", fontsize=11)

    # after: specks gone
    draw_filament(axes[1], [(-3.4, 1.1), (3.0, -2.3)], color=DARK, lw=12, alpha=0.95)
    draw_filament(axes[1], [(-2.4, -2.0), (3.5, 2.0)], color=DARK, lw=12, alpha=0.95)
    axes[1].set_title("After: CCs < min_component_area dropped", fontsize=11)
    axes[1].text(0, -3.7, "tiny disconnected blobs removed",
                 ha="center", color=GREEN, fontsize=9, fontweight="bold")

    fig.suptitle("Preprocess step 2: drop small connected components", fontsize=12, fontweight="bold")
    save_fig(fig, PRE_DIR / "02_drop_specks.png")


def preprocess_oriented_close():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax in axes:
        setup_axes(ax, lim=4)

    # before: filament with gap
    seg_a = [(-3.4, 2.4), (-0.6, 0.4)]
    seg_b = [(0.5, -0.3), (3.4, -2.4)]
    draw_filament(axes[0], seg_a, color=DARK, lw=11)
    draw_filament(axes[0], seg_b, color=DARK, lw=11)
    axes[0].add_patch(Circle((-0.05, 0.05), 0.6, fill=False, edgecolor=RED, lw=2, ls=":"))
    axes[0].text(-0.05, 1.0, "small gap", color=RED, ha="center", fontsize=9)
    # oriented kernel hint
    axes[0].plot([-0.4, 0.4], [0.3, -0.1], color=ACCENT, lw=2.5)
    axes[0].text(2.6, 2.4, "oriented kernel\nlength = --pre-line-close-len",
                 color=ACCENT, fontsize=8, ha="center")
    axes[0].set_title("Before: one filament with a small gap", fontsize=11)

    # after: gap closed
    draw_filament(axes[1], [(-3.4, 2.4), (3.4, -2.4)], color=DARK, lw=11)
    axes[1].set_title("After: oriented close bridges the gap", fontsize=11)
    axes[1].text(0, -3.7, "morph close uses a line kernel aligned to the branch angle",
                 ha="center", color=GREEN, fontsize=8.5, style="italic")

    fig.suptitle("Preprocess step 3: oriented morphological close", fontsize=12, fontweight="bold")
    save_fig(fig, PRE_DIR / "03_oriented_close.png")


def preprocess_dominant_path():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5))
    for ax in axes:
        setup_axes(ax, lim=4)

    # 4-arm skeleton with the long axis from upper-left to lower-right (the
    # actual skeleton diameter via double-BFS picks these two tips). Two
    # shorter side branches off the center are the "spurs".
    center = (0, 0)
    long_arm_a = (-3.3, 2.6)   # longest arm 1 (UL)
    long_arm_b = (3.3, -2.6)   # longest arm 2 (LR)  -> path A-center-B is the diameter
    short_arm_c = (-1.2, -2.0) # spur 1
    short_arm_d = (2.4, 1.2)   # spur 2

    arms = [long_arm_a, long_arm_b, short_arm_c, short_arm_d]

    # ---- left panel: input with all 4 tips equally weighted ----
    for tip in arms:
        draw_filament(axes[0], [center, tip], color=DARK, lw=11)
        draw_tip(axes[0], tip)
    # mark arm lengths
    for tip in arms:
        L = math.hypot(tip[0], tip[1])
        mx, my = tip[0] * 0.55, tip[1] * 0.55
        axes[0].text(mx + 0.15, my + 0.2, f"{L:.1f}", fontsize=8, color="#555")
    axes[0].set_title("Before: 4-tip skeleton CC\n(numbers = arm length, in arbitrary units)",
                      fontsize=10)

    # ---- right panel: dominant path = longest tip-to-tip route ----
    # The dominant path is the skeleton's diameter: BFS-farthest from any
    # vertex, then BFS-farthest from that node. In a star like this it picks
    # the two longest arms; the side branches become spurs that are dropped.
    draw_filament(axes[1], [long_arm_a, center], color=DARK, lw=11)
    draw_filament(axes[1], [center, long_arm_b], color=DARK, lw=11)
    draw_tip(axes[1], long_arm_a)
    draw_tip(axes[1], long_arm_b)
    # ghosts of dropped (shorter) arms
    for tip in (short_arm_c, short_arm_d):
        draw_filament(axes[1], [center, tip], color="#cccccc", lw=11, alpha=0.55)
    # label the diameter
    L_long = math.hypot(long_arm_a[0], long_arm_a[1]) + math.hypot(long_arm_b[0], long_arm_b[1])
    axes[1].text(0, 3.4,
                 f"skeleton diameter = {math.hypot(*long_arm_a):.1f} + "
                 f"{math.hypot(*long_arm_b):.1f} = {L_long:.1f}",
                 ha="center", fontsize=9, color=GREEN, fontweight="bold")
    axes[1].set_title("After --pre-clean-to-path:\ndiameter (longest endpoint-to-endpoint route) survives",
                      fontsize=10)
    axes[1].text(0, -3.7,
                 "two-pass BFS picks the two farthest tips along the skeleton;\n"
                 "shorter arms (spurs) are discarded",
                 ha="center", color="#555", fontsize=8.5, style="italic")

    fig.suptitle("Preprocess step 4: dominant-path reduction",
                 fontsize=12, fontweight="bold")
    save_fig(fig, PRE_DIR / "04_dominant_path.png")


def preprocess_bspline():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax in axes:
        setup_axes(ax, lim=4)

    rng = np.random.default_rng(5)
    ts = np.linspace(0, 1, 30)
    base_x = -3 + 6 * ts
    base_y = 1.5 * np.sin(2 * ts * np.pi - 0.6) + rng.normal(0, 0.18, len(ts))

    axes[0].plot(base_x, base_y, color=DARK, lw=2.5, marker="o", markersize=4)
    axes[0].set_title("Before: jagged dominant-path polyline", fontsize=11)

    # smooth
    from scipy.interpolate import splprep, splev
    tck, _ = splprep([base_x, base_y], s=2.0, k=3)
    u = np.linspace(0, 1, 200)
    sx, sy = splev(u, tck)
    axes[1].plot(sx, sy, color=ACCENT, lw=3.5)
    axes[1].plot(base_x, base_y, color=DARK, lw=0, marker="o", markersize=3, alpha=0.4)
    axes[1].set_title("After: parametric B-spline fit", fontsize=11)
    axes[1].text(0, -3.7,
                 "degree=--pre-fit-degree, smoothing=--pre-fit-smoothing\n"
                 "yields a physically smooth curve before reconnect",
                 ha="center", color=GREEN, fontsize=8.5, style="italic")

    fig.suptitle("Preprocess step 5: B-spline smoothing", fontsize=12, fontweight="bold")
    save_fig(fig, PRE_DIR / "05_bspline.png")


def gen_preprocess():
    print("preprocess...")
    preprocess_threshold()
    preprocess_drop_specks()
    preprocess_oriented_close()
    preprocess_dominant_path()
    preprocess_bspline()


# ===========================================================================
# 03. RECONNECT
# ===========================================================================

REC_DIR = FIG_BASE / "03_reconnect"


def _draw_reconnect_panel(ax, frag_a, frag_b, dist, opp_text, fwd_text=None,
                          residual_text=None, verdict=None, verdict_color="green",
                          extra_notes=None, title=None, sub=None):
    setup_axes(ax, lim=4.2, title=title, subtitle=sub)
    draw_filament(ax, frag_a, color=RED, lw=11)
    draw_filament(ax, frag_b, color=ACCENT, lw=11)
    for p in (frag_a[0], frag_a[-1], frag_b[0], frag_b[-1]):
        draw_tip(ax, p)
    # gap arrow between inner tips (frag_a[-1] to frag_b[0])
    annotate_arrow(ax, frag_a[-1], frag_b[0], label=f"dist = {dist}",
                   color="black", lw=1.2)

    lines = []
    if opp_text:
        lines.append(f"opposition: {opp_text}")
    if fwd_text:
        lines.append(f"forward cos: {fwd_text}")
    if residual_text:
        lines.append(f"line residual: {residual_text}")
    if lines:
        ax.text(0, -2.6, "\n".join(lines), ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                          edgecolor="#bbb"))
    if extra_notes:
        ax.text(0, 3.4, extra_notes, ha="center", fontsize=8.5, color="#555",
                style="italic")
    if verdict:
        ax.text(0, -3.7, verdict, ha="center", fontsize=11,
                fontweight="bold", color=verdict_color)


def reconnect_stage_clear_accept():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_reconnect_panel(
        ax,
        frag_a=[(-3.7, 0), (-0.6, 0)],
        frag_b=[(0.6, 0), (3.7, 0)],
        dist="1.2 px",
        opp_text="≈ 1.0  (≥ 0.6 ✓)",
        residual_text="0.3 px  (≤ 6 ✓)",
        verdict="stage_clear ACCEPT",
        verdict_color=GREEN,
        title="stage_clear — accept",
        sub="Tiny gap, perfectly aligned inward tangents.",
    )
    save_fig(fig, REC_DIR / "01_stage_clear_accept.png")


def reconnect_stage_clear_reject_dist():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_reconnect_panel(
        ax,
        frag_a=[(-3.7, 0), (-2.4, 0)],
        frag_b=[(2.5, 0), (3.7, 0)],
        dist="18.5 px",
        opp_text="≈ 1.0 ✓",
        residual_text="0.0 px ✓",
        verdict="REJECT (dist > 16)",
        verdict_color=RED,
        extra_notes="Geometry is perfect, but gap exceeds clear_merge_max_dist_px.",
        title="stage_clear — reject on distance",
        sub="Too far apart for the cheapest gate; will be retried in later stages.",
    )
    save_fig(fig, REC_DIR / "02_stage_clear_reject_distance.png")


def reconnect_stage_strict_accept():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_reconnect_panel(
        ax,
        frag_a=[(-3.7, 0.3), (-1.2, 0.1)],
        frag_b=[(1.2, -0.1), (3.7, -0.3)],
        dist="6.0 px",
        opp_text="≈ 0.96  (≥ 0.55 ✓)",
        fwd_text="0.99 / 0.99  (≥ 0.85 ✓)",
        residual_text="0.5 px  (≤ 6 ✓)",
        verdict="stage_strict ACCEPT",
        verdict_color=GREEN,
        title="stage_strict — accept",
        sub="Moderate gap; full forward/opposition/residual checks pass.",
    )
    save_fig(fig, REC_DIR / "03_stage_strict_accept.png")


def reconnect_stage_strict_reject_opposition():
    fig, ax = plt.subplots(figsize=(7, 6.0))
    _draw_reconnect_panel(
        ax,
        frag_a=[(-3.7, 0), (-0.3, 0)],
        frag_b=[(0, -0.3), (0, -3.4)],
        dist="0.4 px",
        opp_text="≈ 0  (< 0.55 ✗)",
        fwd_text="0.01 / 0.01  (< 0.85 ✗)",
        residual_text="2.1 px",
        verdict="REJECT (opposition / forward)",
        verdict_color=RED,
        extra_notes="Classic T-junction: tips touch, but tangents are perpendicular.",
        title="stage_strict — reject on opposition",
        sub="Distance alone isn't enough; alignment matters.",
    )
    save_fig(fig, REC_DIR / "04_stage_strict_reject_opposition.png")


def reconnect_stage_relaxed_accept():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_reconnect_panel(
        ax,
        frag_a=[(-3.7, 0.6), (-1.8, 0.2)],
        frag_b=[(1.8, -0.2), (3.7, -0.6)],
        dist="32.0 px",
        opp_text="≈ 0.85  (≥ 0.4 ✓)",
        fwd_text="0.88 / 0.88  (≥ 0.72 ✓)",
        residual_text="3.0 px  (≤ 16 ✓)",
        verdict="stage_relaxed ACCEPT",
        verdict_color=GREEN,
        title="stage_relaxed — accept",
        sub="Wider gap, but alignment is still convincing.",
    )
    save_fig(fig, REC_DIR / "05_stage_relaxed_accept.png")


def reconnect_stage_relaxed_reject_residual():
    """Two parallel offset fragments: tangent lines run far from the gap midpoint.

    The schematic numbers (left in pixels) are 12 × the matplotlib units, so
    a unit distance of 1.5 in the figure reads as 18 px in the labels - large
    enough to fail stage_relaxed's residual_max = 16.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    setup_axes(
        ax, lim=4.5,
        title="stage_relaxed — reject on line residual",
        subtitle="Line residual = avg perpendicular distance from the gap midpoint to each tip's tangent line.",
    )

    # Two parallel fragments offset in y. Tangent lines are y = +1.5 and y = -1.5.
    frag_a = [(-3.8, 1.5), (-0.6, 1.5)]   # horizontal, upper
    frag_b = [(0.6, -1.5), (3.8, -1.5)]   # horizontal, lower
    tip_a = frag_a[-1]
    tip_b = frag_b[0]
    draw_filament(ax, frag_a, color=RED, lw=11)
    draw_filament(ax, frag_b, color=ACCENT, lw=11)
    for p in (frag_a[0], frag_a[-1], frag_b[0], frag_b[-1]):
        draw_tip(ax, p)

    # Outward tangents at the inner tips
    out_a = (1.0, 0.0)    # tip A points right
    out_b = (-1.0, 0.0)   # tip B points left
    ray_len = 4.0
    # Dashed tangent ray extensions
    ax.plot([tip_a[0], tip_a[0] + out_a[0] * ray_len],
            [tip_a[1], tip_a[1] + out_a[1] * ray_len],
            color=RED, lw=1.6, ls="--", alpha=0.85)
    ax.plot([tip_b[0], tip_b[0] + out_b[0] * ray_len],
            [tip_b[1], tip_b[1] + out_b[1] * ray_len],
            color=ACCENT, lw=1.6, ls="--", alpha=0.85)
    ax.text(2.6, 1.8, "tip A tangent line", color=RED, fontsize=8.5, ha="left")
    ax.text(-2.6, -1.85, "tip B tangent line", color=ACCENT, fontsize=8.5, ha="right")

    # Gap midpoint (between the two inner tips)
    mid = ((tip_a[0] + tip_b[0]) / 2.0, (tip_a[1] + tip_b[1]) / 2.0)
    ax.plot(mid[0], mid[1], "o", color="purple", markersize=11, zorder=11)
    ax.annotate("gap midpoint M", xy=(mid[0], mid[1]), xytext=(mid[0] + 1.4, mid[1] - 0.3),
                fontsize=9.5, color="purple", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2))

    # Perpendicular drop from M to each tangent line (vertical, since tangents
    # are horizontal here).
    foot_a = (mid[0], 1.5)
    foot_b = (mid[0], -1.5)
    ax.plot([mid[0], foot_a[0]], [mid[1], foot_a[1]], color="purple", lw=1.6, ls=":")
    ax.plot([mid[0], foot_b[0]], [mid[1], foot_b[1]], color="purple", lw=1.6, ls=":")
    # right-angle marker at each foot
    def right_angle_marker(ax, foot, away_x, away_y, size=0.18):
        ax.plot([foot[0], foot[0] + away_x * size],
                [foot[1], foot[1] + 0], color="purple", lw=1.2)
        ax.plot([foot[0] + away_x * size, foot[0] + away_x * size],
                [foot[1], foot[1] + away_y * size], color="purple", lw=1.2)
    right_angle_marker(ax, foot_a, 1, -1)
    right_angle_marker(ax, foot_b, 1, 1)

    da = abs(mid[1] - foot_a[1])
    db = abs(mid[1] - foot_b[1])
    ax.text(mid[0] + 0.15, (mid[1] + foot_a[1]) / 2,
            f"d_A = {da*12:.0f} px", fontsize=9, color="purple", va="center")
    ax.text(mid[0] + 0.15, (mid[1] + foot_b[1]) / 2,
            f"d_B = {db*12:.0f} px", fontsize=9, color="purple", va="center")

    # tip-to-tip distance arrow
    dist_px = math.hypot(tip_a[0] - tip_b[0], tip_a[1] - tip_b[1]) * 12
    annotate_arrow(ax, tip_a, tip_b, label=f"dist = {dist_px:.0f} px",
                   color="black", lw=1.2, text_offset=(-0.5, 0.15))

    # Correct gate values for this geometry: opp ≈ 1 (tangents are anti-parallel),
    # fwd_cos ≈ 0.37 (tangents point sideways relative to the line connecting
    # the tips), residual = 18 px. fwd_cos and residual are correlated: when
    # the tangent lines miss the midpoint, the tangents also fail to point at
    # the other tip.
    avg_px = (da + db) / 2.0 * 12
    ax.text(
        0, -3.2,
        f"line residual = (d_A + d_B) / 2 = ({da*12:.0f} + {db*12:.0f}) / 2 = {avg_px:.0f} px",
        ha="center", fontsize=10, color="purple", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e9ff", edgecolor="purple"),
    )
    ax.text(0, -4.05,
            f"opp 1.0 ≥ 0.4 ✓   fwd_cos 0.37 < 0.72 ✗   residual {avg_px:.0f} > 16 ✗  →  REJECT",
            ha="center", fontsize=10.5, color=RED, fontweight="bold")
    save_fig(fig, REC_DIR / "06_stage_relaxed_reject_residual.png")


def reconnect_same_layer_relaxation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    _draw_reconnect_panel(
        axes[0],
        frag_a=[(-3.6, -0.4), (-1.2, 0.4)],
        frag_b=[(1.2, -0.4), (3.6, 0.4)],
        dist="12.0 px",
        opp_text="0.95 ✓",
        fwd_text="0.92 / 0.92 ✓",
        residual_text="7.5 px  (> 6 ✗ global)",
        verdict="REJECT under global strict gate",
        verdict_color=RED,
        title="Without same-layer relax",
        sub="High residual blocks the merge.",
    )
    _draw_reconnect_panel(
        axes[1],
        frag_a=[(-3.6, -0.4), (-1.2, 0.4)],
        frag_b=[(1.2, -0.4), (3.6, 0.4)],
        dist="12.0 px",
        opp_text="0.95 ✓",
        fwd_text="0.92 / 0.92 ✓",
        residual_text="7.5 px  (≤ same_layer 12 ✓)",
        verdict="ACCEPT under same_layer_*",
        verdict_color=GREEN,
        extra_notes="Both Components share the same orientation bin → looser residual gate applies.",
        title="With same-layer relax",
        sub="Same-bin pieces get same_layer_max_line_resid_px instead.",
    )
    fig.suptitle("Reconnect same-layer relaxation",
                 fontsize=12, fontweight="bold")
    save_fig(fig, REC_DIR / "07_same_layer_relaxation.png")


def reconnect_overlap_trim():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in axes:
        setup_axes(ax, lim=4.2)
    # left: two components overlapping
    draw_filament(axes[0], [(-3.5, 0.6), (3.5, 0.6)], color=RED, lw=14)
    draw_filament(axes[0], [(-1.5, 0.0), (3.5, 0.0)], color=ACCENT, lw=10)
    axes[0].set_title("Before: two reconnect Components overlap heavily",
                      fontsize=11)
    axes[0].text(0, -2.0, "overlap_kill / overlap_trim threshold = kill_thr",
                 ha="center", fontsize=9, color="#555", style="italic")

    # right: trim mode keeps surviving fragments
    draw_filament(axes[1], [(-3.5, 0.6), (3.5, 0.6)], color=RED, lw=14)
    draw_filament(axes[1], [(-1.5, 0.0), (-0.6, 0.0)], color=ACCENT, lw=10)
    axes[1].set_title("After mode = trim:\nlarger component intact, smaller's"
                      "\nnon-overlap fragments kept as new Components",
                      fontsize=11)
    axes[1].text(0, -2.0, "(mode = kill would drop the smaller entirely)",
                 ha="center", fontsize=9, color="#555", style="italic")
    fig.suptitle("Reconnect overlap handling: trim vs kill",
                 fontsize=12, fontweight="bold")
    save_fig(fig, REC_DIR / "08_overlap_trim.png")


def reconnect_smooth_pass():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        "iteration 1: stage_clear / strict / relaxed run",
        "iteration 2: after merges, re-evaluate neighbours",
        "iteration 3: until no new accepts",
    ]
    for ax, t in zip(axes, titles):
        setup_axes(ax, lim=4.2, title=t)

    # iter 1: 3 small fragments
    pieces = [
        [(-3.5, 1.5), (-0.8, 1.3)],
        [(-0.4, 1.2), (1.6, 1.0)],
        [(2.0, 0.95), (3.5, 0.85)],
    ]
    colors = [RED, ACCENT, GREEN]
    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            for p, c in zip(pieces, colors):
                draw_filament(ax, p, color=c, lw=11)
                for q in (p[0], p[-1]):
                    draw_tip(ax, q)
        elif ax_idx == 1:
            # merge first two
            draw_filament(ax, [(-3.5, 1.5), (1.6, 1.0)], color=RED, lw=11)
            draw_filament(ax, pieces[2], color=GREEN, lw=11)
            for q in [(-3.5, 1.5), (1.6, 1.0), pieces[2][0], pieces[2][-1]]:
                draw_tip(ax, q)
            ax.text(0, -1.0, "pieces 1+2 accepted in iter 1\n"
                    "now reconsider piece 3 as a candidate against piece 1+2",
                    ha="center", fontsize=8.5, color="#555", style="italic")
        else:
            draw_filament(ax, [(-3.5, 1.5), (3.5, 0.85)], color=RED, lw=11)
            for q in [(-3.5, 1.5), (3.5, 0.85)]:
                draw_tip(ax, q)
            ax.text(0, -1.0, "all three Components merged; converged.",
                    ha="center", fontsize=9, color=GREEN, fontweight="bold")
    fig.suptitle("Reconnect smooth passes: iterate until fixpoint",
                 fontsize=12, fontweight="bold")
    save_fig(fig, REC_DIR / "09_smooth_pass.png")


def gen_reconnect():
    print("reconnect...")
    reconnect_stage_clear_accept()
    reconnect_stage_clear_reject_dist()
    reconnect_stage_strict_accept()
    reconnect_stage_strict_reject_opposition()
    reconnect_stage_relaxed_accept()
    reconnect_stage_relaxed_reject_residual()
    reconnect_same_layer_relaxation()
    reconnect_overlap_trim()
    reconnect_smooth_pass()


# ===========================================================================
# 04. POSTPROCESS
# ===========================================================================

POST_DIR = FIG_BASE / "04_postprocess"


def postprocess_skel_and_path():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax in axes:
        setup_axes(ax, lim=4)

    # input layer mask (thick blob)
    poly = np.array([[-3.2, 0.4], [-1.0, 0.9], [1.5, 0.6], [3.0, 1.1],
                     [3.0, -0.5], [1.5, -0.7], [-1.0, -0.4], [-3.2, -0.8]])
    axes[0].add_patch(Polygon(poly, facecolor=ACCENT, alpha=0.6, edgecolor=ACCENT))
    axes[0].set_title("Reconnect multilabel layer\n(one bundle's pixels)", fontsize=11)

    # skeletonized layer
    skel_xy = np.array([[-3.0, 0.0], [-1.0, 0.2], [1.0, -0.1], [2.9, 0.3]])
    axes[1].plot(skel_xy[:, 0], skel_xy[:, 1], color=DARK, lw=2.5, marker="o", markersize=5)
    axes[1].set_title("Step 1: skeletonize the layer", fontsize=11)

    # dominant path = endpoint-to-endpoint diameter
    axes[2].plot(skel_xy[:, 0], skel_xy[:, 1], color=GREEN, lw=4)
    draw_tip(axes[2], skel_xy[0])
    draw_tip(axes[2], skel_xy[-1])
    axes[2].set_title("Step 2: dominant endpoint-to-endpoint path", fontsize=11)

    fig.suptitle("Postprocess: from layer mask to a single centerline path",
                 fontsize=12, fontweight="bold")
    save_fig(fig, POST_DIR / "01_skeletonize_and_path.png")


def postprocess_smooth_window():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax in axes:
        setup_axes(ax, lim=4)
    rng = np.random.default_rng(2)
    ts = np.linspace(0, 1, 40)
    x = -3 + 6 * ts
    y = 1.2 * np.sin(2 * ts * np.pi - 0.4) + rng.normal(0, 0.15, len(ts))
    axes[0].plot(x, y, color=DARK, lw=2.5, marker="o", markersize=4)
    axes[0].set_title("Raw dominant path (jagged from skeleton noise)", fontsize=11)

    # smooth with simple moving average
    w = 5
    sx = np.convolve(x, np.ones(w) / w, mode="valid")
    sy = np.convolve(y, np.ones(w) / w, mode="valid")
    axes[1].plot(sx, sy, color=ACCENT, lw=3.5)
    axes[1].plot(x, y, color=DARK, lw=0, marker="o", markersize=3, alpha=0.35)
    axes[1].set_title(f"After moving-window smooth\n(--smooth-window = {w})", fontsize=11)

    fig.suptitle("Postprocess step 3: smooth the centerline",
                 fontsize=12, fontweight="bold")
    save_fig(fig, POST_DIR / "02_smooth_window.png")


def postprocess_smart_width_sampling():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    setup_axes(ax, lim=4.2,
               title="Postprocess step 4: smart-width SEM edge sampling",
               subtitle="At each centerline sample, walk along the normal and find the SEM edge on each side.")

    # SEM filament: a bright band along y = 0.3 x
    rng = np.random.default_rng(8)
    ts = np.linspace(-3.0, 3.0, 80)
    band_xs = []; band_ys = []
    for t in ts:
        for off in np.linspace(-0.55, 0.55, 7):
            band_xs.append(t)
            band_ys.append(0.3 * t + off + rng.normal(0, 0.06))
    ax.scatter(band_xs, band_ys, s=14, color="#cccccc")

    # centerline (smoothed path)
    cx = np.linspace(-3.0, 3.0, 11)
    cy = 0.3 * cx
    ax.plot(cx, cy, color=ACCENT, lw=2.5)

    # normals + sampled edges at 3 sample points
    samples = [-2.0, 0.0, 2.0]
    for sx in samples:
        sy = 0.3 * sx
        # normal direction = (-sin(theta), cos(theta)) for theta = atan(0.3)
        theta = math.atan(0.3)
        nx = -math.sin(theta); ny = math.cos(theta)
        L = 1.0
        ax.plot([sx - L * nx, sx + L * nx], [sy - L * ny, sy + L * ny],
                color=ORANGE, lw=1.5)
        # edge dots on the band edge (just at +/- 0.55 along normal)
        for sign in (-1, 1):
            px = sx + sign * 0.55 * nx
            py = sy + sign * 0.55 * ny
            ax.plot(px, py, "o", color=RED, markersize=8, markeredgecolor="black")
    ax.text(0, -2.5,
            "  ▸ orange = normal probe\n"
            "  ▸ red dots = opposite-slope SEM edge pair on either side of the bundle",
            ha="center", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e0", edgecolor=ORANGE))
    save_fig(fig, POST_DIR / "03_smart_width_sampling.png")


def postprocess_smart_width_median():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: per-sample widths (scatter with outliers)
    rng = np.random.default_rng(33)
    n = 30
    widths = 6.0 + rng.normal(0, 0.6, n)
    widths[5] = 12.0  # outlier
    widths[12] = 1.5  # outlier
    axes[0].scatter(range(n), widths, color=ACCENT, s=30, edgecolor="black")
    axes[0].set_xlabel("sample index along the path")
    axes[0].set_ylabel("measured width (px)")
    axes[0].set_title("Per-sample widths (SEM edge pair separations)", fontsize=11)
    median_w = np.median(widths)
    axes[0].axhline(median_w, color=GREEN, lw=2.5, label=f"median = {median_w:.1f}px")
    axes[0].axhline(median_w * 1.5, color="#aaa", lw=1, ls="--", label="trim ± 50%")
    axes[0].axhline(median_w * 0.5, color="#aaa", lw=1, ls="--")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # right: chosen ID width applied to render
    ax = axes[1]
    setup_axes(ax, lim=4)
    cx = np.linspace(-3, 3, 20)
    cy = 0.2 * cx
    ax.plot(cx, cy, color=DARK, lw=2.5)
    # band rendered at median_w
    band_h = median_w / 8  # scale for visual
    poly_top = list(zip(cx, cy + band_h))
    poly_bot = list(zip(cx[::-1], cy[::-1] - band_h))
    ax.add_patch(Polygon(poly_top + poly_bot, facecolor=ACCENT, alpha=0.4,
                          edgecolor=ACCENT))
    ax.set_title(f"Bundle rendered at the chosen width\n(={median_w:.1f}px from SEM evidence)",
                 fontsize=11)
    ax.text(0, -3.6, "fallback: if SEM evidence is weak, use --thicken-px",
            ha="center", fontsize=9, color="#555", style="italic")

    fig.suptitle("Postprocess step 5: median width per ID, with outlier trim",
                 fontsize=12, fontweight="bold")
    save_fig(fig, POST_DIR / "04_smart_width_median.png")


def postprocess_overlap_absorb():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        setup_axes(ax, lim=4)

    # left: two near-duplicate bundles
    band1_top = np.array([[-3.0, 0.6], [3.0, 0.6]])
    band1_bot = np.array([[3.0, -0.6], [-3.0, -0.6]])
    band2_top = np.array([[-2.5, 0.45], [2.5, 0.45]])
    band2_bot = np.array([[2.5, -0.45], [-2.5, -0.45]])
    axes[0].add_patch(Polygon(np.vstack([band1_top, band1_bot]),
                              facecolor=ACCENT, alpha=0.55))
    axes[0].add_patch(Polygon(np.vstack([band2_top, band2_bot]),
                              facecolor=RED, alpha=0.55))
    axes[0].set_title("Two rendered layers overlap heavily\n(near-duplicate bundles)",
                      fontsize=11)
    axes[0].text(0, -2.4,
                 "intersection ≥ overlap_absorb_thr × smaller_area\n→ trigger absorb",
                 ha="center", fontsize=9, color="#555",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e0", edgecolor=ORANGE))

    # right: merged
    axes[1].add_patch(Polygon(np.vstack([band1_top, band1_bot]),
                              facecolor=ACCENT, alpha=0.7))
    axes[1].set_title("After absorb: smaller layer's mask folded into the larger,\n"
                      "smaller ID removed", fontsize=11)

    fig.suptitle("Postprocess: overlap absorb",
                 fontsize=12, fontweight="bold")
    save_fig(fig, POST_DIR / "05_overlap_absorb.png")


def postprocess_occlusion_trim():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        setup_axes(ax, lim=4)
    # left: a long high-priority layer + a short layer half-covered by it
    long_top = np.array([[-3.5, 0.5], [3.5, 0.5]])
    long_bot = np.array([[3.5, -0.5], [-3.5, -0.5]])
    axes[0].add_patch(Polygon(np.vstack([long_top, long_bot]),
                              facecolor=ACCENT, alpha=0.5,
                              label="layer 1 (longer)"))
    short_top = np.array([[-1.5, 1.5], [2.5, 0.55]])
    short_bot = np.array([[2.5, -0.55], [-1.5, 0.6]])
    axes[0].add_patch(Polygon(np.vstack([short_top, short_bot]),
                              facecolor=RED, alpha=0.6,
                              label="layer 2 (shorter)"))
    axes[0].set_title("Lower-priority layer 2 is mostly\nalready covered by layer 1", fontsize=11)
    axes[0].text(0, -2.7, "hidden fraction ≥ occlusion_trim_thr\n→ trim hidden pixels",
                 ha="center", fontsize=9, color="#555",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e0", edgecolor=ORANGE))

    # right: occluded part trimmed; only the visible tip survives
    axes[1].add_patch(Polygon(np.vstack([long_top, long_bot]),
                              facecolor=ACCENT, alpha=0.5))
    surv_top = np.array([[-1.5, 1.5], [0.0, 0.5]])
    surv_bot = np.array([[0.0, 0.5], [-1.5, 0.6]])
    # the leftover sliver
    axes[1].add_patch(Polygon(np.vstack([surv_top, surv_bot]),
                              facecolor=RED, alpha=0.65))
    axes[1].set_title("After trim: hidden pixels removed,\n"
                      "only the visible tip of layer 2 remains", fontsize=11)
    axes[1].text(0, -2.7, "if surviving area < occlusion_trim_min_px, ID is dropped entirely",
                 ha="center", fontsize=9, color="#555", style="italic")

    fig.suptitle("Postprocess: occlusion trim", fontsize=12, fontweight="bold")
    save_fig(fig, POST_DIR / "06_occlusion_trim.png")


def gen_postprocess():
    print("postprocess...")
    postprocess_skel_and_path()
    postprocess_smooth_window()
    postprocess_smart_width_sampling()
    postprocess_smart_width_median()
    postprocess_overlap_absorb()
    postprocess_occlusion_trim()


# ===========================================================================

def main():
    gen_stringart()
    gen_preprocess()
    gen_reconnect()
    gen_postprocess()


if __name__ == "__main__":
    main()
