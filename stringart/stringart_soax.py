"""
SOAX-inspired Filament Tracker  (stringart_soax.py)
Stretching Open Active Contours for dense filament network instance segmentation.

Fixes vs v1:
  - Snakes rendered as filled polylines (cv2.polylines), not sparse control points
  - Coverage: junction regions now also seeded; short leftover branches included
  - SNAKE_BETA reduced to avoid rectangular/over-rigid artifacts
  - RIDGE_GAMMA increased so snakes actually follow the field
  - Orientation layers rasterised via polylines too (consistent with rendering)

Usage:
    python stringart_soax.py
    python stringart_soax.py input.png out/
    python stringart_soax.py input_folder/ out/

pip install opencv-python numpy matplotlib scikit-image scipy pandas
"""

import argparse, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import (convolve, label, gaussian_filter,
                           map_coordinates, distance_transform_edt)
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH    = r"C:\Repos\filaments_quantification\2_stringart\input\case00000_0000"
OUTPUT_FOLDER = r"C:\Repos\filaments_quantification\2_stringart\output\case00000_0000SOAX"

# --- Preprocessing ---
ASSUME_BINARY    = True
THRESH_MODE      = "auto"    # "auto" | "fixed" | "otsu" | "adaptive"
FIXED_THRESHOLD  = 127
INVERT           = False

# --- Ridge potential ---
# RIDGE_MODE "dist" uses distance transform (best for clean binary masks — peaks at skeleton,
#   giving inward-pointing gradients that correctly pull snakes to filament centres).
# RIDGE_MODE "gauss" uses Gaussian blur (better for noisy grayscale fluorescence images).
RIDGE_MODE       = "dist"
RIDGE_SIGMA      = 1.5       # only used when RIDGE_MODE="gauss"
RIDGE_GAMMA      = 8.0       # strength of image force. Increase if snakes don't move enough.

# --- Snake internal energy ---
SNAKE_ALPHA      = 0.05      # elasticity  (resist unequal spacing)
SNAKE_BETA       = 0.005     # rigidity    (resist bending). Very low: let image force dominate.
SNAKE_DT         = 0.5       # time step
MAX_ITERS        = 300
CONV_TOL         = 0.02

# --- Initialization ---
RESAMPLE_SPACING = 4         # inter-point distance (px)
MIN_SNAKE_PTS    = 4
PRUNE_SPURS      = True
PRUNE_ITERS      = 4         # was 6 — over-pruning removed valid short branches
MIN_BRANCH_PX    = 6

# --- Tip stretching ---
STRETCH          = True
STRETCH_STEP     = 2.0
STRETCH_MAX      = 50
STRETCH_THRESH   = 0.03      # was 0.05 — too high for binary input

# --- Junction attachment ---
SNAP_ENABLED     = True
SNAP_DIST        = 10.0

# --- Rendering ---
DRAW_THICKNESS   = 2         # polyline thickness when rendering snake map

# --- Orientation export ---
NUM_BINS         = 12

# --- Output ---
SAVE_DEBUG       = True
SAVE_SUMMARY_FIG = True
SAVE_CSV         = True
SHOW_PLOTS       = False
FIG_DPI          = 250
RNG_SEED         = 0
RECURSIVE        = True
SUPPORTED_EXT    = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_DEG_K = np.ones((3, 3), np.uint8); _DEG_K[1, 1] = 0
_MATRIX_CACHE: Dict[int, np.ndarray] = {}


# ============================================================
# DATA
# ============================================================

@dataclass
class Snake:
    sid: int
    pts: np.ndarray  # float32, shape (N, 2), columns = (y, x)


# ============================================================
# UTILITIES
# ============================================================

def is_binary_like(img):
    s = img[::2, ::2] if img.size > 2_000_000 else img
    if np.unique(s).size <= 4: return True
    return (np.mean(s <= 10) + np.mean(s >= 245)) > 0.98

def _pts_to_polyline(pts):
    """(N,2) (y,x) float -> OpenCV polyline format (N,1,2) int (x,y)."""
    return np.round(pts[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)


# ============================================================
# PREPROCESSING
# ============================================================

def load(filepath):
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot load: {filepath}")
    raw = img.copy()
    if ASSUME_BINARY or (THRESH_MODE == "auto" and is_binary_like(img)):
        binary = (img > 0).astype(np.uint8) * 255
        return raw, (255 - binary if INVERT else binary)
    mode = THRESH_MODE.lower()
    if mode == "fixed":
        _, binary = cv2.threshold(img, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY)
    elif mode in ("otsu", "auto"):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 5)
    if mode == "auto" and np.mean(binary > 0) > 0.5:
        binary = 255 - binary
    if INVERT: binary = 255 - binary
    return raw, binary

def make_ridge(binary_u8):
    """
    Ridge potential: filament centres -> 1.0, background -> 0.0.

    dist mode  (recommended for binary masks):
        Distance transform of the binary mask peaks at the medial axis,
        giving strong inward-pointing gradients that pull snakes toward
        filament centres. This is geometrically correct for binary input.

    gauss mode (for noisy grayscale input):
        Gaussian blur of the intensity image approximates the ridge but
        produces a flat plateau at filament centres where gradient -> 0.
    """
    if RIDGE_MODE == "dist":
        fg = (binary_u8 > 0).astype(np.uint8)
        dt = distance_transform_edt(fg).astype(np.float32)
        mx = dt.max()
        ridge = (dt / mx) if mx > 0 else dt
        # light Gaussian smoothing to avoid staircase gradient artifacts
        return gaussian_filter(ridge, sigma=0.8)
    else:
        r = gaussian_filter(binary_u8.astype(np.float32) / 255.0, sigma=RIDGE_SIGMA)
        mx = r.max()
        return (r / mx) if mx > 0 else r


# ============================================================
# SNAKE MATRIX  (semi-implicit open active contour)
# ============================================================

def _build_inv(N: int) -> np.ndarray:
    """Cached (I + dt*A)^-1 — computed once per unique snake length N."""
    if N in _MATRIX_CACHE: return _MATRIX_CACHE[N]
    D1 = np.zeros((max(0, N-1), N))
    for i in range(N-1): D1[i, i] = -1.0; D1[i, i+1] = 1.0
    D2 = np.zeros((max(0, N-2), N))
    for i in range(N-2): D2[i, i] = 1.0; D2[i, i+1] = -2.0; D2[i, i+2] = 1.0
    Ae = D1.T @ D1 if N >= 2 else np.zeros((N, N))
    Ab = D2.T @ D2 if N >= 3 else np.zeros((N, N))
    M = np.eye(N) + SNAKE_DT * (SNAKE_ALPHA * Ae + SNAKE_BETA * Ab)
    _MATRIX_CACHE[N] = np.linalg.inv(M)
    return _MATRIX_CACHE[N]


# ============================================================
# SNAKE OPERATIONS
# ============================================================

def _resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    if len(pts) < 2: return pts
    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = arc[-1]
    if total < 1e-6: return pts
    n = max(2, int(round(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, arc, pts[:, 0]),
                     np.interp(t, arc, pts[:, 1])], axis=1).astype(np.float32)

def _sample(pts, field, h, w):
    return map_coordinates(field,
                           [np.clip(pts[:, 0], 0, h-1), np.clip(pts[:, 1], 0, w-1)],
                           order=1, mode='nearest')

def evolve_snake(snake: Snake, ridge, grad_y, grad_x, h, w) -> Snake:
    pts = _resample(snake.pts, RESAMPLE_SPACING).astype(np.float64)
    if len(pts) < MIN_SNAKE_PTS:
        return Snake(sid=snake.sid, pts=pts.astype(np.float32))

    M_inv = _build_inv(len(pts))

    for _ in range(MAX_ITERS):
        fy = _sample(pts, grad_y, h, w) * RIDGE_GAMMA
        fx = _sample(pts, grad_x, h, w) * RIDGE_GAMMA
        new_pts = np.stack([
            np.clip(M_inv @ (pts[:, 0] + SNAKE_DT * fy), 0, h-1),
            np.clip(M_inv @ (pts[:, 1] + SNAKE_DT * fx), 0, w-1),
        ], axis=1)
        disp = np.mean(np.hypot(new_pts[:, 0]-pts[:, 0], new_pts[:, 1]-pts[:, 1]))
        pts = new_pts
        if disp < CONV_TOL: break

    if STRETCH:
        pts = _stretch_tips(pts, ridge, h, w)

    return Snake(sid=snake.sid, pts=pts.astype(np.float32))

def _stretch_tips(pts, ridge, h, w):
    for side in (0, -1):
        for _ in range(STRETCH_MAX):
            if len(pts) < 2: break
            tang = (pts[0] - pts[1]) if side == 0 else (pts[-1] - pts[-2])
            n = math.hypot(tang[0], tang[1])
            if n < 1e-6: break
            new_pt = np.clip(pts[side] + tang / n * STRETCH_STEP,
                             [0, 0], [h-1, w-1]).reshape(1, 2)
            if _sample(new_pt, ridge, h, w)[0] < STRETCH_THRESH: break
            pts = np.vstack([new_pt, pts]) if side == 0 else np.vstack([pts, new_pt])
    return pts


# ============================================================
# INITIALIZATION FROM SKELETON
# ============================================================

def _prune_spurs(sk, iters):
    sk = sk.copy().astype(np.uint8)
    for _ in range(iters):
        deg = convolve(sk, _DEG_K, mode="constant", cval=0)
        ends = (sk == 1) & (deg == 1)
        if not ends.any(): break
        sk[ends] = 0
    return sk

def _walk_branch(ys, xs):
    coords = set(zip(ys.tolist(), xs.tolist()))
    start = next(
        ((y, x) for y, x in coords if sum(
            1 for dy in [-1,0,1] for dx in [-1,0,1]
            if (dy,dx) != (0,0) and (y+dy,x+dx) in coords) <= 1),
        (int(ys[0]), int(xs[0]))
    )
    ordered, visited, cur = [start], {start}, start
    while True:
        y, x = cur
        nxt = next(
            ((y+dy, x+dx) for dy in [-1,0,1] for dx in [-1,0,1]
             if (dy,dx) != (0,0) and (y+dy,x+dx) in coords
             and (y+dy,x+dx) not in visited), None)
        if nxt is None: break
        visited.add(nxt); ordered.append(nxt); cur = nxt
    return ordered if len(ordered) >= 2 else None

def initialize_snakes(binary_u8: np.ndarray) -> List[Snake]:
    sk = skeletonize(binary_u8 > 0).astype(np.uint8)
    if PRUNE_SPURS:
        sk = _prune_spurs(sk, PRUNE_ITERS)

    deg = convolve(sk, _DEG_K, mode="constant", cval=0)
    junc_mask = (sk == 1) & (deg > 2)
    junc_dil = cv2.dilate(junc_mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    branches = (sk == 1) & (~junc_dil)

    labeled, n = label(branches.astype(np.uint8), structure=np.ones((3,3), np.uint8))

    snakes, sid = [], 1
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < MIN_BRANCH_PX: continue
        ordered = _walk_branch(ys, xs)
        if not ordered or len(ordered) < MIN_SNAKE_PTS: continue
        pts = _resample(np.array(ordered, np.float32), RESAMPLE_SPACING)
        if len(pts) >= MIN_SNAKE_PTS:
            snakes.append(Snake(sid=sid, pts=pts)); sid += 1

    # Also seed junction regions so they aren't left uncovered
    junc_labeled, nj = label(junc_mask.astype(np.uint8), structure=np.ones((3,3), np.uint8))
    for i in range(1, nj + 1):
        ys, xs = np.where(junc_labeled == i)
        if len(ys) < 2: continue
        ordered = _walk_branch(ys, xs)
        if not ordered or len(ordered) < 2: continue
        pts = _resample(np.array(ordered, np.float32), RESAMPLE_SPACING)
        if len(pts) >= 2:
            snakes.append(Snake(sid=sid, pts=pts)); sid += 1

    return snakes


# ============================================================
# JUNCTION ATTACHMENT  (KD-tree tip snapping)
# ============================================================

def junction_attachment(snakes: List[Snake]) -> List[Snake]:
    if not SNAP_ENABLED or len(snakes) < 2: return snakes
    all_pts = np.vstack([s.pts for s in snakes]).astype(np.float32)
    owner = np.concatenate([np.full(len(s.pts), si) for si, s in enumerate(snakes)])
    tree = cKDTree(all_pts)
    for si, snake in enumerate(snakes):
        for tip_idx in (0, -1):
            dists, idxs = tree.query(snake.pts[tip_idx], k=20)
            for d, idx in zip(dists.flat, idxs.flat):
                if d > SNAP_DIST: break
                if owner[idx] == si: continue
                snakes[si].pts[tip_idx] = all_pts[idx]
                break
    return snakes


# ============================================================
# RENDERING  — polylines, not sparse points
# ============================================================

def render_snake_map(h, w, snakes, n_colors, binary_u8=None):
    """Draw each snake as a filled polyline. Returns uint8 RGB.
    If binary_u8 supplied, uncovered foreground pixels are filled via
    nearest-snake propagation so coverage matches the input mask."""
    rng = np.random.default_rng(RNG_SEED)
    palette = rng.integers(60, 255, (n_colors + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for s in snakes:
        if len(s.pts) < 2: continue
        color = palette[s.sid % n_colors + 1].tolist()
        cv2.polylines(canvas, [_pts_to_polyline(s.pts)], isClosed=False,
                      color=color, thickness=DRAW_THICKNESS, lineType=cv2.LINE_AA)

    # Fallback: fill uncovered foreground pixels by propagating nearest drawn pixel
    if binary_u8 is not None:
        covered = np.any(canvas > 0, axis=2)
        fg = binary_u8 > 0
        gap = fg & ~covered
        if gap.any() and covered.any():
            _, inds = distance_transform_edt(~covered, return_indices=True)
            for c in range(3):
                canvas[:, :, c] = np.where(gap, canvas[inds[0], inds[1], c], canvas[:, :, c])

    return canvas


# ============================================================
# ORIENTATION LAYERS  — rasterised as polylines
# ============================================================

def snake_orientation(snake: Snake) -> Optional[float]:
    if len(snake.pts) < 2: return None
    y0, x0 = snake.pts[0]; y1, x1 = snake.pts[-1]
    return float(np.degrees(np.arctan2(y1-y0, x1-x0)) % 180.0)

def export_orientation_layers(mask_u8, snakes, out_dir, base):
    h, w = mask_u8.shape
    bin_step = 180.0 / NUM_BINS
    label_canvas = np.zeros((h, w), np.int32)

    for s in snakes:
        ang = snake_orientation(s)
        if ang is None or len(s.pts) < 2: continue
        lab = int(ang // bin_step) % NUM_BINS + 1
        tmp = np.zeros((h, w), np.uint8)
        cv2.polylines(tmp, [_pts_to_polyline(s.pts)], isClosed=False,
                      color=255, thickness=DRAW_THICKNESS, lineType=cv2.LINE_AA)
        label_canvas[tmp > 0] = lab

    # Fill unlabelled foreground via nearest-seed propagation
    if (mask_u8 > 0).any() and label_canvas.any():
        _, inds = distance_transform_edt((label_canvas == 0).astype(np.uint8),
                                          return_indices=True)
        propagated = label_canvas[inds[0], inds[1]]
        label_canvas = np.where(mask_u8 > 0, propagated, 0).astype(np.int32)

    layer_dir = Path(out_dir) / "orientation_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_BINS):
        cv2.imwrite(str(layer_dir / f"{base}_branch_{i+1}.png"),
                    ((label_canvas == i+1).astype(np.uint8) * 255))

    rng = np.random.default_rng(RNG_SEED)
    colors = np.vstack([[[0,0,0]],
                        rng.integers(32, 255, (NUM_BINS, 3), dtype=np.uint8)]).astype(np.uint8)
    viz = colors[np.clip(label_canvas, 0, NUM_BINS)].astype(np.uint8)
    cv2.imwrite(str(layer_dir / f"{base}_orientation_rgb.png"),
                cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


# ============================================================
# STATS & SUMMARY
# ============================================================

def _snake_length(s: Snake) -> float:
    p = s.pts
    return float(sum(math.hypot(p[i,0]-p[i-1,0], p[i,1]-p[i-1,1]) for i in range(1, len(p))))

def save_outputs(raw, binary, ridge, snakes, out_dir, base):
    h, w = binary.shape
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    n = max((s.sid for s in snakes), default=1)
    rgb = render_snake_map(h, w, snakes, n, binary_u8=binary)
    cv2.imwrite(str(out_dir / "color_coded_snakes.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if SAVE_SUMMARY_FIG:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (im, title, cmap) in zip(axes, [
            (raw,                            "Raw",         "gray"),
            (binary,                         "Binary",      "gray"),
            ((ridge * 255).astype(np.uint8), "Ridge Field", "hot"),
            (rgb,                            "SOAX Snakes", None),
        ]):
            ax.imshow(im, cmap=cmap); ax.set_title(title); ax.axis("off")
        plt.tight_layout()
        fig.savefig(str(out_dir / "summary_plot.png"), dpi=FIG_DPI, bbox_inches="tight")
        if SHOW_PLOTS: plt.show()
        else: plt.close(fig)

    if SAVE_CSV:
        pd.DataFrame([{"sid": s.sid, "n_pts": len(s.pts),
                        "length_px": _snake_length(s),
                        "orientation_deg": snake_orientation(s)} for s in snakes]
                     ).to_csv(str(out_dir / f"{base}_snakes.csv"), index=False)


# ============================================================
# PIPELINE
# ============================================================

def process(filepath, output_root):
    filepath = Path(filepath)
    base, out_dir = filepath.stem, Path(output_root) / filepath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {filepath.name}")

    raw, binary = load(filepath)
    h, w = binary.shape

    if SAVE_DEBUG:
        dbg = out_dir / "debug"; dbg.mkdir(exist_ok=True)
        cv2.imwrite(str(dbg / "raw.png"), raw)
        cv2.imwrite(str(dbg / "binary.png"), binary)

    ridge = make_ridge(binary)
    grad_y, grad_x = np.gradient(ridge.astype(np.float64))

    if SAVE_DEBUG:
        cv2.imwrite(str(dbg / "ridge.png"), (ridge * 255).astype(np.uint8))

    print("  Initializing snakes from skeleton...")
    snakes = initialize_snakes(binary)
    print(f"  Initialized {len(snakes)} snakes.")

    print(f"  Evolving (max {MAX_ITERS} iters/snake)...")
    every = max(1, len(snakes) // 10)
    evolved = []
    for i, s in enumerate(snakes):
        if (i + 1) % every == 0:
            print(f"    {i+1}/{len(snakes)}")
        e = evolve_snake(s, ridge, grad_y, grad_x, h, w)
        if len(e.pts) >= MIN_SNAKE_PTS:
            evolved.append(e)

    for i, s in enumerate(evolved): s.sid = i + 1

    if SNAP_ENABLED:
        print("  Junction attachment (tip snapping)...")
        evolved = junction_attachment(evolved)

    print(f"  Final: {len(evolved)} snakes  ->  {out_dir}")
    save_outputs(raw, binary, ridge, evolved, out_dir, base)
    export_orientation_layers(binary, evolved, out_dir, base)


def find_inputs(path):
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED_EXT else []
    return sorted(f for f in (p.rglob if RECURSIVE else p.glob)("*")
                  if f.suffix.lower() in SUPPORTED_EXT)

def main():
    parser = argparse.ArgumentParser(description="SOAX-inspired filament tracker")
    parser.add_argument("input",  nargs="?", default=INPUT_PATH)
    parser.add_argument("output", nargs="?", default=OUTPUT_FOLDER)
    args = parser.parse_args()
    files = find_inputs(args.input)
    if not files: raise RuntimeError(f"No images found at: {args.input}")
    print(f"Found {len(files)} image(s).")
    failures = []
    for fp in files:
        try: process(fp, args.output)
        except Exception as e:
            failures.append((fp, e)); print(f"  ERROR [{Path(fp).name}]: {e}")
    print(f"\nDone. {len(files)-len(failures)}/{len(files)} succeeded.")
    for fp, e in failures: print(f"  FAILED: {fp}\n    {e}")

if __name__ == "__main__":
    main()