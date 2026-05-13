# stringart_tiles.py
"""
Tile-wise greedy line reconstruction ("string art"-style) for binary filament masks.

Core idea per tile:
  - optionally skeletonize the mask
  - track the remaining residual pixels
  - process candidate lines one angle bin at a time
  - accept candidates that explain enough new residual pixels
  - subtract accepted lines and write per-branch outputs

Outputs: original.png, reconstructed.png, overlay.png,
         branches/<base>_branch_i.png + merge, run_config.json

Additional features:
  - automatic width-based scaling for Hough and acceptance thresholds
  - runtime parameter bundles for auto-tuning
  - optional local thickness search around candidate lines
  - OpenCV thinning when available, with a Zhang-Suen fallback
"""

from __future__ import annotations
import argparse
import os, json, math, time
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# CONFIG  –  adjust here
# ============================================================
CONFIG: Dict[str, Any] = {
    # --- IO ---
    "INPUT_PATH":         r"C:\Repos\filaments_quantification\1.stringart\input\cnt_orient_0003_merge.png",
    "OUTPUT_ROOT":        r"C:\Repos\filaments_quantification\1.stringart\output\cnt_orient_0003_merge",
    "OUTPUT_FOLDER_NAME": None,   # None → timestamp; or set a string like "my_run"
    "BIN_THRESHOLD":      127,

    # --- Tiling / binning ---
    "TILE_SIZE":       128,   # tile width/height in px; smaller = more local fits, more seams
    "ANGLE_STEP_DEG":  15,    # angle-bin width over [0°,180°); smaller = more bins
    # Multi-grid voting: None/1 = single grid; integer N = generated offsets;
    # or set an explicit list of [oy, ox] tile-grid origins (in px). The image is
    # processed once per offset; per-orientation branches are aggregated via
    # majority voting (a pixel is kept if it appears in >= TILE_GRID_VOTE_MIN of
    # the grids). This makes the result almost insensitive to where the tile grid
    # lies and rejects single-grid noise. Cost is ~Nx the stringart stage where
    # N = len(offsets). For TILE_SIZE=128, 4 -> [[0,0],[64,0],[0,64],[64,64]].
    "TILE_GRID_OFFSETS":  None,
    "TILE_GRID_VOTE_MIN": 2,    # majority: pixel kept if in >= this many grids (rejects single-grid noise)

    # --- Preprocessing ---
    "USE_SKELETONIZE":    False,  # thin to ~1 px centerlines before Hough
    "THINNING_MAX_ITERS": 3,      # max Zhang-Suen iterations (ignored when ximgproc is used)

    # --- Optional intersection removal (blob-like junctions) ---
    "REMOVE_INTERSECTIONS":      False,
    "INTERSECTION_KERNEL":       3,   # opening kernel size (odd); larger = more aggressive
    "INTERSECTION_ITERS":        1,   # opening iterations
    "INTERSECTION_BLOB_EXPAND_PX": 1, # extra dilation radius around blobs before subtract

    # --- Greedy acceptance ---
    "MAX_LINES_PER_TILE":  300,
    "MIN_ACCEPT_NEWPIX":   6,     # baseline at width=2 px (autoscaled if AUTO_SCALE_PARAMS)
    "DRAW_THICKNESS":      1,     # MANUAL – never autoscaled

    # --- HoughLinesP ---
    "HOUGH_RHO":           1.0,
    "HOUGH_THETA_RAD":     float(np.pi / 180),
    "HOUGH_THRESHOLD":     18,    # baseline at width=2 px (autoscaled if AUTO_SCALE_PARAMS)
    "HOUGH_MIN_LINE_LENGTH": 4,   # baseline at width=2 px (autoscaled if AUTO_SCALE_PARAMS)
    "HOUGH_MAX_LINE_GAP":  5,     # baseline at width=2 px (autoscaled if AUTO_SCALE_PARAMS)

    # --- Candidate evaluation ---
    "MAX_CANDIDATES_TO_TRY": 300,   # try only top-N longest candidates per iteration
    "MIN_HARD_LINE_LENGTH":  3,     # hard floor for HOUGH_MIN_LINE_LENGTH after scaling/tuning
    "MIN_ACCEPT_DENSITY":    0.45,  # fraction of candidate line that must lie on the original mask

    # --- Residual dilation before Hough (kept manual) ---
    "RESIDUAL_DILATE_KERNEL": 2,   # kernel size; <3 = disabled
    "RESIDUAL_DILATE_ITERS":  1,   # dilation iterations

    # --- Output control ---
    "SAVE_BRANCHES": True,   # save per-angle-bin images + merge
    "SAVE_OVERLAY":  True,   # save alpha-blended overlay

    # --- Overlay appearance ---
    "OVERLAY_ALPHA":              0.55,
    "OVERLAY_RECON_COLOR_BGR":    [0, 255, 0],
    "OVERLAY_REMOVED_COLOR_BGR":  [0, 0, 255],

    # --- Optional auto-tune (tunes only the 4 autoscaled params) ---
    "AUTO_TUNE":       False,
    "TUNE_NUM_TILES":  12,     # non-empty tiles to evaluate
    "TUNE_MAX_COMBOS": 180,    # hard cap on tested combos
    "TUNE_SEED":       0,
    "TUNE_W_COVERAGE": 1.0,    # weight: thick coverage of wanted pixels
    "TUNE_W_NOISEFRAC": 1.2,   # weight: penalise noise/thin
    "TUNE_W_PREC":     0.2,    # weight: reward thin precision

    # --- Width estimation + autoscaling (only the 4 Hough/accept params) ---
    "AUTO_SCALE_PARAMS":      True,
    "AUTO_SCALE_SAMPLE_MAX":  200000,
    "AUTO_SCALE_PERCENTILE":  60,    # percentile of DT radii for width estimate
    "AUTO_SCALE_MIN_WIDTH":   1.0,
    "AUTO_SCALE_MAX_WIDTH":   12.0,
    "AUTO_SCALE_REF_WIDTH_PX": 2.0,  # anchor: at this width use CONFIG baseline values

    # Per-parameter multipliers (1.0 = no bias)
    "AUTO_SCALE_HOUGH_THRESHOLD_MULT":      1.0,
    "AUTO_SCALE_HOUGH_MINLEN_MULT":         1.0,
    "AUTO_SCALE_HOUGH_MAXGAP_MULT":         1.0,
    "AUTO_SCALE_MIN_ACCEPT_NEWPIX_MULT":    1.0,

    # Try small DRAW_THICKNESS neighbourhood for best pixel overlap
    # (DRAW_THICKNESS itself remains the manual base, not tuned)
    "AUTO_MULTI_THICKNESS_TRY":   True,
    "AUTO_MULTI_THICKNESS_DELTA": 1,

    # --- Experiment comparison grid ---
    # Set to a list of (label, config_override_dict) to compare multiple runs.
    # When non-empty, main() prints a quality table before the normal run.
    # Example:
    #   "EXPERIMENT_GRID": [
    #       ("baseline",     {}),
    #       ("gap-10",       {"HOUGH_MAX_LINE_GAP": 10}),
    #       ("gap-15",       {"HOUGH_MAX_LINE_GAP": 15}),
    #       ("no-skel",      {"USE_SKELETONIZE": False}),
    #       ("thin-1px",     {"DRAW_THICKNESS": 1}),
    #   ],
    "EXPERIMENT_GRID": [],
}


# ============================================================
# CLI overrides, while keeping CONFIG as the user-editable default
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tile-wise stringart vectorization for binary filament masks.")
    ap.add_argument("--input", type=str, default=None, help="Override CONFIG['INPUT_PATH'].")
    ap.add_argument("--output-root", type=str, default=None, help="Override CONFIG['OUTPUT_ROOT'].")
    ap.add_argument("--output-folder-name", type=str, default=None, help="Named run folder under output root.")
    ap.add_argument("--bin-threshold", type=int, default=None)
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--angle-step-deg", type=int, default=None)
    ap.add_argument("--draw-thickness", type=int, default=None)
    ap.add_argument("--hough-threshold", type=int, default=None)
    ap.add_argument("--hough-min-line-length", type=int, default=None)
    ap.add_argument("--hough-max-line-gap", type=int, default=None)
    ap.add_argument("--min-accept-newpix", type=int, default=None)
    ap.add_argument("--min-accept-density", type=float, default=None)
    ap.add_argument("--residual-dilate-kernel", type=int, default=None)
    ap.add_argument("--residual-dilate-iters", type=int, default=None)
    ap.add_argument("--use-skeletonize", action="store_true", default=None)
    ap.add_argument("--no-skeletonize", action="store_false", dest="use_skeletonize")
    ap.add_argument("--auto-scale", action="store_true", default=None)
    ap.add_argument("--no-auto-scale", action="store_false", dest="auto_scale")
    ap.add_argument("--tile-grid-offsets", type=str, default=None,
                    help='Tile-grid offset count (1,2,3,4,...) or JSON list of [oy,ox] origins, '
                         'e.g. "4" or "[[0,0],[64,0],[0,64],[64,64]]". Omit for single-grid.')
    ap.add_argument("--tile-grid-vote-min", type=int, default=None,
                    help="Min number of grids a pixel must appear in to be kept. "
                         "1 = OR (any grid). 2+ = strict (rejects single-grid noise).")
    return ap.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "input": "INPUT_PATH",
        "output_root": "OUTPUT_ROOT",
        "output_folder_name": "OUTPUT_FOLDER_NAME",
        "bin_threshold": "BIN_THRESHOLD",
        "tile_size": "TILE_SIZE",
        "angle_step_deg": "ANGLE_STEP_DEG",
        "draw_thickness": "DRAW_THICKNESS",
        "hough_threshold": "HOUGH_THRESHOLD",
        "hough_min_line_length": "HOUGH_MIN_LINE_LENGTH",
        "hough_max_line_gap": "HOUGH_MAX_LINE_GAP",
        "min_accept_newpix": "MIN_ACCEPT_NEWPIX",
        "min_accept_density": "MIN_ACCEPT_DENSITY",
        "residual_dilate_kernel": "RESIDUAL_DILATE_KERNEL",
        "residual_dilate_iters": "RESIDUAL_DILATE_ITERS",
        "use_skeletonize": "USE_SKELETONIZE",
        "auto_scale": "AUTO_SCALE_PARAMS",
    }
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            CONFIG[cfg_name] = val
    if getattr(args, "tile_grid_offsets", None):
        CONFIG["TILE_GRID_OFFSETS"] = parse_tile_grid_offsets(args.tile_grid_offsets)
    if getattr(args, "tile_grid_vote_min", None) is not None:
        CONFIG["TILE_GRID_VOTE_MIN"] = int(args.tile_grid_vote_min)


def apply_tile_size_scaling() -> None:
    s2 = (max(1, int(CONFIG["TILE_SIZE"])) / 128.0) ** 2
    for k in ("MAX_LINES_PER_TILE", "MAX_CANDIDATES_TO_TRY"):
        CONFIG[k] = max(1, int(round(CONFIG[k] * s2)))


def parse_tile_grid_offsets(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return None if s.lower() in ("", "none", "null") else json.loads(s)
    return value


def resolve_tile_grid_offsets(spec: Any, tile_size: int) -> List[List[int]]:
    spec = parse_tile_grid_offsets(spec)
    if spec is None:
        return [[0, 0]]
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        n, half = max(1, int(spec)), max(1, int(tile_size) // 2)
        if n <= 1: return [[0, 0]]
        if n == 2: return [[0, 0], [half, half]]
        if n == 3: return [[0, 0], [half, 0], [0, half]]
        if n == 4: return [[0, 0], [half, 0], [0, half], [half, half]]
        return [[int(((i * 0.61803398875) % 1.0) * tile_size),
                 int(((i * 0.41421356237) % 1.0) * tile_size)] for i in range(n)]
    return spec


# ============================================================
# Small utilities
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def binarize(img: np.ndarray, thr: int) -> np.ndarray:
    """Threshold to 0/255."""
    return cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)[1]

def bool255(img: np.ndarray) -> np.ndarray:
    """Ensure strictly 0/255 (collapses any non-zero to 255)."""
    return (img > 0).astype(np.uint8) * 255

def rp_get(name: str, rp: Optional[Dict[str, Any]], cfg_key: Optional[str] = None) -> Any:
    """Read from runtime_params if present, else fall back to CONFIG."""
    if rp is not None and name in rp:
        return rp[name]
    return CONFIG[cfg_key or name]


# ============================================================
# Preprocessing helpers
# ============================================================

def remove_intersection_blobs(
    bin255: np.ndarray, kernel_sz: int, iters: int, expand_px: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect junction blobs via morphological opening and subtract them."""
    k = int(kernel_sz) | 1  # force odd
    if k < 1:
        return bin255.copy(), np.zeros_like(bin255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blobs = bool255(cv2.morphologyEx(bin255, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iters))))
    if expand_px > 0:
        ek = 2 * int(expand_px) + 1
        blobs = bool255(cv2.dilate(blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))))
    cleaned = bool255(cv2.bitwise_and(bin255, cv2.bitwise_not(blobs)))
    return cleaned, blobs


def _zhang_suen_py(bin255: np.ndarray, max_iters: int) -> np.ndarray:
    """Pure-Python Zhang-Suen thinning fallback (slow on large images)."""
    img = np.pad((bin255 > 0).astype(np.uint8), 1, constant_values=0)

    def nb8(x: int, y: int) -> List[int]:
        return [img[x-1,y], img[x-1,y+1], img[x,y+1], img[x+1,y+1],
                img[x+1,y], img[x+1,y-1], img[x,y-1], img[x-1,y-1]]

    def trans(n: List[int]) -> int:
        return sum((n[i] == 0 and n[(i+1) % 8] == 1) for i in range(8))

    for _ in range(int(max_iters)):
        changed = False
        for step, c1, c2 in ((0, (1,3,5), (3,5,7)), (1, (1,3,7), (1,5,7))):
            to_del = []
            for x, y in zip(*np.where(img == 1)):
                if x == 0 or y == 0 or x >= img.shape[0]-1 or y >= img.shape[1]-1:
                    continue
                n = nb8(x, y)
                B = sum(n)
                if not (2 <= B <= 6) or trans(n) != 1:
                    continue
                if any(n[i] == 1 for i in c1) and any(n[i] == 1 for i in c2):
                    continue
                to_del.append((x, y))
            if to_del:
                for x, y in to_del:
                    img[x, y] = 0
                changed = True
        if not changed:
            break
    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


def skeletonize(bin255: np.ndarray, max_iters: int) -> np.ndarray:
    """
    Thin binary mask to ~1 px centerlines.
    Uses cv2.ximgproc.thinning (fast C++) when available;
    falls back to pure-Python Zhang-Suen otherwise.
    """
    try:
        # ximgproc ignores max_iters but is orders of magnitude faster
        return cv2.ximgproc.thinning(bool255(bin255), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        return _zhang_suen_py(bin255, max_iters)


# ============================================================
# Width estimation + restricted autoscaling (only 4 params)
# ============================================================

def estimate_filament_width_px(
    bin255: np.ndarray,
    sample_max: int = 200_000,
    pct: float = 60.0,
) -> Dict[str, float]:
    """
    Estimate characteristic filament width via distance transform.
    width ≈ 2 × percentile(DT radii on foreground pixels).
    """
    fg = (bin255 > 0).astype(np.uint8)
    if fg.sum() == 0:
        return {"width_px": 1.0, "radius_px": 0.5, "n_samples": 0}
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    vals = dt[fg > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return {"width_px": 1.0, "radius_px": 0.5, "n_samples": 0}
    if vals.size > sample_max:
        idx = np.random.default_rng(0).choice(vals.size, size=sample_max, replace=False)
        vals = vals[idx]
    rad = float(np.percentile(vals, pct))
    return {"width_px": max(1.0, 2.0 * rad), "radius_px": rad, "n_samples": int(vals.size)}


def autoscale_4_params(cfg: Dict[str, Any], est_width_px: float) -> Dict[str, Any]:
    """
    Scale exactly 4 params anchored at AUTO_SCALE_REF_WIDTH_PX (default 2 px).
    Scaling laws are conservative sub-linear power functions:
      threshold  ~ s^0.60   (mild growth — thicker lines need slightly more votes)
      min_length ~ s^0.70   (suppress tiny junk on thick masks)
      max_gap    ~ s^0.85   (allow more bridging for thicker structures)
      min_newpix ~ s^0.95   (overlap area grows nearly linearly with width)
    DRAW_THICKNESS and dilation params are passed through unchanged.
    """
    w = float(np.clip(est_width_px,
                      cfg.get("AUTO_SCALE_MIN_WIDTH", 1.0),
                      cfg.get("AUTO_SCALE_MAX_WIDTH", 12.0)))
    s = w / max(0.5, float(cfg.get("AUTO_SCALE_REF_WIDTH_PX", 2.0)))
    hard_min = int(cfg.get("MIN_HARD_LINE_LENGTH", 3))

    def scale(base_key: str, exp: float, mult_key: str, floor: int = 1) -> int:
        return int(round(max(floor, cfg[base_key] * (s ** exp) * float(cfg.get(mult_key, 1.0)))))

    return {
        # autoscaled
        "HOUGH_THRESHOLD":      scale("HOUGH_THRESHOLD",     0.60, "AUTO_SCALE_HOUGH_THRESHOLD_MULT"),
        "HOUGH_MIN_LINE_LENGTH": scale("HOUGH_MIN_LINE_LENGTH", 0.70, "AUTO_SCALE_HOUGH_MINLEN_MULT", floor=hard_min),
        "HOUGH_MAX_LINE_GAP":   scale("HOUGH_MAX_LINE_GAP",   0.85, "AUTO_SCALE_HOUGH_MAXGAP_MULT",  floor=0),
        "MIN_ACCEPT_NEWPIX":    scale("MIN_ACCEPT_NEWPIX",    0.95, "AUTO_SCALE_MIN_ACCEPT_NEWPIX_MULT"),
        # manual passthroughs
        "DRAW_THICKNESS":          int(cfg["DRAW_THICKNESS"]),
        "MIN_ACCEPT_DENSITY":      float(cfg.get("MIN_ACCEPT_DENSITY", 0.0)),
        "RESIDUAL_DILATE_KERNEL":  int(cfg.get("RESIDUAL_DILATE_KERNEL", 0)),
        "RESIDUAL_DILATE_ITERS":   int(cfg.get("RESIDUAL_DILATE_ITERS", 1)),
        # metadata
        "_width_scale_factor":     float(s),
        "_est_width_px_clamped":   float(w),
        "_autoscaled_only_4_params": True,
    }


# ============================================================
# Greedy decomposition helpers
# ============================================================

def seg_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

def angle_bins(step_deg: int) -> List[Tuple[float, float]]:
    step = float(max(1, int(step_deg)))
    bins, a = [], 0.0
    while a < 180.0:
        bins.append((a, min(180.0, a + step)))
        a += step
    return bins


def draw_line_mask(shape_hw: Tuple[int, int], x1: int, y1: int, x2: int, y2: int, thickness: int) -> np.ndarray:
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.line(m, (x1, y1), (x2, y2), 255, thickness=max(1, int(thickness)), lineType=cv2.LINE_8)
    return m

def thickness_try_list(base_th: int, cfg: Dict[str, Any]) -> List[int]:
    """Build list of thickness values to try; base first, then neighbours."""
    base_th = max(1, int(base_th))
    if not cfg.get("AUTO_MULTI_THICKNESS_TRY", False):
        return [base_th]
    d = max(0, int(cfg.get("AUTO_MULTI_THICKNESS_DELTA", 1)))
    variants = sorted({max(1, base_th + dd) for dd in range(-d, d + 1)})
    return [base_th] + [v for v in variants if v != base_th]


# ============================================================
# Core greedy tile decomposition
# ============================================================

def greedy_decompose_tile(
    tile_bin255: np.ndarray,
    runtime_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], dict]:
    """
    Greedy string-art decomposition of a single binary tile.
    Returns (recon, recon_thin, branches_per_bin, stats).
    recon_thin is a thickness=1 reconstruction used only for metrics.
    """
    tile_bin255 = bool255(tile_bin255)
    target = (skeletonize(tile_bin255, int(CONFIG["THINNING_MAX_ITERS"]))
              if CONFIG["USE_SKELETONIZE"] else tile_bin255.copy())
    residual = target.copy()

    bins    = angle_bins(CONFIG["ANGLE_STEP_DEG"])
    n_bins  = len(bins)
    branches = [np.zeros_like(tile_bin255) for _ in bins]
    recon    = np.zeros_like(tile_bin255)
    recon_thin = np.zeros_like(tile_bin255)  # thickness-1 mask for stats only

    # --- resolve runtime params (autoscaled or CONFIG fallback) ---
    def rp(name: str) -> Any:
        return rp_get(name, runtime_params)

    # optional residual dilation before Hough (manual, not autoscaled)
    dil_kernel = None
    k = int(rp("RESIDUAL_DILATE_KERNEL"))
    if k >= 3:
        k = k + (k % 2 == 0)  # force odd
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil_iters = max(0, int(rp("RESIDUAL_DILATE_ITERS")))

    # Hough / acceptance params (may be autoscaled)
    h_rho    = float(CONFIG["HOUGH_RHO"])
    h_theta  = float(CONFIG["HOUGH_THETA_RAD"])
    h_thr    = int(rp("HOUGH_THRESHOLD"))
    h_minlen = int(rp("HOUGH_MIN_LINE_LENGTH"))
    h_maxgap = int(rp("HOUGH_MAX_LINE_GAP"))
    min_newpix   = int(rp("MIN_ACCEPT_NEWPIX"))
    min_density  = float(rp("MIN_ACCEPT_DENSITY"))
    base_thickness = int(rp("DRAW_THICKNESS"))
    max_lines    = int(CONFIG["MAX_LINES_PER_TILE"])
    max_try      = int(CONFIG["MAX_CANDIDATES_TO_TRY"])
    th_try       = thickness_try_list(base_thickness, CONFIG)

    accepted_total   = 0
    accepted_per_bin = [0] * n_bins
    thickness_hist: Dict[int, int] = {}

    for bi, (a0, a1) in enumerate(bins):
        while accepted_total < max_lines:
            if cv2.countNonZero(residual) == 0:
                break

            # optionally dilate residual to bridge tiny holes before Hough
            edges = (cv2.dilate(residual, dil_kernel, iterations=dil_iters)
                     if (dil_kernel is not None and dil_iters > 0) else residual)
            lines = cv2.HoughLinesP(edges, h_rho, h_theta, h_thr,
                                    minLineLength=h_minlen, maxLineGap=h_maxgap)
            if lines is None or len(lines) == 0:
                break

            # filter to current angle bin, sort longest first
            cand = []
            for x1, y1, x2, y2 in lines[:, 0, :].astype(int):
                ang = seg_angle_deg(x1, y1, x2, y2)
                in_bin = (a0 <= ang < a1) if bi < n_bins - 1 else (a0 <= ang <= a1)
                if not in_bin:
                    continue
                L = math.hypot(x2 - x1, y2 - y1)
                if L < h_minlen:
                    continue
                cand.append((L, (x1, y1, x2, y2)))
            if not cand:
                break
            cand.sort(reverse=True)

            accepted = False
            for _, (x1, y1, x2, y2) in cand[:max_try]:
                # Accept only lines supported by the original mask; this blocks
                # long chords that touch a few pixels but mostly cross black BG.
                best_lm, best_newpix, best_th = None, -1, None
                for th in th_try:
                    lm = draw_line_mask(residual.shape, x1, y1, x2, y2, th)
                    line_px = max(1, cv2.countNonZero(lm))
                    new_px = int(cv2.countNonZero(cv2.bitwise_and(residual, lm)))
                    support_px = int(cv2.countNonZero(cv2.bitwise_and(target, lm)))
                    density = support_px / float(line_px)
                    if new_px >= min_newpix and density >= min_density and new_px > best_newpix:
                        best_newpix, best_lm, best_th = new_px, lm, th

                if best_lm is None:
                    continue  # not enough new pixels — try next candidate

                # accept: subtract from residual, add to reconstruction
                residual   = cv2.bitwise_and(residual, cv2.bitwise_not(best_lm))
                recon      = cv2.bitwise_or(recon,     best_lm)
                branches[bi] = cv2.bitwise_or(branches[bi], best_lm)
                recon_thin = cv2.bitwise_or(recon_thin,
                                            draw_line_mask(residual.shape, x1, y1, x2, y2, 1))
                accepted_total += 1
                accepted_per_bin[bi] += 1
                thickness_hist[best_th] = thickness_hist.get(best_th, 0) + 1
                accepted = True
                break

            if not accepted:
                break  # no candidate survived — advance to next bin

    stats = {
        "accepted_total":        int(accepted_total),
        "accepted_per_bin":      [int(v) for v in accepted_per_bin],
        "residual_pixels_left":  int(cv2.countNonZero(residual)),
        "initial_target_pixels": int(cv2.countNonZero(target)),
        "angle_bins":            bins,
        "used_skeletonize":      bool(CONFIG["USE_SKELETONIZE"]),
        "used_thickness_hist":   {str(k): int(v) for k, v in sorted(thickness_hist.items())},
        "min_accept_density":    float(min_density),
        "runtime_params":        runtime_params or {},
    }
    return recon, recon_thin, branches, stats


# ============================================================
# Tiling + stitching
# ============================================================

def iter_tiles(h: int, w: int, tile: int):
    """Yield (y0, y1, x0, x1) for each tile."""
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            yield y0, min(h, y0 + tile), x0, min(w, x0 + tile)

def process_image_tiled(
    img_bin255: np.ndarray,
    runtime_params: Optional[Dict[str, Any]] = None,
):
    h, w = img_bin255.shape
    tile_sz = int(CONFIG["TILE_SIZE"])
    bins    = angle_bins(CONFIG["ANGLE_STEP_DEG"])

    recon_full      = np.zeros((h, w), dtype=np.uint8)
    recon_thin_full = np.zeros((h, w), dtype=np.uint8)
    branches_full   = [np.zeros((h, w), dtype=np.uint8) for _ in bins]
    tile_stats: List[dict] = []

    for y0, y1, x0, x1 in iter_tiles(h, w, tile_sz):
        tile = img_bin255[y0:y1, x0:x1]
        if cv2.countNonZero(tile) == 0:
            continue
        recon_t, recon_thin_t, branches_t, stats = greedy_decompose_tile(tile, runtime_params)
        recon_full[y0:y1, x0:x1]      = cv2.bitwise_or(recon_full[y0:y1, x0:x1],      recon_t)
        recon_thin_full[y0:y1, x0:x1] = cv2.bitwise_or(recon_thin_full[y0:y1, x0:x1], recon_thin_t)
        for i in range(len(bins)):
            branches_full[i][y0:y1, x0:x1] = cv2.bitwise_or(
                branches_full[i][y0:y1, x0:x1], branches_t[i])
        stats["tile_bbox"] = [int(y0), int(y1), int(x0), int(x1)]
        tile_stats.append(stats)

    return recon_full, recon_thin_full, branches_full, tile_stats, bins


def process_image_multigrid(
    img_bin255: np.ndarray,
    runtime_params: Optional[Dict[str, Any]] = None,
):
    """Run process_image_tiled at each offset in TILE_GRID_OFFSETS and OR the
    per-orientation branches. Padding shifts the tile grid relative to the image,
    so features near a boundary in one grid land in the interior of another.
    Returns the same 5-tuple as process_image_tiled; tile_stats are concatenated.
    """
    # Sanity: clamp offsets to [0, TILE_SIZE) and de-dup
    tsz = int(CONFIG["TILE_SIZE"])
    offsets = resolve_tile_grid_offsets(CONFIG.get("TILE_GRID_OFFSETS"), tsz)
    seen = set()
    norm: List[Tuple[int, int]] = []
    for oy, ox in offsets:
        key = (int(oy) % tsz, int(ox) % tsz)
        if key not in seen:
            seen.add(key)
            norm.append(key)
    if len(norm) == 1 and norm[0] == (0, 0):
        return process_image_tiled(img_bin255, runtime_params)

    vote_min = max(1, int(CONFIG.get("TILE_GRID_VOTE_MIN", 1)))
    vote_min = min(vote_min, len(norm))   # cap at number of grids

    h, w = img_bin255.shape
    # Vote counters (uint8 large enough for typical N <= 16 grids)
    recon_votes  = np.zeros((h, w), dtype=np.uint8)
    thin_votes   = np.zeros((h, w), dtype=np.uint8)
    branch_votes: Optional[List[np.ndarray]] = None
    all_stats: List[dict] = []
    bins_out = None

    for oy, ox in norm:
        if oy == 0 and ox == 0:
            shifted = img_bin255
        else:
            shifted = cv2.copyMakeBorder(img_bin255, oy, 0, ox, 0, cv2.BORDER_CONSTANT, value=0)
        recon, thin, branches, stats_list, bins_out = process_image_tiled(shifted, runtime_params)
        if oy or ox:
            recon    = recon[oy:oy + h, ox:ox + w]
            thin     = thin[oy:oy + h, ox:ox + w]
            branches = [b[oy:oy + h, ox:ox + w] for b in branches]
        recon_votes += (recon > 0).astype(np.uint8)
        thin_votes  += (thin  > 0).astype(np.uint8)
        if branch_votes is None:
            branch_votes = [(b > 0).astype(np.uint8) for b in branches]
        else:
            for i, b in enumerate(branches):
                branch_votes[i] += (b > 0).astype(np.uint8)
        for s in stats_list:
            s = dict(s); s["grid_offset_yx"] = [int(oy), int(ox)]
            all_stats.append(s)

    accum_recon = ((recon_votes >= vote_min).astype(np.uint8)) * 255
    accum_thin  = ((thin_votes  >= vote_min).astype(np.uint8)) * 255
    accum_branches = [((bv >= vote_min).astype(np.uint8)) * 255 for bv in (branch_votes or [])]

    print(f"[multigrid] tile_size={tsz}  offsets={norm}  ({len(norm)} grids, vote_min={vote_min})")
    return accum_recon, accum_thin, accum_branches, all_stats, bins_out


# ============================================================
# Optional overlay visualisation
# ============================================================

def overlay_masks_on_gray(
    gray: np.ndarray,
    recon255: np.ndarray,
    removed255: Optional[np.ndarray],
    alpha: float,
    recon_bgr: Tuple[int, int, int],
    removed_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Alpha-blend coloured masks onto a grayscale image."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    a   = float(np.clip(alpha, 0.0, 1.0))

    def blend(mask: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
        if mask is None or not np.any(mask > 0):
            return
        m   = mask > 0
        col = np.array(color, dtype=np.float32)
        out[m] = np.clip((1.0 - a) * out[m].astype(np.float32) + a * col, 0, 255).astype(np.uint8)

    blend(recon255, recon_bgr)
    blend(removed255, removed_bgr)   # drawn last so it's visible over recon
    return out


# ============================================================
# Optional auto-tune  (tunes only the 4 autoscaled params)
# ============================================================

def _eval_tile(
    tile_in: np.ndarray,
    tile_wanted: np.ndarray,
    rp: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float]:
    """Return (coverage_thick, noise_frac_thin, precision_thin) for one tile."""
    recon_t, recon_thin_t, _, _ = greedy_decompose_tile(tile_in, rp)
    wanted_px = max(1, int(cv2.countNonZero(tile_wanted)))
    thin_px   = max(1, int(cv2.countNonZero(recon_thin_t)))
    hit_thick = int(cv2.countNonZero(cv2.bitwise_and(recon_t,      tile_wanted)))
    hit_thin  = int(cv2.countNonZero(cv2.bitwise_and(recon_thin_t, tile_wanted)))
    noise_thin = int(cv2.countNonZero(cv2.bitwise_and(recon_thin_t, cv2.bitwise_not(tile_wanted))))
    return (hit_thick / wanted_px,
            noise_thin / thin_px,
            hit_thin   / thin_px)


def auto_tune_params(
    img_bin_for_algo: np.ndarray,
    img_wanted: np.ndarray,
    runtime_seed_params: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Grid-search tune only: HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH,
    HOUGH_MAX_LINE_GAP, MIN_ACCEPT_NEWPIX.
    DRAW_THICKNESS and dilation params remain fixed.
    Seeded from runtime_seed_params (e.g. autoscaled values) when provided.
    """
    t     = int(CONFIG["TILE_SIZE"])
    coords = [(y0, y1, x0, x1)
               for y0, y1, x0, x1 in iter_tiles(img_bin_for_algo.shape[0], img_bin_for_algo.shape[1], t)
               if cv2.countNonZero(img_bin_for_algo[y0:y1, x0:x1]) > 0]
    if not coords:
        return {}

    rng = np.random.default_rng(int(CONFIG.get("TUNE_SEED", 0)))
    rng.shuffle(coords)
    coords = coords[:max(1, int(CONFIG.get("TUNE_NUM_TILES", 12)))]

    seed     = runtime_seed_params or {}
    hard_min = int(CONFIG.get("MIN_HARD_LINE_LENGTH", 3))

    # Base values (seed from autoscaled if provided, else CONFIG)
    base = {k: int(seed.get(k, CONFIG[k])) for k in
            ("HOUGH_THRESHOLD", "HOUGH_MIN_LINE_LENGTH", "HOUGH_MAX_LINE_GAP", "MIN_ACCEPT_NEWPIX")}
    fixed = {
        "DRAW_THICKNESS":         int(seed.get("DRAW_THICKNESS",         CONFIG["DRAW_THICKNESS"])),
        "MIN_ACCEPT_DENSITY":     float(seed.get("MIN_ACCEPT_DENSITY",   CONFIG.get("MIN_ACCEPT_DENSITY", 0.0))),
        "RESIDUAL_DILATE_KERNEL": int(seed.get("RESIDUAL_DILATE_KERNEL", CONFIG.get("RESIDUAL_DILATE_KERNEL", 0))),
        "RESIDUAL_DILATE_ITERS":  int(seed.get("RESIDUAL_DILATE_ITERS",  CONFIG.get("RESIDUAL_DILATE_ITERS", 1))),
    }

    # Small neighbourhood search around base values; ±3 for threshold, ±2 for others
    def nbhd(key: str, deltas: range, floor: int = 1) -> List[int]:
        return sorted({max(floor, base[key] + d) for d in deltas})

    combos = [
        (thr, ml, gp, npix)
        for thr  in nbhd("HOUGH_THRESHOLD",     range(-3, 4))
        for ml   in nbhd("HOUGH_MIN_LINE_LENGTH", range(-2, 3), floor=hard_min)
        for gp   in nbhd("HOUGH_MAX_LINE_GAP",   range(-2, 3), floor=0)
        for npix in nbhd("MIN_ACCEPT_NEWPIX",    range(-2, 3))
    ]
    rng.shuffle(combos)
    combos = combos[:max(1, int(CONFIG.get("TUNE_MAX_COMBOS", 180)))]

    w_cov, w_nf, w_prec = (float(CONFIG.get(k, d)) for k, d in
                            (("TUNE_W_COVERAGE", 1.0), ("TUNE_W_NOISEFRAC", 1.2), ("TUNE_W_PREC", 0.2)))

    best_score, best = -1e18, None
    for thr, ml, gp, npix in combos:
        rp = {**fixed, **seed,
              "HOUGH_THRESHOLD": thr, "HOUGH_MIN_LINE_LENGTH": ml,
              "HOUGH_MAX_LINE_GAP": gp, "MIN_ACCEPT_NEWPIX": npix}

        results = [_eval_tile(img_bin_for_algo[y0:y1, x0:x1], img_wanted[y0:y1, x0:x1], rp)
                   for y0, y1, x0, x1 in coords]
        cov_m, nf_m, pr_m = (float(np.mean([r[i] for r in results])) for i in range(3))
        score = w_cov * cov_m - w_nf * nf_m + w_prec * pr_m

        if score > best_score:
            best_score = score
            best = {**rp,
                    "_tune_score":      float(score),
                    "_tune_cov_thick":  float(cov_m),
                    "_tune_noise_frac": float(nf_m),
                    "_tune_prec_thin":  float(pr_m)}
    return best or {}


# ============================================================
# Run quality metrics + experiment grid
# ============================================================

def compute_run_quality(
    original_bin255: np.ndarray,
    recon255: np.ndarray,
    recon_thin255: np.ndarray,
) -> dict:
    """Compute precision/recall/F1/noise metrics for one reconstruction."""
    wanted_px  = max(1, int(cv2.countNonZero(original_bin255)))
    thin_px    = max(1, int(cv2.countNonZero(recon_thin255)))
    hit_thick  = int(cv2.countNonZero(cv2.bitwise_and(recon255,      original_bin255)))
    hit_thin   = int(cv2.countNonZero(cv2.bitwise_and(recon_thin255, original_bin255)))
    noise_thin = int(cv2.countNonZero(cv2.bitwise_and(recon_thin255, cv2.bitwise_not(original_bin255))))
    recall    = hit_thick / wanted_px
    precision = hit_thin  / thin_px
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    noise_frac = noise_thin / thin_px
    return {
        "recall":     round(recall,    4),
        "precision":  round(precision, 4),
        "f1":         round(f1,        4),
        "noise_frac": round(noise_frac,4),
        "wanted_px":  wanted_px,
        "hit_thick":  hit_thick,
        "hit_thin":   hit_thin,
        "noise_thin": noise_thin,
        "thin_px":    thin_px,
    }


def run_experiment_set(
    experiments: List[Tuple[str, Dict[str, Any]]],
    img_bin_orig: np.ndarray,
) -> None:
    """
    Run the full pipeline for each (label, config_overrides) pair and print
    a side-by-side quality table so you can compare parameter choices.

    Metrics:
      Recall     — fraction of original pixels covered by thick reconstruction
      Precision  — fraction of thin-recon pixels that land on original mask
      F1         — harmonic mean of the two
      Noise%     — thin-recon pixels outside original mask (lower is better)
    """
    if not experiments:
        return

    print("\n" + "=" * 76)
    print("  EXPERIMENT COMPARISON GRID")
    print("=" * 76)
    print(f"  {'Label':<22} {'Recall':>8} {'Prec':>8} {'F1':>8} {'Noise%':>8} {'ThinPx':>8}")
    print("-" * 76)

    rows = []
    for label, overrides in experiments:
        saved = {k: CONFIG[k] for k in overrides if k in CONFIG}
        CONFIG.update(overrides)
        try:
            img_algo = img_bin_orig.copy()
            blobs_exp = None
            if CONFIG.get("REMOVE_INTERSECTIONS", False):
                img_algo, blobs_exp = remove_intersection_blobs(
                    img_algo,
                    kernel_sz=int(CONFIG.get("INTERSECTION_KERNEL", 3)),
                    iters=int(CONFIG.get("INTERSECTION_ITERS", 1)),
                    expand_px=int(CONFIG.get("INTERSECTION_BLOB_EXPAND_PX", 0)),
                )
            rp: Dict[str, Any] = {
                "DRAW_THICKNESS":         int(CONFIG["DRAW_THICKNESS"]),
                "MIN_ACCEPT_DENSITY":     float(CONFIG.get("MIN_ACCEPT_DENSITY", 0.0)),
                "RESIDUAL_DILATE_KERNEL": int(CONFIG.get("RESIDUAL_DILATE_KERNEL", 0)),
                "RESIDUAL_DILATE_ITERS":  int(CONFIG.get("RESIDUAL_DILATE_ITERS",  1)),
                "HOUGH_THRESHOLD":        int(CONFIG["HOUGH_THRESHOLD"]),
                "HOUGH_MIN_LINE_LENGTH":  int(CONFIG["HOUGH_MIN_LINE_LENGTH"]),
                "HOUGH_MAX_LINE_GAP":     int(CONFIG["HOUGH_MAX_LINE_GAP"]),
                "MIN_ACCEPT_NEWPIX":      int(CONFIG["MIN_ACCEPT_NEWPIX"]),
            }
            if CONFIG.get("AUTO_SCALE_PARAMS", False):
                wi = estimate_filament_width_px(
                    img_algo,
                    sample_max=int(CONFIG.get("AUTO_SCALE_SAMPLE_MAX", 200_000)),
                    pct=float(CONFIG.get("AUTO_SCALE_PERCENTILE", 60)),
                )
                rp.update(autoscale_4_params(CONFIG, wi["width_px"]))
            recon, recon_thin, _, _, _ = process_image_tiled(img_algo, rp)
            if blobs_exp is not None:
                nb = cv2.bitwise_not(blobs_exp)
                recon      = cv2.bitwise_and(recon,      nb)
                recon_thin = cv2.bitwise_and(recon_thin, nb)
            m = compute_run_quality(img_bin_orig, recon, recon_thin)
            rows.append((label, m))
            print(
                f"  {label:<22} {m['recall']:>8.4f} {m['precision']:>8.4f} {m['f1']:>8.4f} "
                f"{m['noise_frac']*100:>7.2f}% {m['thin_px']:>8d}"
            )
        finally:
            CONFIG.update(saved)

    print("=" * 76)
    if rows:
        best = max(rows, key=lambda r: r[1]["f1"])
        print(f"  Best F1 → {best[0]}  ({best[1]['f1']:.4f})")
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    apply_cli_overrides(parse_args())
    apply_tile_size_scaling()
    img          = read_gray(CONFIG["INPUT_PATH"])
    img_bin_orig = bool255(binarize(img, CONFIG["BIN_THRESHOLD"]))
    img_bin      = img_bin_orig.copy()

    blobs = None
    if CONFIG.get("REMOVE_INTERSECTIONS", False):
        img_bin, blobs = remove_intersection_blobs(
            img_bin,
            kernel_sz  = int(CONFIG.get("INTERSECTION_KERNEL", 3)),
            iters      = int(CONFIG.get("INTERSECTION_ITERS", 1)),
            expand_px  = int(CONFIG.get("INTERSECTION_BLOB_EXPAND_PX", 0)),
        )

    # Initialise runtime_params from CONFIG; will be overwritten by autoscale/tune below
    runtime_params: Dict[str, Any] = {
        "DRAW_THICKNESS":          int(CONFIG["DRAW_THICKNESS"]),
        "MIN_ACCEPT_DENSITY":      float(CONFIG.get("MIN_ACCEPT_DENSITY", 0.0)),
        "RESIDUAL_DILATE_KERNEL":  int(CONFIG.get("RESIDUAL_DILATE_KERNEL", 0)),
        "RESIDUAL_DILATE_ITERS":   int(CONFIG.get("RESIDUAL_DILATE_ITERS", 1)),
        "HOUGH_THRESHOLD":         int(CONFIG["HOUGH_THRESHOLD"]),
        "HOUGH_MIN_LINE_LENGTH":   int(CONFIG["HOUGH_MIN_LINE_LENGTH"]),
        "HOUGH_MAX_LINE_GAP":      int(CONFIG["HOUGH_MAX_LINE_GAP"]),
        "MIN_ACCEPT_NEWPIX":       int(CONFIG["MIN_ACCEPT_NEWPIX"]),
    }

    # Step 1: autoscale 4 params from estimated filament width
    width_info = None
    if CONFIG.get("AUTO_SCALE_PARAMS", False):
        width_info = estimate_filament_width_px(
            img_bin,
            sample_max = int(CONFIG.get("AUTO_SCALE_SAMPLE_MAX", 200_000)),
            pct        = float(CONFIG.get("AUTO_SCALE_PERCENTILE", 60)),
        )
        runtime_params.update(autoscale_4_params(CONFIG, width_info["width_px"]))
        wp, rp_, s = width_info["width_px"], width_info["radius_px"], width_info["n_samples"]
        print(f"[AUTO_SCALE] width~{wp:.3f}px  radius~{rp_:.3f}px  samples={s}")
        print(f"[AUTO_SCALE] autoscaled -> thr={runtime_params['HOUGH_THRESHOLD']}  "
              f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
              f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
              f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}")
        print(f"[AUTO_SCALE] manual     -> thickness={runtime_params['DRAW_THICKNESS']}  "
              f"min_density={runtime_params['MIN_ACCEPT_DENSITY']:.2f}  "
              f"dil_k={runtime_params['RESIDUAL_DILATE_KERNEL']}  "
              f"dil_i={runtime_params['RESIDUAL_DILATE_ITERS']}")

    # Step 2: optionally fine-tune the 4 autoscaled params on a sample of tiles
    if CONFIG.get("AUTO_TUNE", False):
        best = auto_tune_params(img_bin, img_bin_orig, runtime_seed_params=runtime_params)
        if best:
            runtime_params = {k: v for k, v in best.items() if not str(k).startswith("_")}
            print(f"[TUNE] -> thr={runtime_params['HOUGH_THRESHOLD']}  "
                  f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
                  f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
                  f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}  "
                  f"thickness(fixed)={runtime_params['DRAW_THICKNESS']}")
            print(f"[TUNE]   score={best['_tune_score']:.6g}  "
                  f"cov={best['_tune_cov_thick']:.6g}  "
                  f"noise={best['_tune_noise_frac']:.6g}  "
                  f"prec={best['_tune_prec_thin']:.6g}")

    # --- optional experiment comparison grid ---
    if CONFIG.get("EXPERIMENT_GRID"):
        run_experiment_set(CONFIG["EXPERIMENT_GRID"], img_bin_orig)

    # --- output folder ---
    folder_name = (CONFIG.get("OUTPUT_FOLDER_NAME") or "").strip()
    base_name   = folder_name if folder_name else ts()
    run_dir     = os.path.join(CONFIG["OUTPUT_ROOT"], base_name)
    if os.path.exists(run_dir):
        run_dir = f"{run_dir}_{ts()}"
    branches_dir = os.path.join(run_dir, "branches")
    ensure_dir(branches_dir)

    # --- run ---
    recon, recon_thin, branches, tile_stats, bins = process_image_multigrid(img_bin, runtime_params)

    # Mask out removed blobs from recon/branches (if intersection removal was used)
    if blobs is not None:
        nb  = cv2.bitwise_not(blobs)
        branches  = [bool255(cv2.bitwise_and(b, nb)) for b in branches]
        recon     = np.zeros_like(recon)
        for b in branches:
            recon = cv2.bitwise_or(recon, b)
        recon_thin = bool255(cv2.bitwise_and(recon_thin, nb))

    # --- save outputs ---
    cv2.imwrite(os.path.join(run_dir, "original.png"),      img_bin_orig)
    cv2.imwrite(os.path.join(run_dir, "reconstructed.png"), recon)

    if CONFIG.get("SAVE_OVERLAY", True):
        overlay = overlay_masks_on_gray(
            gray        = img_bin_orig,
            recon255    = recon,
            removed255  = blobs,
            alpha       = float(CONFIG.get("OVERLAY_ALPHA", 0.55)),
            recon_bgr   = tuple(int(v) for v in CONFIG.get("OVERLAY_RECON_COLOR_BGR",   [0, 255, 0])),
            removed_bgr = tuple(int(v) for v in CONFIG.get("OVERLAY_REMOVED_COLOR_BGR", [0, 0, 255])),
        )
        cv2.imwrite(os.path.join(run_dir, "overlay.png"), overlay)

    base_stem = os.path.splitext(os.path.basename(CONFIG["INPUT_PATH"]))[0]
    if CONFIG.get("SAVE_BRANCHES", True):
        for i, b in enumerate(branches):
            cv2.imwrite(os.path.join(branches_dir, f"{base_stem}_branch_{i+1}.png"), b)
        branches_merge = np.zeros_like(recon)
        for b in branches:
            branches_merge = cv2.bitwise_or(branches_merge, b)
        cv2.imwrite(os.path.join(branches_dir, f"{base_stem}_branches_merge.png"), branches_merge)
        # Width hint for reconnect pipeline — written alongside branches so
        # reconnect_run.py can auto-scale dilate_px to match filament scale.
        with open(os.path.join(branches_dir, "filament_width.json"), "w", encoding="utf-8") as _fw:
            wi = width_info or {"width_px": 1.0, "radius_px": 0.5}
            json.dump({"filament_width_px": wi.get("width_px", 1.0),
                       "radius_px":         wi.get("radius_px", 0.5)}, _fw, indent=2)

    # --- metrics ---
    wanted    = img_bin_orig
    wanted_px = max(1, int(cv2.countNonZero(wanted)))
    thin_px   = max(1, int(cv2.countNonZero(recon_thin)))
    hit_thick = int(cv2.countNonZero(cv2.bitwise_and(recon,      wanted)))
    hit_thin  = int(cv2.countNonZero(cv2.bitwise_and(recon_thin, wanted)))
    noise_thin = int(cv2.countNonZero(cv2.bitwise_and(recon_thin, cv2.bitwise_not(wanted))))
    cov_thick = hit_thick / wanted_px
    prec_thin = hit_thin  / thin_px

    # --- JSON config dump ---
    cfg_out = {
        "CONFIG":                CONFIG,
        "run_dir":               run_dir,
        "angle_bins":            bins,
        "auto_scale_width_info": width_info,
        "runtime_params_final":  runtime_params,
        "summary": {
            "wanted_input_pixels":       wanted_px,
            "recon_thin_pixels":         thin_px,
            "recon_thin_hit_pixels":     hit_thin,
            "recon_thin_noise_pixels":   noise_thin,
            "wanted_coverage_thick":     cov_thick,
            "hit_over_recon_thin":       prec_thin,
            "recon_pixels_thick_reference": int(cv2.countNonZero(recon)),
            **({"removed_intersection_pixels": int(cv2.countNonZero(blobs))} if blobs is not None else {}),
        },
        "tile_stats": tile_stats,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\n[OK] -> {run_dir}")
    print(f"     bins={len(bins)} (step={CONFIG['ANGLE_STEP_DEG']} deg)  "
          f"wanted={wanted_px}px  thin_recon={thin_px}px")
    print(f"     noise/thin={noise_thin/thin_px:.6g}  hit/thin={prec_thin:.6g}  "
          f"coverage_thick={cov_thick:.6g}")
    print(f"     skeletonize={CONFIG['USE_SKELETONIZE']}  "
          f"rm_intersections={CONFIG.get('REMOVE_INTERSECTIONS', False)}  "
          f"thickness(manual)={runtime_params.get('DRAW_THICKNESS')}")
    if width_info is not None:
        print(f"     filament_width~{width_info['width_px']:.3f}px  ->  "
              f"thr={runtime_params['HOUGH_THRESHOLD']}  "
              f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
              f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
              f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}")


if __name__ == "__main__":
    main()
