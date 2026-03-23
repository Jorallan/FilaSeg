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
import os, json, math, time
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# CONFIG  –  adjust here
# ============================================================
CONFIG: Dict[str, Any] = {
    # --- IO ---
    "INPUT_PATH":         r"C:\Repos\filaments_quantification\stringart\input\cnt_orient_0003_merge.png",
    "OUTPUT_ROOT":        r"C:\Repos\filaments_quantification\stringart\output\cnt_orient_0003_merge",
    "OUTPUT_FOLDER_NAME": None,   # None → timestamp; or set a string like "my_run"
    "BIN_THRESHOLD":      127,

    # --- Tiling / binning ---
    "TILE_SIZE":       128,   # tile width/height in px; smaller = more local fits, more seams
    "ANGLE_STEP_DEG":  15,   # angle-bin width over [0°,180°); smaller = more bins

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
}


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
                # try each thickness; pick the one that covers the most new residual pixels
                best_lm, best_newpix, best_th = None, -1, None
                for th in th_try:
                    lm = draw_line_mask(residual.shape, x1, y1, x2, y2, th)
                    np_ = cv2.countNonZero(cv2.bitwise_and(residual, lm))
                    if np_ > best_newpix:
                        best_newpix, best_lm, best_th = int(np_), lm, th

                if best_lm is None or best_newpix < min_newpix:
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
# Main
# ============================================================

def main() -> None:
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
        print(f"[AUTO_SCALE] width≈{wp:.3f}px  radius≈{rp_:.3f}px  samples={s}")
        print(f"[AUTO_SCALE] autoscaled → thr={runtime_params['HOUGH_THRESHOLD']}  "
              f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
              f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
              f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}")
        print(f"[AUTO_SCALE] manual     → thickness={runtime_params['DRAW_THICKNESS']}  "
              f"dil_k={runtime_params['RESIDUAL_DILATE_KERNEL']}  "
              f"dil_i={runtime_params['RESIDUAL_DILATE_ITERS']}")

    # Step 2: optionally fine-tune the 4 autoscaled params on a sample of tiles
    if CONFIG.get("AUTO_TUNE", False):
        best = auto_tune_params(img_bin, img_bin_orig, runtime_seed_params=runtime_params)
        if best:
            runtime_params = {k: v for k, v in best.items() if not str(k).startswith("_")}
            print(f"[TUNE] → thr={runtime_params['HOUGH_THRESHOLD']}  "
                  f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
                  f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
                  f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}  "
                  f"thickness(fixed)={runtime_params['DRAW_THICKNESS']}")
            print(f"[TUNE]   score={best['_tune_score']:.6g}  "
                  f"cov={best['_tune_cov_thick']:.6g}  "
                  f"noise={best['_tune_noise_frac']:.6g}  "
                  f"prec={best['_tune_prec_thin']:.6g}")

    # --- output folder ---
    folder_name = (CONFIG.get("OUTPUT_FOLDER_NAME") or "").strip()
    base_name   = folder_name if folder_name else ts()
    run_dir     = os.path.join(CONFIG["OUTPUT_ROOT"], base_name)
    if os.path.exists(run_dir):
        run_dir = f"{run_dir}_{ts()}"
    branches_dir = os.path.join(run_dir, "branches")
    ensure_dir(branches_dir)

    # --- run ---
    recon, recon_thin, branches, tile_stats, bins = process_image_tiled(img_bin, runtime_params)

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
            json.dump({"filament_width_px": width_info.get("width_px", 1.0),
                       "radius_px":         width_info.get("radius_px", 0.5)}, _fw, indent=2)

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

    print(f"\n[OK] → {run_dir}")
    print(f"     bins={len(bins)} (step={CONFIG['ANGLE_STEP_DEG']}°)  "
          f"wanted={wanted_px}px  thin_recon={thin_px}px")
    print(f"     noise/thin={noise_thin/thin_px:.6g}  hit/thin={prec_thin:.6g}  "
          f"coverage_thick={cov_thick:.6g}")
    print(f"     skeletonize={CONFIG['USE_SKELETONIZE']}  "
          f"rm_intersections={CONFIG.get('REMOVE_INTERSECTIONS', False)}  "
          f"thickness(manual)={runtime_params.get('DRAW_THICKNESS')}")
    if width_info is not None:
        print(f"     filament_width≈{width_info['width_px']:.3f}px  →  "
              f"thr={runtime_params['HOUGH_THRESHOLD']}  "
              f"minlen={runtime_params['HOUGH_MIN_LINE_LENGTH']}  "
              f"gap={runtime_params['HOUGH_MAX_LINE_GAP']}  "
              f"min_newpix={runtime_params['MIN_ACCEPT_NEWPIX']}")


if __name__ == "__main__":
    main()
