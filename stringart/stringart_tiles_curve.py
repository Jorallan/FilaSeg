# stringart_tiles.py
"""
Tile-wise greedy reconstruction ("string art"-style) for binary filament masks
using ONLY a quadratic curve primitive (Bezier curve), reconstructed in stages
of increasing curvature.

Key changes vs the original:
1) No line primitive drawing path (everything is a quadratic curve).
2) Stage-based fitting:
   - Stage 1: near-straight curves (bend ~ 0)
   - Stage 2..N: progressively higher bend
   - Final stages: high bend (approaches arc-like / circular-looking local fits)
3) Stages can be user-defined or auto-generated.

Notes:
- Candidate generation still uses HoughLinesP to propose chord endpoints (p0,p2).
- Actual drawing is ALWAYS a quadratic curve p0 -> p1 -> p2.
- "Circular" here is approximate (quadratic Bezier cannot represent a perfect circle).
"""

from __future__ import annotations
import os, json, math, time
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# CONFIG (ADJUST HERE)
# ============================================================

CONFIG: Dict[str, Any] = {
    # IO
    "INPUT_PATH": r"C:\Repos\filaments_quantification\2_stringart\input\experiment1.png",
    "OUTPUT_ROOT": r"C:\Repos\filaments_quantification\2_stringart\output",
    "BIN_THRESHOLD": 127,
    "OUTPUT_FOLDER_NAME": "",  # optional custom output folder; if exists, timestamp suffix is added

    # Tiling / angle bins
    "TILE_SIZE": 128,
    "ANGLE_STEP_DEG": 15,

    # Preprocessing
    "USE_SKELETONIZE": True,
    "THINNING_MAX_ITERS": 1,

    # Optional junction/blob removal
    "REMOVE_INTERSECTIONS": False,
    "INTERSECTION_KERNEL": 3,
    "INTERSECTION_ITERS": 1,
    "INTERSECTION_BLOB_EXPAND_PX": 1,

    # Greedy acceptance
    "MAX_CURVES_PER_TILE": 350,
    "MIN_ACCEPT_NEWPIX": 6,      # min residual pixels covered by accepted candidate
    "DRAW_THICKNESS": 3,         # thick raster used for subtraction + output
    "MAX_CANDIDATES_TO_TRY": 300,

    # Hough candidate generation (chord proposals only)
    "HOUGH_RHO": 1.0,
    "HOUGH_THETA_RAD": float(np.pi / 180),
    "HOUGH_THRESHOLD": 15,
    "HOUGH_MIN_LINE_LENGTH": 4,
    "HOUGH_MAX_LINE_GAP": 5,

    # Residual helper for Hough
    "RESIDUAL_DILATE_KERNEL": 2,
    "RESIDUAL_DILATE_ITERS": 1,

    # -------- Quadratic curve primitive --------
    # Curve sampling step (pixels between sampled points along the Bezier)
    "CURVE_STEP_PX": 1.0,

    # Try +bend and -bend (curve bows to either side of chord)
    "CURVE_TRY_BOTH_SIDES": True,

    # Relative bend candidates INSIDE a stage.
    # bend_px = factor * chord_length
    # Keep small list to avoid exploding compute.
    "CURVE_BEND_FACTORS_WITHIN_STAGE": [1.0, 0.6, 0.3],

    # -------- Staged reconstruction --------
    # If True, build stages automatically from tile size (recommended).
    "AUTO_CURVE_STAGES": True,

    # Manual stages (used only if AUTO_CURVE_STAGES=False)
    # Each stage defines a max bend factor relative to chord length.
    # Example: 0.00 ~ straight, 0.05 mild, 0.15 stronger, 0.35 very curved
    "CURVE_STAGE_MAX_BEND_FACTORS": [0.00, 0.04, 0.08, 0.14, 0.22, 0.34, 0.50],

    # Auto-stage generation knobs
    "AUTO_STAGE_COUNT_MIN": 10,
    "AUTO_STAGE_COUNT_MAX": 20,
    "AUTO_STAGE_MAX_BEND_FACTOR": 0.7,   # ~very curved, near semi-circle-ish local arcs
    "AUTO_STAGE_GAMMA": 1.8,              # >1 packs more stages into low-curvature region first

    # Stage behavior
    "STAGE_STOP_NO_IMPROVEMENT_ROUNDS": 1,  # stop stage when no candidate accepted this many outer rounds
    "STAGE_MIN_NEWPIX_SCALE": [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.75, 0.75, 0.7],  # per stage relaxation

    # Output control
    "SAVE_BRANCHES": True,
    "SAVE_OVERLAY": True,

    # Overlay control
    "OVERLAY_ALPHA": 0.55,
    "OVERLAY_RECON_COLOR_BGR": [0, 255, 0],
    "OVERLAY_REMOVED_COLOR_BGR": [0, 0, 255],
}


# ============================================================
# Small utils
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

def binarize_u8(img_u8: np.ndarray, thr: int) -> np.ndarray:
    return cv2.threshold(img_u8, int(thr), 255, cv2.THRESH_BINARY)[1]

def to_bool255(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8 > 0).astype(np.uint8) * 255)

def overlay_masks_alpha_on_gray(
    gray_u8: np.ndarray,
    recon255: np.ndarray,
    removed255: Optional[np.ndarray],
    alpha: float,
    recon_color_bgr: Tuple[int, int, int],
    removed_color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR) if gray_u8.ndim == 2 else gray_u8.copy()
    a = float(max(0.0, min(1.0, alpha)))

    def blend(mask255: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
        if mask255 is None:
            return
        m = mask255 > 0
        if not np.any(m):
            return
        c = np.array(color, dtype=np.float32)
        pix = out[m].astype(np.float32)
        out[m] = np.clip((1.0 - a) * pix + a * c, 0, 255).astype(np.uint8)

    blend(recon255, recon_color_bgr)
    blend(removed255, removed_color_bgr)
    return out

def remove_intersection_blobs(
    bin255: np.ndarray, kernel_sz: int, iters: int, expand_px: int
) -> Tuple[np.ndarray, np.ndarray]:
    k = int(kernel_sz)
    if k < 1:
        return bin255.copy(), np.zeros_like(bin255)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    blobs = cv2.morphologyEx(bin255, cv2.MORPH_OPEN, kernel, iterations=int(max(1, iters)))
    blobs = to_bool255(blobs)

    e = int(expand_px)
    if e > 0:
        ek = 2 * e + 1
        eker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        blobs = to_bool255(cv2.dilate(blobs, eker, iterations=1))

    cleaned = to_bool255(cv2.bitwise_and(bin255, cv2.bitwise_not(blobs)))
    return cleaned, blobs


# ============================================================
# Zhang-Suen thinning (binary 0/255 -> 0/255)
# ============================================================

def zhang_suen_thinning(bin255: np.ndarray, max_iters: int = 200) -> np.ndarray:
    img = (bin255 > 0).astype(np.uint8)
    img = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    def neighbors(x: int, y: int) -> List[int]:
        p2 = img[x - 1, y]
        p3 = img[x - 1, y + 1]
        p4 = img[x, y + 1]
        p5 = img[x + 1, y + 1]
        p6 = img[x + 1, y]
        p7 = img[x + 1, y - 1]
        p8 = img[x, y - 1]
        p9 = img[x - 1, y - 1]
        return [p2, p3, p4, p5, p6, p7, p8, p9]

    def transitions(nb: List[int]) -> int:
        n = nb + [nb[0]]
        return sum((n[i] == 0 and n[i + 1] == 1) for i in range(8))

    changed, it = True, 0
    while changed and it < int(max_iters):
        changed = False
        it += 1
        for step in (0, 1):
            to_del = []
            xs, ys = np.where(img == 1)
            for x, y in zip(xs, ys):
                if x == 0 or y == 0 or x == img.shape[0] - 1 or y == img.shape[1] - 1:
                    continue
                nb = neighbors(x, y)
                B = sum(nb)
                if not (2 <= B <= 6):
                    continue
                if transitions(nb) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = nb
                if step == 0:
                    if p2 * p4 * p6 != 0: continue
                    if p4 * p6 * p8 != 0: continue
                else:
                    if p2 * p4 * p8 != 0: continue
                    if p2 * p6 * p8 != 0: continue
                to_del.append((x, y))
            if to_del:
                for x, y in to_del:
                    img[x, y] = 0
                changed = True

    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


# ============================================================
# Geometry / curve helpers
# ============================================================

def seg_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return (math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0)

def angle_bins(step_deg: int) -> List[Tuple[float, float]]:
    step = float(max(1, int(step_deg)))
    out, a = [], 0.0
    while a < 180.0:
        out.append((a, min(180.0, a + step)))
        a += step
    return out

def _quad_bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, step_px: float) -> np.ndarray:
    """
    Sample a quadratic Bezier with approximately step_px pixel spacing.

    step_px:
      Smaller -> denser sampling -> smoother rasterized curve, slower
      Larger  -> fewer points -> faster but can look jagged / skip pixels
    """
    # polygon length estimate for sample count (better than chord-only)
    poly_len = float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1))
    n = int(max(3, math.ceil(poly_len / max(1e-6, float(step_px))) + 1))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    pts = ((1 - t) * (1 - t) * p0[None, :] +
           2 * (1 - t) * t * p1[None, :] +
           (t * t) * p2[None, :])
    return pts

def draw_curve_mask_quadratic(
    shape_hw: Tuple[int, int],
    x1: int, y1: int, x2: int, y2: int,
    bend_px: float,
    step_px: float,
    thickness: int,
    bend_sign: float = +1.0,
) -> np.ndarray:
    """
    Draw a quadratic curve primitive (p0,p1,p2).
    p1 = midpoint + bend_sign * bend_px * unit_perp(chord)

    bend_px:
      Control-point offset magnitude in pixels measured perpendicular to the chord.
      0 px -> exactly straight (but still rasterized through the curve path logic)
      Larger -> more curvature.
    """
    H, W = shape_hw
    p0 = np.array([float(x1), float(y1)], dtype=np.float32)
    p2 = np.array([float(x2), float(y2)], dtype=np.float32)
    v = p2 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(m, (int(round(x1)), int(round(y1))), max(1, thickness // 2), 255, -1)
        return m

    perp = np.array([-v[1], v[0]], dtype=np.float32) / float(L)
    mid = 0.5 * (p0 + p2)
    p1 = mid + float(bend_sign) * float(bend_px) * perp

    pts = _quad_bezier_points(p0, p1, p2, step_px=float(step_px))
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    m = np.zeros((H, W), dtype=np.uint8)
    if len(pts) == 1:
        cv2.circle(m, tuple(pts[0]), max(1, thickness // 2), 255, -1)
    else:
        cv2.polylines(m, [pts.reshape(-1, 1, 2)], isClosed=False, color=255,
                      thickness=int(thickness), lineType=cv2.LINE_8)
    return m


# ============================================================
# Stage schedule (curvature progression)
# ============================================================

def build_curve_stage_bend_factors(tile_size: int) -> List[float]:
    """
    Returns a list of stage max bend factors (relative to chord length).
    bend_px = factor * chord_length

    If AUTO_CURVE_STAGES=True:
      Generate a monotonic schedule from low curvature to high curvature.
      Gamma > 1 biases resolution toward low-curvature stages.
    """
    if not bool(CONFIG.get("AUTO_CURVE_STAGES", True)):
        vals = [float(v) for v in CONFIG.get("CURVE_STAGE_MAX_BEND_FACTORS", [0.0, 0.06, 0.15, 0.35])]
        vals = sorted(max(0.0, v) for v in vals)
        if not vals or vals[0] != 0.0:
            vals = [0.0] + vals
        return vals

    # crude adaptive stage count based on tile size (larger tile => potentially more geometry variety)
    t = int(max(16, tile_size))
    n_min = int(CONFIG.get("AUTO_STAGE_COUNT_MIN", 5))
    n_max = int(CONFIG.get("AUTO_STAGE_COUNT_MAX", 9))
    n = int(round(np.interp(t, [64, 256], [n_min, n_max])))
    n = max(n_min, min(n_max, n))

    max_bf = float(CONFIG.get("AUTO_STAGE_MAX_BEND_FACTOR", 0.55))
    gamma = float(CONFIG.get("AUTO_STAGE_GAMMA", 1.8))
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    vals = (x ** gamma) * max_bf
    vals[0] = 0.0
    return [float(v) for v in vals]


# ============================================================
# Greedy decomposition (tile)
# ============================================================

def greedy_decompose_tile(tile_bin255: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], dict]:
    tile_bin255 = to_bool255(tile_bin255)
    target = zhang_suen_thinning(tile_bin255, int(CONFIG["THINNING_MAX_ITERS"])) if CONFIG["USE_SKELETONIZE"] else tile_bin255.copy()
    residual = target.copy()

    bins = angle_bins(int(CONFIG["ANGLE_STEP_DEG"]))
    branches = [np.zeros_like(tile_bin255) for _ in bins]
    recon = np.zeros_like(tile_bin255)
    recon_thin = np.zeros_like(tile_bin255)

    # Hough helper dilation
    dil_kernel = None
    k = int(CONFIG.get("RESIDUAL_DILATE_KERNEL", 0))
    if k >= 3:
        if k % 2 == 0: k += 1
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil_iters = int(max(0, CONFIG.get("RESIDUAL_DILATE_ITERS", 0)))

    # knobs
    h_rho = float(CONFIG["HOUGH_RHO"])
    h_theta = float(CONFIG["HOUGH_THETA_RAD"])
    h_thr = int(CONFIG["HOUGH_THRESHOLD"])
    h_minlen = int(CONFIG["HOUGH_MIN_LINE_LENGTH"])
    h_maxgap = int(CONFIG["HOUGH_MAX_LINE_GAP"])

    max_curves = int(CONFIG["MAX_CURVES_PER_TILE"])
    base_min_newpix = int(CONFIG["MIN_ACCEPT_NEWPIX"])
    thickness = int(CONFIG["DRAW_THICKNESS"])
    max_try = int(CONFIG["MAX_CANDIDATES_TO_TRY"])

    curve_step = float(CONFIG.get("CURVE_STEP_PX", 1.0))
    curve_try_both = bool(CONFIG.get("CURVE_TRY_BOTH_SIDES", True))
    within_stage_factors = [float(v) for v in CONFIG.get("CURVE_BEND_FACTORS_WITHIN_STAGE", [1.0, 0.6, 0.3])]

    stage_bend_factors = build_curve_stage_bend_factors(tile_size=max(tile_bin255.shape))
    stage_min_newpix_scales = [float(v) for v in CONFIG.get("STAGE_MIN_NEWPIX_SCALE", [1.0])]

    accepted_total = 0
    accepted_per_bin = [0] * len(bins)
    accepted_per_stage = [0] * len(stage_bend_factors)
    stage_log = []

    # Main staged fitting loop
    for si, stage_max_bf in enumerate(stage_bend_factors):
        if accepted_total >= max_curves:
            break
        if cv2.countNonZero(residual) == 0:
            break

        # progressively relax acceptance in later stages
        scale = stage_min_newpix_scales[min(si, len(stage_min_newpix_scales) - 1)] if stage_min_newpix_scales else 1.0
        min_newpix = max(1, int(round(base_min_newpix * scale)))

        no_improve_rounds = 0
        stage_accept_before = accepted_total

        for bi, (a0, a1) in enumerate(bins):
            if accepted_total >= max_curves or cv2.countNonZero(residual) == 0:
                break

            accepted_any_in_bin = False
            while accepted_total < max_curves:
                if cv2.countNonZero(residual) == 0:
                    break

                edges = cv2.dilate(residual, dil_kernel, iterations=dil_iters) if (dil_kernel is not None and dil_iters > 0) else residual
                lines = cv2.HoughLinesP(
                    edges,
                    rho=h_rho,
                    theta=h_theta,
                    threshold=h_thr,
                    minLineLength=h_minlen,
                    maxLineGap=h_maxgap,
                )
                if lines is None or len(lines) == 0:
                    break

                # chord candidates in this angle bin
                cand = []
                for x1, y1, x2, y2 in lines[:, 0, :]:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    L = float(math.hypot(x2 - x1, y2 - y1))
                    if L < h_minlen:
                        continue
                    ang = seg_angle_deg(x1, y1, x2, y2)
                    in_bin = (a0 <= ang < a1) if (bi < len(bins) - 1) else (a0 <= ang <= a1)
                    if not in_bin:
                        continue
                    cand.append((L, (x1, y1, x2, y2)))

                if not cand:
                    break

                cand.sort(key=lambda t: t[0], reverse=True)

                accepted = False
                for _, (x1, y1, x2, y2) in cand[:max_try]:
                    chord_len = float(math.hypot(x2 - x1, y2 - y1))
                    # stage bend in pixels
                    stage_bend_px = stage_max_bf * chord_len

                    # Build bend options for this stage:
                    # - Stage 0 always includes exactly 0
                    # - Later stages try current max and smaller fractions of it
                    bend_px_options = []
                    if stage_bend_px <= 1e-9:
                        bend_px_options = [0.0]
                    else:
                        bend_px_options = [max(0.0, stage_bend_px * f) for f in within_stage_factors]
                        # include exact 0 in all stages for safety, but prioritize curved first
                        bend_px_options.append(0.0)
                        # unique + sorted descending (try stronger curvature first inside stage)
                        bend_px_options = sorted(set(float(round(v, 6)) for v in bend_px_options), reverse=True)

                    best_lm = None
                    best_lm_thin = None
                    best_newpix = -1
                    best_meta = None

                    for bend_px in bend_px_options:
                        signs = [+1.0, -1.0] if (curve_try_both and bend_px > 1e-9) else [+1.0]
                        for sgn in signs:
                            lm = draw_curve_mask_quadratic(
                                residual.shape, x1, y1, x2, y2,
                                bend_px=bend_px, step_px=curve_step,
                                thickness=thickness, bend_sign=sgn
                            )
                            newpix = int(cv2.countNonZero(cv2.bitwise_and(residual, lm)))
                            if newpix > best_newpix:
                                best_newpix = newpix
                                best_lm = lm
                                best_meta = {"bend_px": float(bend_px), "bend_sign": float(sgn)}
                                best_lm_thin = draw_curve_mask_quadratic(
                                    residual.shape, x1, y1, x2, y2,
                                    bend_px=bend_px, step_px=curve_step,
                                    thickness=1, bend_sign=sgn
                                )

                    if best_newpix < min_newpix or best_lm is None:
                        continue

                    # accept
                    residual = cv2.bitwise_and(residual, cv2.bitwise_not(best_lm))
                    recon = cv2.bitwise_or(recon, best_lm)
                    recon_thin = cv2.bitwise_or(recon_thin, best_lm_thin)
                    branches[bi] = cv2.bitwise_or(branches[bi], best_lm)

                    accepted_total += 1
                    accepted_per_bin[bi] += 1
                    accepted_per_stage[si] += 1
                    accepted_any_in_bin = True
                    accepted = True
                    break

                if not accepted:
                    break

            if not accepted_any_in_bin:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0

            if no_improve_rounds >= int(CONFIG.get("STAGE_STOP_NO_IMPROVEMENT_ROUNDS", 1)):
                # stage stalled
                pass

        stage_log.append({
            "stage_index": int(si),
            "stage_max_bend_factor": float(stage_max_bf),
            "stage_max_bend_px_at_chord10": float(stage_max_bf * 10.0),
            "min_accept_newpix_used": int(min_newpix),
            "accepted_in_stage": int(accepted_total - stage_accept_before),
            "residual_pixels_after_stage": int(cv2.countNonZero(residual)),
        })

    stats = {
        "accepted_total": int(accepted_total),
        "accepted_per_bin": [int(v) for v in accepted_per_bin],
        "accepted_per_stage": [int(v) for v in accepted_per_stage],
        "residual_pixels_left": int(cv2.countNonZero(residual)),
        "initial_target_pixels": int(cv2.countNonZero(target)),
        "angle_bins": bins,
        "used_skeletonize": bool(CONFIG["USE_SKELETONIZE"]),
        "curve_step_px": float(curve_step),
        "curve_try_both_sides": bool(curve_try_both),
        "curve_stage_max_bend_factors": [float(v) for v in stage_bend_factors],
        "stage_log": stage_log,
        "primitive_mode": "quadratic_curve_only",
    }
    return recon, recon_thin, branches, stats


# ============================================================
# Tiling + stitching
# ============================================================

def iter_tiles(h: int, w: int, tile: int):
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            yield y0, y1, x0, x1

def process_image_tiled(img_bin255: np.ndarray):
    h, w = img_bin255.shape
    t = int(CONFIG["TILE_SIZE"])
    bins = angle_bins(int(CONFIG["ANGLE_STEP_DEG"]))

    recon_full = np.zeros((h, w), dtype=np.uint8)
    recon_thin_full = np.zeros((h, w), dtype=np.uint8)
    branches_full = [np.zeros((h, w), dtype=np.uint8) for _ in bins]
    tile_stats = []

    for y0, y1, x0, x1 in iter_tiles(h, w, t):
        tile = img_bin255[y0:y1, x0:x1]
        if cv2.countNonZero(tile) == 0:
            continue

        recon_t, recon_thin_t, branches_t, stats = greedy_decompose_tile(tile)
        recon_full[y0:y1, x0:x1] = cv2.bitwise_or(recon_full[y0:y1, x0:x1], recon_t)
        recon_thin_full[y0:y1, x0:x1] = cv2.bitwise_or(recon_thin_full[y0:y1, x0:x1], recon_thin_t)

        for i in range(len(bins)):
            branches_full[i][y0:y1, x0:x1] = cv2.bitwise_or(branches_full[i][y0:y1, x0:x1], branches_t[i])

        stats["tile_bbox"] = [int(y0), int(y1), int(x0), int(x1)]
        tile_stats.append(stats)

    return recon_full, recon_thin_full, branches_full, tile_stats, bins


# ============================================================
# Main
# ============================================================

def main():
    img = read_gray(CONFIG["INPUT_PATH"])
    img_bin_orig = to_bool255(binarize_u8(img, int(CONFIG["BIN_THRESHOLD"])))
    img_bin = img_bin_orig.copy()

    blobs = None
    if bool(CONFIG.get("REMOVE_INTERSECTIONS", False)):
        img_bin, blobs = remove_intersection_blobs(
            img_bin,
            kernel_sz=int(CONFIG.get("INTERSECTION_KERNEL", 3)),
            iters=int(CONFIG.get("INTERSECTION_ITERS", 1)),
            expand_px=int(CONFIG.get("INTERSECTION_BLOB_EXPAND_PX", 0)),
        )

    # output folder handling (custom name + collision-safe suffix)
    folder_name = CONFIG.get("OUTPUT_FOLDER_NAME")
    folder_name = str(folder_name).strip() if folder_name is not None else ""
    base_name = folder_name if folder_name else ts()

    run_dir = os.path.join(CONFIG["OUTPUT_ROOT"], base_name)
    if os.path.exists(run_dir):
        run_dir = os.path.join(CONFIG["OUTPUT_ROOT"], f"{base_name}_{ts()}")

    branches_dir = os.path.join(run_dir, "branches")
    ensure_dir(branches_dir)

    recon, recon_thin, branches, tile_stats, bins = process_image_tiled(img_bin)

    # remove blobs from outputs if requested
    if blobs is not None:
        nb = cv2.bitwise_not(blobs)
        branches = [to_bool255(cv2.bitwise_and(b, nb)) for b in branches]
        recon = np.zeros_like(recon)
        for b in branches:
            recon = cv2.bitwise_or(recon, b)
        recon_thin = to_bool255(cv2.bitwise_and(recon_thin, nb))

    # save images
    cv2.imwrite(os.path.join(run_dir, "original.png"), img_bin_orig)
    cv2.imwrite(os.path.join(run_dir, "reconstructed.png"), recon)
    cv2.imwrite(os.path.join(run_dir, "reconstructed_thin.png"), recon_thin)

    if bool(CONFIG.get("SAVE_OVERLAY", True)):
        overlay = overlay_masks_alpha_on_gray(
            gray_u8=img_bin_orig,
            recon255=recon,
            removed255=blobs,
            alpha=float(CONFIG.get("OVERLAY_ALPHA", 0.55)),
            recon_color_bgr=tuple(int(v) for v in CONFIG.get("OVERLAY_RECON_COLOR_BGR", [0, 255, 0])),
            removed_color_bgr=tuple(int(v) for v in CONFIG.get("OVERLAY_REMOVED_COLOR_BGR", [0, 0, 255])),
        )
        cv2.imwrite(os.path.join(run_dir, "overlay.png"), overlay)

    base = os.path.splitext(os.path.basename(CONFIG["INPUT_PATH"]))[0]
    if bool(CONFIG.get("SAVE_BRANCHES", True)):
        for i in range(len(bins)):
            cv2.imwrite(os.path.join(branches_dir, f"{base}_branch_{i+1}.png"), branches[i])

        branches_merge = np.zeros_like(recon)
        for b in branches:
            branches_merge = cv2.bitwise_or(branches_merge, b)
        cv2.imwrite(os.path.join(branches_dir, f"{base}_branches_merge.png"), branches_merge)

    # summary metrics
    wanted = img_bin_orig
    wanted_px = int(cv2.countNonZero(wanted))
    hit_thin_px = int(cv2.countNonZero(cv2.bitwise_and(recon_thin, wanted)))
    noise_thin_px = int(cv2.countNonZero(cv2.bitwise_and(recon_thin, cv2.bitwise_not(wanted))))
    thin_px = int(cv2.countNonZero(recon_thin))
    hit_thick_px = int(cv2.countNonZero(cv2.bitwise_and(recon, wanted)))

    coverage_thick = float(hit_thick_px) / float(wanted_px if wanted_px > 0 else 1)
    precision_like_thin = float(hit_thin_px) / float(thin_px if thin_px > 0 else 1)

    cfg_out = {
        "CONFIG": CONFIG,
        "run_dir": run_dir,
        "angle_bins": bins,
        "summary": {
            "wanted_input_pixels": wanted_px,
            "recon_thin_pixels": thin_px,
            "recon_thin_hit_pixels": hit_thin_px,
            "recon_thin_noise_pixels": noise_thin_px,
            "wanted_coverage_thick": coverage_thick,
            "hit_over_recon_thin": precision_like_thin,
            "recon_pixels_thick_reference": int(cv2.countNonZero(recon)),
        },
        "tile_stats": tile_stats,
    }
    if blobs is not None:
        cfg_out["summary"]["removed_intersection_pixels"] = int(cv2.countNonZero(blobs))

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    # compact terminal output
    stage_schedule = build_curve_stage_bend_factors(int(CONFIG["TILE_SIZE"]))
    print(f"[OK] Saved run to: {run_dir}")
    print(f"     bins: {len(bins)} (step={CONFIG['ANGLE_STEP_DEG']} deg)")
    print(f"     primitive: quadratic_curve_only")
    print(f"     stages (bend_factor max): {[round(v,4) for v in stage_schedule]}")
    print(f"     curve_step_px={CONFIG.get('CURVE_STEP_PX', 1.0)}")
    print(f"     wanted input pixels: {wanted_px}")
    print(f"     recon_thin pixels: {thin_px} (hit: {hit_thin_px}, noise: {noise_thin_px})")
    print(f"     thin ratios: noise/thin={noise_thin_px / (thin_px if thin_px > 0 else 1):.6g}, hit/thin={precision_like_thin:.6g}")
    print(f"     wanted coverage (thick): {coverage_thick:.6g}")
    print(f"     skeletonize: {CONFIG['USE_SKELETONIZE']}")
    print(f"     rm_intersections: {CONFIG.get('REMOVE_INTERSECTIONS', False)}")


if __name__ == "__main__":
    main()