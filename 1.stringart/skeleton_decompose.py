"""
WORK IN PROGRESS / EXPERIMENTAL.

Skeleton-first per-orientation decomposition. This is being evaluated as a
possible stage-1 replacement for stringart_tiles.py, but it is not final
pipeline behavior yet. Keep outputs reviewable and expect tuning changes.

Algorithm:
  1. Read binary mask.
  2. Estimate filament width with the distance transform (median DT * 2 on fg).
  3. Skeletonize the full mask (cv2.ximgproc.thinning, with skimage fallback).
  4. Find junctions (skeleton pixels with >=3 skeleton neighbors) and remove
     them to split the skeleton into single-curve pieces.
  5. For each piece: order pixels endpoint-to-endpoint (BFS diameter), then
     compute a smoothed per-pixel tangent using a fixed window.
  6. Bin each pixel by its tangent angle (mod 180 deg) into N angle bins
     (N = 180 / ANGLE_STEP_DEG).
  7. Re-attach junction pixels to every bin that any neighbor was assigned to.
  8. For each bin, dilate the assigned skeleton pixels by ~filament_radius
     and AND with the original mask so branches never leak past the mask.

Output contract (matches stringart_tiles.py for downstream stages):
  <output_root>/<output_folder>/
      original.png                 - copy of input mask
      reconstructed.png            - union of all bin masks (sanity preview)
      overlay.png                  - green-recon over original
      branches/
          mask_branch_1.png ... mask_branch_N.png
          mask_branches_merge.png
          filament_width.json
      run_config.json              - has angle_bins + ANGLE_STEP_DEG + meta
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ============================================================
# CONFIG (experimental defaults; CLI overrides win)
# ============================================================
CONFIG = {
    "INPUT_PATH":         "",
    "OUTPUT_ROOT":        "",
    "OUTPUT_FOLDER_NAME": "1.stringart",
    "BIN_THRESHOLD":      127,
    "ANGLE_STEP_DEG":     15,

    # binning_mode = "per_pixel" -> each skeleton pixel binned by its local tangent
    #                              (handles curvature; can fragment if window too small)
    #               = "per_piece" -> whole piece binned by its median tangent
    #                              (much cleaner branches; loses smooth-curve splitting)
    #               = "hybrid"    -> per_piece for short pieces, per_pixel for longer
    #                              pieces. Default.
    "BINNING_MODE":       "hybrid",
    "HYBRID_LEN_PX":      20,

    # tangent smoothing window (pixels along the path used to estimate direction)
    # If 0 -> auto: max(12, 5*filament_width_px).
    "TANGENT_WINDOW_PX":  0,

    # junction handling: dilation radius (px) used when removing junctions to
    # ensure pieces fully separate. 0 disables.
    "JUNCTION_DILATE_PX": 1,

    # spur pruning: remove skeleton tips whose path back to a junction is
    # shorter than this. 0 disables. Helps avoid spurious junctions caused by
    # bumpy mask edges.
    "SPUR_PRUNE_PX":      5,

    # filament width estimation (same convention as stringart_tiles)
    "WIDTH_PERCENTILE":   60.0,
    "WIDTH_SAMPLE_MAX":   200_000,
    "WIDTH_MIN":          1.0,
    "WIDTH_MAX":          12.0,

    # per-branch dilation: extra px added to filament_radius (+1 ensures
    # skeleton sits squarely inside the branch mask)
    "BRANCH_DILATE_PAD_PX": 1,

    # discard pieces whose ordered length is below this many pixels
    "MIN_PIECE_LEN":      4,
}


# ============================================================
# Helpers
# ============================================================

def angle_bins(step_deg: int) -> List[Tuple[float, float]]:
    step = float(max(1, int(step_deg)))
    bins, a = [], 0.0
    while a < 180.0:
        bins.append((a, min(180.0, a + step)))
        a += step
    return bins


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def binarize(img: np.ndarray, thr: int) -> np.ndarray:
    return (cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)[1] > 0)


def skeletonize_full(mask_bool: np.ndarray) -> np.ndarray:
    """Whole-image skeletonization. Uses cv2.ximgproc if available."""
    src = (mask_bool.astype(np.uint8)) * 255
    try:
        return (cv2.ximgproc.thinning(src,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0)
    except AttributeError:
        try:
            from skimage.morphology import skeletonize as _skel
            return _skel(mask_bool)
        except Exception:
            raise RuntimeError(
                "Neither cv2.ximgproc.thinning nor skimage.morphology.skeletonize "
                "is available. Install opencv-contrib-python or scikit-image.")


def neighbor_count_8(skel_bool: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-neighbors that are also skeleton, valid only on skel."""
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = cv2.filter2D(skel_bool.astype(np.uint8), -1, k,
                      borderType=cv2.BORDER_CONSTANT)
    return nb * skel_bool.astype(np.uint8)


def find_junctions(skel_bool: np.ndarray) -> np.ndarray:
    nb = neighbor_count_8(skel_bool)
    return (nb >= 3) & skel_bool


def find_endpoints(skel_bool: np.ndarray) -> np.ndarray:
    nb = neighbor_count_8(skel_bool)
    return (nb == 1) & skel_bool


def prune_spurs(skel_bool: np.ndarray, max_spur_len: int) -> np.ndarray:
    """Remove only short endpoint-to-junction side branches."""
    if max_spur_len <= 0:
        return skel_bool.copy()

    sk = skel_bool.copy()
    max_len = int(max_spur_len)
    changed = True
    while changed:
        changed = False
        nb = neighbor_count_8(sk)
        endpoints = np.argwhere((nb == 1) & sk)
        junctions = (nb >= 3) & sk
        remove = np.zeros_like(sk, dtype=bool)

        for r, c in endpoints:
            path = [(int(r), int(c))]
            prev = None
            cur = path[0]
            while len(path) <= max_len:
                cr, cc = cur
                nbrs = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc2 = cr + dr, cc + dc
                        if (0 <= rr < sk.shape[0] and 0 <= cc2 < sk.shape[1]
                                and sk[rr, cc2] and (rr, cc2) != prev):
                            nbrs.append((rr, cc2))
                if not nbrs:
                    break
                if len(nbrs) > 1:
                    break
                nxt = nbrs[0]
                if junctions[nxt]:
                    rr, cc2 = zip(*path)
                    remove[np.asarray(rr), np.asarray(cc2)] = True
                    changed = True
                    break
                if nb[nxt] == 1:
                    break
                prev, cur = cur, nxt
                path.append(cur)

        sk &= ~remove
    return sk


def estimate_filament_width_px(mask_bool: np.ndarray, cfg: dict) -> dict:
    fg = mask_bool.astype(np.uint8)
    if fg.sum() == 0:
        return {"width_px": 1.0, "radius_px": 0.5, "n_samples": 0}
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    vals = dt[fg > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return {"width_px": 1.0, "radius_px": 0.5, "n_samples": 0}
    sample_max = int(cfg["WIDTH_SAMPLE_MAX"])
    if vals.size > sample_max:
        idx = np.random.default_rng(0).choice(vals.size, size=sample_max, replace=False)
        vals = vals[idx]
    rad = float(np.percentile(vals, float(cfg["WIDTH_PERCENTILE"])))
    w = float(np.clip(2.0 * rad, cfg["WIDTH_MIN"], cfg["WIDTH_MAX"]))
    return {"width_px": w, "radius_px": rad, "n_samples": int(vals.size)}


# ============================================================
# Piece extraction + ordering
# ============================================================

def split_at_junctions(skel_bool: np.ndarray, junction_dilate_px: int):
    """Return (pieces_bool, junctions_bool_dilated).

    pieces_bool excludes a small junction neighborhood so connected components
    on it correspond to single-curve segments.
    """
    junc = find_junctions(skel_bool)
    if junction_dilate_px > 0:
        junc_dil = cv2.dilate(junc.astype(np.uint8),
                              np.ones((3, 3), np.uint8),
                              iterations=int(junction_dilate_px)).astype(bool)
    else:
        junc_dil = junc
    pieces = skel_bool & ~junc_dil
    return pieces, junc_dil


def _bfs_farthest(start: Tuple[int, int],
                  piece_mask: np.ndarray):
    """BFS within an 8-connected piece. Returns (last_visited, parent_map)."""
    H, W = piece_mask.shape
    seen = {start: None}
    queue = deque([start])
    last = start
    while queue:
        p = queue.popleft()
        last = p
        r, c = p
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and piece_mask[rr, cc]:
                    n = (rr, cc)
                    if n in seen:
                        continue
                    seen[n] = p
                    queue.append(n)
    return last, seen


def order_piece(piece_mask: np.ndarray) -> np.ndarray:
    """Return ordered (N,2) array of (row, col) pixels along the piece.
    Handles open curves (uses double-BFS for the diameter) and closed loops.
    """
    pts = np.argwhere(piece_mask)
    if pts.size == 0:
        return pts.astype(np.int32)
    nb = neighbor_count_8(piece_mask)
    if len(pts) > 2 and not np.any((nb == 1) & piece_mask):
        center = pts.astype(np.float32).mean(axis=0)
        angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
        return pts[np.argsort(angles)].astype(np.int32)
    start = tuple(int(v) for v in pts[0])
    a, _ = _bfs_farthest(start, piece_mask)
    b, parent = _bfs_farthest(a, piece_mask)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = parent[p]
    return np.asarray(path[::-1], dtype=np.int32)


# ============================================================
# Tangent + binning
# ============================================================

def smooth_tangents(path_rc: np.ndarray, window: int) -> np.ndarray:
    """Per-point unit tangent (dr, dc), smoothed by a centered window."""
    n = len(path_rc)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    w = max(1, int(window))
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        a = max(0, i - w)
        b = min(n - 1, i + w)
        d = path_rc[b].astype(np.float32) - path_rc[a].astype(np.float32)
        norm = float(np.hypot(d[0], d[1]))
        if norm < 1e-6:
            # fall back to global piece direction
            d = path_rc[-1].astype(np.float32) - path_rc[0].astype(np.float32)
            norm = float(np.hypot(d[0], d[1]))
            if norm < 1e-6:
                out[i] = (1.0, 0.0)
                continue
        out[i] = d / norm
    return out


def bin_from_tangent(tan_rc: np.ndarray, step_deg: float, n_bins: int) -> np.ndarray:
    """Map tangents (rows of (dr, dc)) to bin indices [0..n_bins-1]."""
    # angle = atan2(dr, dc) mod 180, in degrees
    angles = np.degrees(np.arctan2(tan_rc[:, 0], tan_rc[:, 1])) % 180.0
    # clamp 180->just below (so bin = n_bins-1)
    angles = np.minimum(angles, 180.0 - 1e-6)
    idx = (angles / float(step_deg)).astype(np.int32)
    np.clip(idx, 0, n_bins - 1, out=idx)
    return idx


# ============================================================
# Main decomposition
# ============================================================

def decompose(mask_bool: np.ndarray, cfg: dict):
    H, W = mask_bool.shape
    bins = angle_bins(cfg["ANGLE_STEP_DEG"])
    n_bins = len(bins)
    step = float(cfg["ANGLE_STEP_DEG"])

    width_info = estimate_filament_width_px(mask_bool, cfg)
    fw = float(width_info["width_px"])

    # tangent window: auto to ~5*filament_width if not set
    tw = int(cfg["TANGENT_WINDOW_PX"])
    if tw <= 0:
        tw = max(12, int(round(5.0 * fw)))

    # 1) skeletonize whole image
    skel = skeletonize_full(mask_bool)

    # 1b) optional spur pruning to suppress spurious junctions on bumpy edges
    spur_px = int(cfg.get("SPUR_PRUNE_PX", 0))
    if spur_px > 0:
        skel = prune_spurs(skel, spur_px)

    # 2) split at junctions
    pieces_bool, junc_dil = split_at_junctions(skel, int(cfg["JUNCTION_DILATE_PX"]))

    # 3) connected components of pieces
    n_lab, labels = cv2.connectedComponents(pieces_bool.astype(np.uint8), connectivity=8)

    # per-bin skeleton-pixel masks
    bin_skel = [np.zeros((H, W), dtype=bool) for _ in range(n_bins)]

    min_piece_len = int(cfg["MIN_PIECE_LEN"])
    binning_mode = str(cfg.get("BINNING_MODE", "hybrid"))
    hybrid_len = int(cfg.get("HYBRID_LEN_PX", 20))
    n_kept = 0
    n_skipped_short = 0
    n_per_piece = 0
    n_per_pixel = 0

    for lab in range(1, n_lab):
        comp = (labels == lab)
        npx = int(comp.sum())
        if npx < min_piece_len:
            n_skipped_short += 1
            continue
        path = order_piece(comp)
        if len(path) < min_piece_len:
            n_skipped_short += 1
            continue
        n_kept += 1

        # Pick binning strategy for this piece
        if binning_mode == "per_piece":
            mode = "per_piece"
        elif binning_mode == "per_pixel":
            mode = "per_pixel"
        else:  # hybrid
            mode = "per_pixel" if len(path) > hybrid_len else "per_piece"

        if mode == "per_piece":
            n_per_piece += 1
            # Whole-piece direction: average tangent (endpoint-to-endpoint for
            # short pieces, smoothed for longer). Use endpoint chord here -
            # most robust for ~straight pieces.
            d = path[-1].astype(np.float32) - path[0].astype(np.float32)
            norm = float(np.hypot(d[0], d[1]))
            if norm < 1e-6:
                continue
            tan = (d / norm).reshape(1, 2)
            bi = int(bin_from_tangent(tan, step, n_bins)[0])
            bin_skel[bi][path[:, 0], path[:, 1]] = True
        else:
            n_per_pixel += 1
            tans = smooth_tangents(path, tw)
            bidx = bin_from_tangent(tans, step, n_bins)
            for bi in range(n_bins):
                sel = (bidx == bi)
                if not np.any(sel):
                    continue
                pts = path[sel]
                bin_skel[bi][pts[:, 0], pts[:, 1]] = True

    # 4) Re-attach junction regions to every bin that touches them. Propagate
    # through the full removed neighborhood so JUNCTION_DILATE_PX > 0 does not
    # leave the interior orphaned.
    junction_region = junc_dil & skel
    if junction_region.any():
        grow_kernel = np.ones((3, 3), np.uint8)
        for bi in range(n_bins):
            grown = bin_skel[bi].copy()
            while True:
                attach = (
                    cv2.dilate(grown.astype(np.uint8), grow_kernel, iterations=1).astype(bool)
                    & junction_region
                )
                new_grown = grown | attach
                if np.array_equal(new_grown, grown):
                    break
                grown = new_grown
            bin_skel[bi] = grown

    # 5) dilate each per-bin skeleton, then clip it back to the source mask
    rad = max(1, int(round(width_info["radius_px"] + float(cfg["BRANCH_DILATE_PAD_PX"]))))
    k_size = 2 * rad + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask_u8 = mask_bool.astype(np.uint8)
    branches = []
    for bi in range(n_bins):
        d = cv2.dilate(bin_skel[bi].astype(np.uint8), kernel, iterations=1)
        d = cv2.bitwise_and(d, mask_u8)
        branches.append((d * 255).astype(np.uint8))

    stats = {
        "filament_width_px": fw,
        "filament_radius_px": float(width_info["radius_px"]),
        "tangent_window_px": tw,
        "branch_dilate_radius_px": rad,
        "binning_mode": binning_mode,
        "hybrid_len_px": hybrid_len,
        "n_skeleton_pixels": int(skel.sum()),
        "n_junction_pixels": int((junc_dil & skel).sum()),
        "n_pieces_total": int(n_lab - 1),
        "n_pieces_kept": n_kept,
        "n_pieces_per_piece_binned": n_per_piece,
        "n_pieces_per_pixel_binned": n_per_pixel,
        "n_pieces_skipped_short": n_skipped_short,
        "angle_bins": [[float(a), float(b)] for (a, b) in bins],
        "ANGLE_STEP_DEG": int(cfg["ANGLE_STEP_DEG"]),
    }
    return branches, bins, width_info, stats


# ============================================================
# IO / pipeline plumbing
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Skeleton-first per-orientation decomposition.")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--output-folder-name", type=str, default=CONFIG["OUTPUT_FOLDER_NAME"])
    ap.add_argument("--bin-threshold", type=int, default=CONFIG["BIN_THRESHOLD"])
    ap.add_argument("--angle-step-deg", type=int, default=CONFIG["ANGLE_STEP_DEG"])
    ap.add_argument("--tangent-window-px", type=int, default=CONFIG["TANGENT_WINDOW_PX"])
    ap.add_argument("--junction-dilate-px", type=int, default=CONFIG["JUNCTION_DILATE_PX"])
    ap.add_argument("--branch-dilate-pad-px", type=int, default=CONFIG["BRANCH_DILATE_PAD_PX"])
    ap.add_argument("--min-piece-len", type=int, default=CONFIG["MIN_PIECE_LEN"])
    ap.add_argument("--binning-mode", choices=["per_pixel", "per_piece", "hybrid"],
                    default=CONFIG["BINNING_MODE"])
    ap.add_argument("--hybrid-len-px", type=int, default=CONFIG["HYBRID_LEN_PX"])
    ap.add_argument("--spur-prune-px", type=int, default=CONFIG["SPUR_PRUNE_PX"])
    # passthrough args from the pipeline (ignored: kept for CLI compat)
    ap.add_argument("--tile-size", type=int, default=None,
                    help="Ignored (skeleton method is tile-free). Accepted for CLI compatibility.")
    ap.add_argument("--tile-grid-offsets", type=str, default=None,
                    help="Ignored. Accepted for CLI compatibility.")
    ap.add_argument("--tile-grid-vote-min", type=int, default=None,
                    help="Ignored. Accepted for CLI compatibility.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)
    cfg["INPUT_PATH"] = args.input
    cfg["OUTPUT_ROOT"] = args.output_root
    cfg["OUTPUT_FOLDER_NAME"] = args.output_folder_name
    cfg["BIN_THRESHOLD"] = int(args.bin_threshold)
    cfg["ANGLE_STEP_DEG"] = int(args.angle_step_deg)
    cfg["TANGENT_WINDOW_PX"] = int(args.tangent_window_px)
    cfg["JUNCTION_DILATE_PX"] = int(args.junction_dilate_px)
    cfg["BRANCH_DILATE_PAD_PX"] = int(args.branch_dilate_pad_px)
    cfg["MIN_PIECE_LEN"] = int(args.min_piece_len)
    cfg["BINNING_MODE"] = str(args.binning_mode)
    cfg["HYBRID_LEN_PX"] = int(args.hybrid_len_px)
    cfg["SPUR_PRUNE_PX"] = int(args.spur_prune_px)

    out_dir = Path(args.output_root) / args.output_folder_name
    branches_dir = out_dir / "branches"
    branches_dir.mkdir(parents=True, exist_ok=True)

    # Read + binarize
    gray = read_gray(args.input)
    mask = binarize(gray, cfg["BIN_THRESHOLD"])

    t0 = time.time()
    branches, bins, width_info, stats = decompose(mask, cfg)
    elapsed = time.time() - t0

    # Write per-branch + merge
    merge = np.zeros_like(mask, dtype=np.uint8)
    for i, b in enumerate(branches, start=1):
        cv2.imwrite(str(branches_dir / f"mask_branch_{i}.png"), b)
        merge = cv2.bitwise_or(merge, b)
    cv2.imwrite(str(branches_dir / "mask_branches_merge.png"), merge)

    # filament_width.json
    (branches_dir / "filament_width.json").write_text(
        json.dumps({
            "filament_width_px": float(width_info["width_px"]),
            "radius_px": float(width_info["radius_px"]),
        }, indent=2),
        encoding="utf-8",
    )

    # original / reconstructed / overlay
    cv2.imwrite(str(out_dir / "original.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(out_dir / "reconstructed.png"), merge)
    overlay = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    overlay_g = np.zeros_like(overlay)
    overlay_g[..., 1] = merge
    overlay = cv2.addWeighted(overlay, 0.45, overlay_g, 0.55, 0)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    # run_config.json (downstream stages parse this)
    run_cfg = {
        "method": "skeleton_decompose",
        "CONFIG": {
            "INPUT_PATH": args.input,
            "OUTPUT_ROOT": args.output_root,
            "OUTPUT_FOLDER_NAME": args.output_folder_name,
            "BIN_THRESHOLD": cfg["BIN_THRESHOLD"],
            "ANGLE_STEP_DEG": cfg["ANGLE_STEP_DEG"],
            "TANGENT_WINDOW_PX": cfg["TANGENT_WINDOW_PX"],
            "JUNCTION_DILATE_PX": cfg["JUNCTION_DILATE_PX"],
            "BRANCH_DILATE_PAD_PX": cfg["BRANCH_DILATE_PAD_PX"],
            "MIN_PIECE_LEN": cfg["MIN_PIECE_LEN"],
            "BINNING_MODE": cfg["BINNING_MODE"],
            "HYBRID_LEN_PX": cfg["HYBRID_LEN_PX"],
            "SPUR_PRUNE_PX": cfg["SPUR_PRUNE_PX"],
            "WIDTH_PERCENTILE": cfg["WIDTH_PERCENTILE"],
        },
        "angle_bins": [[float(a), float(b)] for (a, b) in bins],
        "stats": stats,
        "elapsed_sec": float(elapsed),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"[skeleton] {out_dir}")
    print(f"  filament_width_px = {stats['filament_width_px']:.3f}"
          f"  radius_px = {stats['filament_radius_px']:.3f}"
          f"  tangent_window_px = {stats['tangent_window_px']}")
    print(f"  pieces kept/skipped/total = {stats['n_pieces_kept']}"
          f" / {stats['n_pieces_skipped_short']} / {stats['n_pieces_total']}")
    print(f"  elapsed = {elapsed:.2f}s")


if __name__ == "__main__":
    main()
