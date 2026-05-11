from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import convolve as _conv
from skimage.morphology import skeletonize as _skeletonize


ROOT = Path(__file__).resolve().parents[1]

# ── User configuration ────────────────────────────────────────────────────
# Only relevant when running this script directly (not via run_full_sem_pipeline.py).
DEFAULT_INPUT  = ROOT / "1.stringart" / "output" / "sem_full_00000_1p66_mask255_crop" / "branches"
DEFAULT_OUTPUT = ROOT / "2.preprocess" / "output" / "sem_full_00000_1p66_mask255_crop" / "branches"

DEFAULT_BIN_THRESHOLD      = 127  # grayscale threshold for binarising each branch mask
DEFAULT_LINE_CLOSE_LEN     = 9    # length of the oriented closing kernel (px); bridges small raster gaps
DEFAULT_LINE_CLOSE_ITERS   = 1    # how many times to apply the closing
DEFAULT_MIN_COMPONENT_AREA = 8    # remove connected components smaller than this (px)
DEFAULT_CLEAN_TO_PATH      = True # reduce each component to its smoothed dominant skeleton path
DEFAULT_CLEAN_SMOOTH_WIN   = 7    # moving-average window applied to the dominant path before redrawing
DEFAULT_TARGET_WIDTH_PX    = 0    # 0 = use per-branch median width; >0 = render every component at this width
DEFAULT_FIT_DEGREE         = 2    # 0=skip spline fit (moving-avg only); 1/2/3=parametric B-spline degree
DEFAULT_FIT_SMOOTHING      = 1.5  # spline smoothing factor multiplier (higher = stiffer, more polynomial-like)
# ─────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean stringart branch masks before reconnect.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--run-config", type=Path, default=None, help="Defaults to ../run_config.json from input.")
    ap.add_argument("--bin-threshold", type=int, default=DEFAULT_BIN_THRESHOLD)
    ap.add_argument("--line-close-len", type=int, default=DEFAULT_LINE_CLOSE_LEN)
    ap.add_argument("--line-close-iters", type=int, default=DEFAULT_LINE_CLOSE_ITERS)
    ap.add_argument("--min-component-area", type=int, default=DEFAULT_MIN_COMPONENT_AREA)
    ap.add_argument("--clean-to-path",    action=argparse.BooleanOptionalAction, default=DEFAULT_CLEAN_TO_PATH,
                    help="Reduce every component to its smoothed dominant skeleton path + uniform width.")
    ap.add_argument("--clean-smooth-win", type=int, default=DEFAULT_CLEAN_SMOOTH_WIN,
                    help="Smoothing window for dominant path before redrawing (1 = off).")
    ap.add_argument("--target-width-px", type=int, default=DEFAULT_TARGET_WIDTH_PX,
                    help="Target re-render width for all components. 0 = use per-branch median width.")
    ap.add_argument("--fit-degree", type=int, default=DEFAULT_FIT_DEGREE,
                    help="Parametric B-spline degree for skeleton smoothing. 0=skip, 1=linear, "
                         "2=quadratic (default), 3=cubic. Forces each path to a physically meaningful curve.")
    ap.add_argument("--fit-smoothing", type=float, default=DEFAULT_FIT_SMOOTHING,
                    help="Spline smoothing factor multiplier (higher = stiffer / more polynomial-like).")
    ap.add_argument("--copy-source", action="store_true", help="Copy raw branch inputs beside output.")
    return ap.parse_args()


def bool255(img: np.ndarray) -> np.ndarray:
    return (img > 0).astype(np.uint8) * 255


def read_bin(path: Path, thr: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return bool255(cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)[1])


def branch_index(path: Path) -> int | None:
    m = re.search(r"_branch_(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def branch_paths(folder: Path) -> list[Path]:
    paths = []
    for p in folder.glob("*_branch_*.png"):
        if branch_index(p) is not None:
            paths.append(p)
    return sorted(paths, key=lambda p: branch_index(p) or 0)


def load_angle_centers(input_dir: Path, run_config: Path | None, n: int) -> list[float]:
    cfg_path = run_config or (input_dir.parent / "run_config.json")
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        bins = data.get("angle_bins", [])
        if len(bins) >= n:
            return [0.5 * (float(lo) + float(hi)) for lo, hi in bins[:n]]
        step = float(data.get("CONFIG", {}).get("ANGLE_STEP_DEG", 180.0 / max(1, n)))
    else:
        step = 180.0 / max(1, n)
    return [(i + 0.5) * step for i in range(n)]


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(float(angle_deg))
    dx = np.cos(rad) * c
    dy = np.sin(rad) * c
    p0 = (int(round(c - dx)), int(round(c - dy)))
    p1 = (int(round(c + dx)), int(round(c + dy)))
    cv2.line(k, p0, p1, 1, 1, cv2.LINE_8)
    return k


def remove_small_components(mask255: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 1:
        return mask255
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask255 > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask255)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 255
    return out


def _skel_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = _conv(skel.astype(np.uint8), k, mode="constant", cval=0)
    return [tuple(int(v) for v in p) for p in np.argwhere((skel > 0) & (nb == 1))]


def _bfs_path(skel: np.ndarray, start: tuple, goal: tuple) -> np.ndarray:
    q = deque([start])
    parent: dict = {start: None}
    while q:
        p = q.popleft()
        if p == goal:
            break
        r, c = p
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if (0 <= nb[0] < skel.shape[0] and 0 <= nb[1] < skel.shape[1]
                        and skel[nb] and nb not in parent):
                    parent[nb] = p
                    q.append(nb)
    if goal not in parent:
        return np.argwhere(skel > 0)
    path: list = []
    p = goal
    while p is not None:
        path.append(p)
        p = parent[p]
    return np.array(path[::-1], dtype=np.float32)


def _smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)
    if len(path) < w:
        return path
    pad = w // 2
    padded = np.pad(path.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w, dtype=np.float32) / w
    out = np.stack([np.convolve(padded[:, i], ker, "valid") for i in range(2)], axis=1)
    out[0] = path[0]; out[-1] = path[-1]
    return out


def _fit_spline(path: np.ndarray, degree: int, smoothing: float) -> np.ndarray:
    """Fit a parametric B-spline of given degree through path points.

    Forces the centerline onto a smooth curve up to the specified polynomial
    order (1=line, 2=quadratic, 3=cubic). Sampled densely along the spline
    arc, preserving the original endpoints exactly. Falls back to the input
    path if there aren't enough unique points for the requested degree.
    """
    if degree < 1 or len(path) < 2:
        return path
    pts = path.astype(np.float64)
    # de-dup near-coincident points; splprep requires strictly increasing param
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.5]
    pts = pts[keep]
    k = min(int(degree), max(1, len(pts) - 1))
    if len(pts) < k + 1:
        return path
    s = float(smoothing) * len(pts)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, k=k)
    except Exception:
        return path
    n_out = max(16, min(len(pts) * 2, 240))
    u = np.linspace(0.0, 1.0, n_out)
    fit = np.stack(splev(u, tck), axis=1).astype(np.float32)
    fit[0]  = path[0]
    fit[-1] = path[-1]
    return fit


def _dominant_path(skel: np.ndarray) -> np.ndarray | None:
    """Return a 2-tip path through the skeleton (most-distant endpoint pair),
    falling back to a PCA axis ordering for ring-like components with no
    endpoints. Returns None if the component is too small to path-ify."""
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 3:
        return None
    eps = _skel_endpoints(skel)
    if len(eps) >= 2:
        best_a, best_b, best_d2 = eps[0], eps[1], 0
        for i, a in enumerate(eps):
            for b in eps[i + 1:]:
                d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d2 > best_d2:
                    best_d2, best_a, best_b = d2, a, b
        path = _bfs_path(skel, best_a, best_b)
        return path if len(path) >= 2 else None
    # 0 or 1 endpoint — order by principal axis projection (rare ring-like case)
    xy = np.stack([skel_pts[:, 1], skel_pts[:, 0]], axis=1).astype(np.float32)
    centred = xy - xy.mean(0)
    axis = np.linalg.eigh(np.cov(centred.T))[1][:, 1]
    order = np.argsort(centred @ axis)
    return skel_pts[order].astype(np.float32)


def smooth_and_normalize(
    mask255: np.ndarray,
    smooth_window: int,
    target_width_px: int,
    fit_degree: int = 0,
    fit_smoothing: float = 1.5,
) -> np.ndarray:
    """Smooth every connected component and re-render at a uniform width.

    Pipeline per CC:
      skeletonize → dominant path (handles ring-like via PCA)
                  → moving-average smooth (if smooth_window > 1)
                  → parametric B-spline fit at fit_degree (if fit_degree > 0)
                  → record area/length width
    Then re-render all paths at a single target width:
      - target_width_px == 0  →  per-branch median of measured widths
      - target_width_px  > 0  →  fixed render width
    Result: each component is a physically-meaningful smooth curve of uniform
    width, making downstream tip-matching and overlap detection much cleaner.
    """
    n, labels = cv2.connectedComponents((mask255 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask255

    items: list[tuple[np.ndarray | None, int]] = []
    for i in range(1, n):
        comp = (labels == i)
        skel = _skeletonize(comp).astype(np.uint8)
        skel_len = int(skel.sum())
        path = _dominant_path(skel)
        if path is None or len(path) < 2 or skel_len < 1:
            items.append((None, 0))
            continue
        if smooth_window > 1:
            path = _smooth_path(path, smooth_window)
        if fit_degree > 0:
            path = _fit_spline(path, fit_degree, fit_smoothing)
        width = max(1, int(round(int(comp.sum()) / max(1, skel_len))))
        items.append((path, width))

    widths = [w for _, w in items if w > 0]
    if not widths:
        return mask255
    if target_width_px > 0:
        target_w = int(target_width_px)
    else:
        target_w = int(round(float(np.median(widths))))
    target_w = max(1, target_w)

    out = np.zeros_like(mask255)
    for path, w in items:
        if path is None:
            continue
        pts_xy = np.round(path[:, ::-1]).astype(np.int32)
        cv2.polylines(out, [pts_xy.reshape(-1, 1, 2)], False, 255, target_w, cv2.LINE_8)
    return out


def process_branch(mask255: np.ndarray, angle_deg: float, args: argparse.Namespace) -> np.ndarray:
    out = remove_small_components(mask255, args.min_component_area)
    if args.line_close_len > 1 and args.line_close_iters > 0:
        ker = line_kernel(args.line_close_len, angle_deg)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=int(args.line_close_iters))
    out = remove_small_components(out, args.min_component_area)
    if args.clean_to_path:
        out = smooth_and_normalize(out, args.clean_smooth_win, args.target_width_px,
                                   args.fit_degree, args.fit_smoothing)
    return bool255(out)


def base_stem_from_paths(paths: list[Path]) -> str:
    stem = paths[0].stem
    return re.sub(r"_branch_\d+$", "", stem)


def main() -> None:
    args = parse_args()
    paths = branch_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No *_branch_*.png files found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.copy_source:
        raw = args.output.parent / "raw_branches"
        if raw.exists():
            shutil.rmtree(raw)
        shutil.copytree(args.input, raw)

    centers = load_angle_centers(args.input, args.run_config, len(paths))
    merge = None
    per_branch = []
    for p, angle in zip(paths, centers):
        cleaned = process_branch(read_bin(p, args.bin_threshold), angle, args)
        out_path = args.output / p.name
        cv2.imwrite(str(out_path), cleaned)
        merge = cleaned if merge is None else cv2.bitwise_or(merge, cleaned)
        per_branch.append({
            "file": p.name,
            "angle_center_deg": angle,
            "input_px": int(cv2.countNonZero(read_bin(p, args.bin_threshold))),
            "output_px": int(cv2.countNonZero(cleaned)),
        })

    base = base_stem_from_paths(paths)
    cv2.imwrite(str(args.output / f"{base}_branches_merge.png"), merge)
    for name in ("filament_width.json",):
        src = args.input / name
        if src.exists():
            shutil.copy2(src, args.output / name)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "branches": len(paths),
        "line_close_len": int(args.line_close_len),
        "line_close_iters": int(args.line_close_iters),
        "min_component_area": int(args.min_component_area),
        "input_merge_px": int(sum(x["input_px"] for x in per_branch)),
        "output_merge_px": int(cv2.countNonZero(merge)),
        "per_branch": per_branch,
    }
    (args.output.parent / "pre_process_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_branch"}, indent=2))


if __name__ == "__main__":
    main()
