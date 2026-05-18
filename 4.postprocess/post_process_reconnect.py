from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage import io as skio
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]

# ── User configuration ────────────────────────────────────────────────────
# Only relevant when running this script directly (not via run_full_sem_pipeline.py).
DEFAULT_INPUT      = ROOT / "output" / "full_pipeline" / "sem_full_00000_1p66_mask255_crop" / "3.reconnect"
DEFAULT_BACKGROUND = ROOT / "input"  / "sem_full_00000_1p66" / "crops" / "sem_full_00000_1p66_overlay_crop.png"
DEFAULT_OUTPUT     = ROOT / "output" / "full_pipeline" / "sem_full_00000_1p66_mask255_crop" / "4.postprocess"

DEFAULT_THICKEN_PX              = 3    # px to thicken each final bundle centerline
DEFAULT_SMOOTH_WINDOW           = 7    # moving-window size for centerline smoothing
DEFAULT_MIN_KEEP_LEN            = 10   # drop bundles whose skeleton is shorter than this (px)
DEFAULT_OVERLAY_ALPHA           = 0.72 # blend weight for the coloured overlay (0 = background only, 1 = labels only)
DEFAULT_SMART_WIDTH             = True # SEM-estimated rendered width also drives cleanup
DEFAULT_SMART_WIDTH_SEARCH_PX   = 16   # normal-ray search radius for SEM edge detection
DEFAULT_SMART_WIDTH_MIN_PX      = 3    # accepted rendered width lower bound
DEFAULT_SMART_WIDTH_MAX_PX      = 24   # accepted rendered width upper bound
DEFAULT_SMART_WIDTH_MIN_EDGE_GRAD = 4.0 # reject weak normal-profile edge responses
DEFAULT_SMART_WIDTH_MAX_SAMPLES = 200  # sampled path cross-sections per component
# ── Overlap absorb / endpoint trim (after thickening) ─────────────────────
DEFAULT_OVERLAP_ABSORB_THR      = 0.0   # 0 = disabled. >=0.5 absorbs near-duplicate IDs into the larger.
DEFAULT_OCCLUSION_TRIM_THR      = 0.0   # 0 = disabled. Trim lower-priority layers mostly hidden by earlier ones.
DEFAULT_OCCLUSION_TRIM_MIN_PX   = 500   # minimum hidden pixels before occlusion trim can trigger
OCCLUSION_FRAGMENT_MIN_PX       = 16    # keep tiny occlusion fragments only when the whole ID is tiny
OCCLUSION_FRAGMENT_MIN_FRAC     = 0.06  # visible fragments below this fraction of the original layer are debris
OCCLUSION_FRAGMENT_ADOPT_MIN_HIDDEN_FRAC = 0.5 # only hand off fragments from mostly occluded layers
OCCLUSION_FRAGMENT_ADOPT_OVERLAP_PX = 3 # hand off split fragments that already share pixels with another layer
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Piece:
    src_id: int
    new_id: int
    mask: np.ndarray
    skel_len: int
    area: int
    dropped: bool = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Post-process reconnect labels into smoother, thicker CNT bundles.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    ap.add_argument("--thicken-px", type=int, default=DEFAULT_THICKEN_PX)
    ap.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--min-keep-len", type=int, default=DEFAULT_MIN_KEEP_LEN)
    ap.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    ap.add_argument("--smart-width", action=argparse.BooleanOptionalAction, default=DEFAULT_SMART_WIDTH,
                    help="Use SEM-guided edge detection for final rendered bundle widths.")
    ap.add_argument("--smart-width-search-px", type=int, default=DEFAULT_SMART_WIDTH_SEARCH_PX,
                    help="Normal-ray search radius used for SEM-guided edge detection.")
    ap.add_argument("--smart-width-min-px", type=int, default=DEFAULT_SMART_WIDTH_MIN_PX,
                    help="Smallest accepted SEM-guided rendered width.")
    ap.add_argument("--smart-width-max-px", type=int, default=DEFAULT_SMART_WIDTH_MAX_PX,
                    help="Largest accepted SEM-guided rendered width.")
    ap.add_argument("--smart-width-min-edge-grad", type=float, default=DEFAULT_SMART_WIDTH_MIN_EDGE_GRAD,
                    help="Minimum per-side SEM edge gradient required for a width sample.")
    ap.add_argument("--smart-width-max-samples", type=int, default=DEFAULT_SMART_WIDTH_MAX_SAMPLES,
                    help="Maximum cross-sections sampled per rendered component.")
    ap.add_argument("--overlap-absorb-thr", type=float, default=DEFAULT_OVERLAP_ABSORB_THR,
                    help="Pairs whose intersection covers >= this fraction of the smaller mask are merged "
                         "into the larger. 0 disables. Try 0.6-0.7.")
    ap.add_argument("--occlusion-trim-thr", type=float, default=DEFAULT_OCCLUSION_TRIM_THR,
                    help="Trim lower-priority rendered layers when this fraction is already covered by "
                         "earlier layers. 0 disables.")
    ap.add_argument("--occlusion-trim-min-px", type=int, default=DEFAULT_OCCLUSION_TRIM_MIN_PX,
                    help="Minimum covered pixels required before --occlusion-trim-thr can trigger.")
    ap.add_argument("--no-copy-source", action="store_true")
    return ap.parse_args()


def find_label_file(folder: Path) -> Path:
    hits = sorted(folder.glob("*_reconnect_labels.tif"))
    hits = [p for p in hits if "dilated" not in p.stem]
    if not hits:
        raise FileNotFoundError(f"No raw reconnect label TIFF found in {folder}")
    return hits[0]


def read_rgb(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = skio.imread(str(path))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.float32)
    mx = float(img.max()) if img.size else 1.0
    if mx > 1.5:
        img /= 255.0 if mx <= 255 else mx
    if img.shape[:2] != shape:
        raise ValueError(f"Background shape {img.shape[:2]} does not match labels {shape}")
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def read_gray(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read background image: {path}")
    if img.shape != shape:
        raise ValueError(f"Background shape {img.shape} does not match labels {shape}")
    return img.astype(np.float32)


def load_layered_components(path: Path, shape: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    data = np.load(str(path))
    saved_shape = tuple(int(v) for v in data["shape"])
    if saved_shape != tuple(shape):
        raise ValueError(f"Layered component shape {saved_shape} does not match labels {shape}")
    ids = data["ids"].astype(np.int64)
    indptr = data["indptr"].astype(np.int64)
    indices = data["indices"].astype(np.int64)
    layers = []
    for i, cid in enumerate(ids):
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        mask[indices[indptr[i]:indptr[i + 1]]] = True
        if np.any(mask):
            layers.append((int(cid), mask.reshape(shape)))
    return layers


def save_layered_components(prefix: Path, layers: list[tuple[int, np.ndarray]]) -> dict:
    layers = [(int(cid), mask.astype(bool)) for cid, mask in layers if mask is not None and np.any(mask)]
    if not layers:
        return {"layered_ids": 0, "overlap_pixels": 0, "max_coverage": 0}

    shape = layers[0][1].shape
    ids, indptr, chunks = [], [0], []
    coverage = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    for cid, mask in layers:
        idx = np.flatnonzero(mask.ravel()).astype(np.uint64)
        ids.append(cid)
        chunks.append(idx)
        indptr.append(indptr[-1] + int(idx.size))
        coverage[idx.astype(np.intp)] += 1

    np.savez_compressed(
        str(prefix.with_name(prefix.name + "_multilabel.npz")),
        shape=np.asarray(shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.uint64),
        indices=np.concatenate(chunks) if chunks else np.array([], dtype=np.uint64),
    )

    import tifffile

    dtype = np.uint16 if max(ids) <= np.iinfo(np.uint16).max else np.uint32
    with tifffile.TiffWriter(str(prefix.with_name(prefix.name + "_multilabel.tif")), bigtiff=True) as tif:
        for cid, mask in layers:
            page = np.zeros(shape, dtype=dtype)
            page[mask] = cid
            tif.write(page, photometric="minisblack")
    prefix.with_name(prefix.name + "_multilabel_ids.json").write_text(
        json.dumps({"page_to_id": ids}, indent=2), encoding="utf-8",
    )
    skio.imsave(
        str(prefix.with_name(prefix.name + "_overlap.png")),
        ((coverage.reshape(shape) > 1).astype(np.uint8) * 255),
        check_contrast=False,
    )
    return {
        "layered_ids": int(len(ids)),
        "overlap_pixels": int(np.count_nonzero(coverage > 1)),
        "max_coverage": int(coverage.max()) if coverage.size else 0,
    }


def endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    pts = np.argwhere((skel > 0) & (nb == 1))
    return [tuple(int(v) for v in p) for p in pts]


def neighbors8(p: tuple[int, int], skel: np.ndarray):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                yield rr, cc


def shortest_path(skel: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    q = deque([start])
    parent = {start: None}
    while q:
        p = q.popleft()
        if p == goal:
            break
        for n in neighbors8(p, skel):
            if n not in parent:
                parent[n] = p
                q.append(n)
    if goal not in parent:
        return np.argwhere(skel > 0).astype(np.float32)
    out = []
    p = goal
    while p is not None:
        out.append(p)
        p = parent[p]
    return np.asarray(out[::-1], dtype=np.float32)


def dominant_path(mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(mask)
    pts = np.argwhere(sk > 0)
    if len(pts) < 3:
        return pts.astype(np.float32)
    eps = endpoints(sk)
    if len(eps) >= 2:
        best = max(((a, b) for i, a in enumerate(eps) for b in eps[i + 1 :]),
                   key=lambda ab: (ab[0][0] - ab[1][0]) ** 2 + (ab[0][1] - ab[1][1]) ** 2)
        return shortest_path(sk, best[0], best[1])
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    axis = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, 1]
    order = np.argsort((xy - xy.mean(0)) @ axis)
    return pts[order].astype(np.float32)


def smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    if len(path) < 3 or window <= 1:
        return path
    w = max(3, int(window) | 1)
    pad = w // 2
    padded = np.pad(path.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w, dtype=np.float32) / float(w)
    out = np.stack([np.convolve(padded[:, i], ker, mode="valid") for i in range(2)], axis=1)
    out[0], out[-1] = path[0], path[-1]
    return out


def render_path(label_img: np.ndarray, path: np.ndarray, label_id: int, thickness: int) -> None:
    if len(path) == 0:
        return
    pts = np.round(path[:, ::-1]).astype(np.int32)  # cv2 wants x,y
    if len(pts) == 1:
        cv2.circle(label_img, tuple(pts[0]), max(1, thickness // 2), int(label_id), -1)
    else:
        cv2.polylines(label_img, [pts.reshape(-1, 1, 2)], False, int(label_id), int(thickness), cv2.LINE_8)


def _sample_indices(n: int, max_samples: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int32)
    edge_skip = min(max(2, n // 20), max(0, (n - 1) // 2))
    start = edge_skip
    stop = max(start, n - 1 - edge_skip)
    count = max(1, min(int(max_samples), stop - start + 1))
    return np.unique(np.linspace(start, stop, count).round().astype(np.int32))


def estimate_sem_guided_width(
    path: np.ndarray,
    sem_gray: np.ndarray,
    *,
    fallback_width: int,
    search_px: int,
    min_width_px: int,
    max_width_px: int,
    min_edge_grad: float,
    max_samples: int,
) -> dict:
    """Estimate one component width from SEM gradients sampled normal to its path."""
    search = max(2, int(search_px))
    min_width = max(1, int(min_width_px))
    max_width = max(min_width, int(max_width_px))
    fallback = int(np.clip(int(fallback_width), min_width, max_width))

    def result(width_px: int, *, used_fallback: bool, n_samples: int, widths: np.ndarray | None = None) -> dict:
        if widths is None or len(widths) == 0:
            return {
                "width_px": int(width_px),
                "used_fallback": used_fallback,
                "n_samples": int(n_samples),
                "n_valid_samples": 0,
                "median_width_px": None,
            }
        median_width = float(np.median(widths))
        return {
            "width_px": int(width_px),
            "used_fallback": used_fallback,
            "n_samples": int(n_samples),
            "n_valid_samples": int(len(widths)),
            "median_width_px": round(median_width, 3),
            "p10_width_px": round(float(np.percentile(widths, 10)), 3),
            "p90_width_px": round(float(np.percentile(widths, 90)), 3),
        }

    if len(path) < 3:
        return result(fallback, used_fallback=True, n_samples=0)

    indices = _sample_indices(len(path), max_samples)
    if len(indices) == 0:
        return result(fallback, used_fallback=True, n_samples=0)

    tangents = []
    centers = []
    for idx in indices:
        lo = max(0, int(idx) - 3)
        hi = min(len(path) - 1, int(idx) + 3)
        delta = path[hi].astype(np.float32) - path[lo].astype(np.float32)
        norm = float(np.hypot(delta[0], delta[1]))
        if norm < 1e-6:
            continue
        tangents.append(delta / norm)
        centers.append(path[int(idx)].astype(np.float32))
    if not centers:
        return result(fallback, used_fallback=True, n_samples=len(indices))

    centers_arr = np.asarray(centers, dtype=np.float32)
    tangents_arr = np.asarray(tangents, dtype=np.float32)
    normals = np.stack([-tangents_arr[:, 1], tangents_arr[:, 0]], axis=1)
    offsets = np.arange(-search, search + 1, dtype=np.float32)
    sample_r = centers_arr[:, 0:1] + normals[:, 0:1] * offsets[None, :]
    sample_c = centers_arr[:, 1:2] + normals[:, 1:2] * offsets[None, :]
    profiles = cv2.remap(
        sem_gray,
        sample_c.astype(np.float32),
        sample_r.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    profiles = cv2.GaussianBlur(profiles.astype(np.float32), (5, 1), 0)
    signed_grads = np.diff(profiles, axis=1)

    center = search
    min_radius = max(1, int(np.floor(min_width / 2)))
    max_radius = min(search - 1, max(1, int(np.ceil(max_width / 2))))
    if max_radius < min_radius:
        return result(fallback, used_fallback=True, n_samples=len(indices))

    left_signed_slice = signed_grads[:, center - max_radius:center - min_radius + 1]
    right_signed_slice = signed_grads[:, center + min_radius - 1:center + max_radius]
    left_slice = np.abs(left_signed_slice)
    right_slice = np.abs(right_signed_slice)
    if left_slice.size == 0 or right_slice.size == 0:
        return result(fallback, used_fallback=True, n_samples=len(indices))

    def best_signed_edge(strength: np.ndarray, signed: np.ndarray, positive: bool) -> tuple[np.ndarray, np.ndarray]:
        allowed = signed > 0 if positive else signed < 0
        masked = np.where(allowed, strength, -1.0)
        argmax = np.argmax(masked, axis=1)
        return argmax, masked[np.arange(len(masked)), argmax]

    # A real bright or dark bundle should enter on one edge and exit on the
    # other, so choose the stronger opposite-slope edge pair on each slice.
    lp_idx, lp_grad = best_signed_edge(left_slice, left_signed_slice, True)
    ln_idx, ln_grad = best_signed_edge(left_slice, left_signed_slice, False)
    rp_idx, rp_grad = best_signed_edge(right_slice, right_signed_slice, True)
    rn_idx, rn_grad = best_signed_edge(right_slice, right_signed_slice, False)
    use_pos_then_neg = (lp_grad + rn_grad) >= (ln_grad + rp_grad)
    left_argmax = np.where(use_pos_then_neg, lp_idx, ln_idx)
    right_argmax = np.where(use_pos_then_neg, rn_idx, rp_idx)
    left_grad = np.where(use_pos_then_neg, lp_grad, ln_grad)
    right_grad = np.where(use_pos_then_neg, rn_grad, rp_grad)
    left_radius = max_radius - left_argmax
    right_radius = min_radius + right_argmax
    widths = left_radius + right_radius
    valid = (
        (left_grad >= float(min_edge_grad))
        & (right_grad >= float(min_edge_grad))
        & (widths >= min_width)
        & (widths <= max_width)
    )
    valid_widths = widths[valid].astype(np.float32)
    min_valid = max(5, int(np.ceil(len(widths) * 0.15)))
    if len(valid_widths) < min_valid:
        return result(fallback, used_fallback=True, n_samples=len(widths))

    q1, q3 = np.percentile(valid_widths, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 0:
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        trimmed = valid_widths[(valid_widths >= lo) & (valid_widths <= hi)]
        if len(trimmed) >= min_valid:
            valid_widths = trimmed
    median_width = float(np.median(valid_widths))
    width_px = int(np.clip(int(round(median_width)), min_width, max_width))
    return result(width_px, used_fallback=False, n_samples=len(widths), widths=valid_widths)


def pieces_from_labels(lbl: np.ndarray) -> list[Piece]:
    pieces: list[Piece] = []
    nid = 1
    for src_id in [int(v) for v in np.unique(lbl) if v]:
        m = lbl == src_id
        if not np.any(m):
            continue
        pieces.append(Piece(src_id, nid, m, int(np.count_nonzero(skeletonize(m))), int(m.sum())))
        nid += 1
    return pieces


def pieces_from_layers(layers: list[tuple[int, np.ndarray]]) -> list[Piece]:
    pieces: list[Piece] = []
    nid = 1
    for src_id, layer_mask in layers:
        if not np.any(layer_mask):
            continue
        pieces.append(Piece(src_id, nid, layer_mask, int(np.count_nonzero(skeletonize(layer_mask))), int(layer_mask.sum())))
        nid += 1
    return pieces


def drop_short_pieces(pieces: list[Piece], min_keep_len: int) -> None:
    for p in pieces:
        if p.skel_len < min_keep_len:
            p.dropped = True


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 0, 0, 0
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r0, r1, c0, c1


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def absorb_overlapping_layers(
    layers: list[tuple[int, np.ndarray]], thr: float
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, int, float]]]:
    """Merge pairs whose intersection covers >= thr of the smaller mask.

    Operates on rendered per-piece masks, where duplicate-bundle overlap
    actually appears -- reconnect's component-level overlap-kill cannot see
    this because it runs before postprocess rendering.
    """
    if thr <= 0 or len(layers) < 2:
        return layers, []
    work = sorted(layers, key=lambda x: -int(x[1].sum()))
    n = len(work)
    bboxes = [_bbox(m) for _, m in work]
    areas = [int(m.sum()) for _, m in work]
    dropped = [False] * n
    log: list[tuple[int, int, float]] = []   # (absorbed_id, into_id, ratio)
    for i in range(n):
        if dropped[i]:
            continue
        for j in range(i + 1, n):
            if dropped[j]:
                continue
            if not _bbox_intersects(bboxes[i], bboxes[j]):
                continue
            inter = int(np.logical_and(work[i][1], work[j][1]).sum())
            if inter == 0:
                continue
            ratio = inter / max(1, min(areas[i], areas[j]))   # vs the smaller
            if ratio < thr:
                continue
            # j is smaller (sorted descending). Absorb j into i.
            merged = np.logical_or(work[i][1], work[j][1])
            work[i] = (work[i][0], merged)
            bboxes[i] = _bbox(merged)
            areas[i] = int(merged.sum())
            dropped[j] = True
            log.append((work[j][0], work[i][0], ratio))
    out = [work[k] for k in range(n) if not dropped[k]]
    return out, log


def trim_occluded_layers(
    layers: list[tuple[int, np.ndarray]],
    thr: float,
    min_px: int,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, int, float]], int]:
    """Trim lower-priority layer pixels already covered by earlier layers."""
    if thr <= 0 or len(layers) < 2:
        return layers, [], 0
    out: list[tuple[int, np.ndarray]] = []
    log: list[tuple[int, int, float]] = []  # (id, hidden_px, hidden_ratio)
    fragment_removed = 0
    split_trimmed_ids: set[int] = set()
    for cid, mask in layers:
        area = int(mask.sum())
        if area <= 0:
            continue
        hidden = np.zeros_like(mask, dtype=bool)
        for _, prev_mask in out:
            inter = np.logical_and(mask, prev_mask)
            inter_px = int(inter.sum())
            if inter_px >= int(min_px) and inter_px / max(1, area) >= float(thr):
                hidden |= inter
        hidden_px = int(hidden.sum())
        hidden_ratio = hidden_px / max(1, area)
        if hidden_px > 0:
            mask = np.logical_and(mask, ~hidden)
            if not np.any(mask):
                log.append((int(cid), hidden_px, hidden_ratio))
                continue
            cc = cc_label(mask, connectivity=2)
            if int(cc.max()) > 1:
                min_fragment_area = max(
                    OCCLUSION_FRAGMENT_MIN_PX,
                    int(round(area * OCCLUSION_FRAGMENT_MIN_FRAC)),
                )
                kept = [cc == idx for idx in range(1, int(cc.max()) + 1)
                        if int(np.count_nonzero(cc == idx)) >= min_fragment_area]
                if kept:
                    clean_mask = np.logical_or.reduce(kept)
                    fragment_removed += int(mask.sum()) - int(clean_mask.sum())
                    mask = clean_mask
                    if len(kept) > 1 and hidden_ratio >= OCCLUSION_FRAGMENT_ADOPT_MIN_HIDDEN_FRAC:
                        split_trimmed_ids.add(int(cid))
            log.append((int(cid), hidden_px, hidden_ratio))
        out.append((int(cid), mask))

    if split_trimmed_ids and len(out) > 1:
        work = [(cid, mask.copy()) for cid, mask in out]
        for i, (cid, mask) in enumerate(work):
            if cid not in split_trimmed_ids:
                continue
            cc = cc_label(mask, connectivity=2)
            if int(cc.max()) < 2:
                continue
            keep_mask = np.zeros_like(mask, dtype=bool)
            for idx in range(1, int(cc.max()) + 1):
                frag = cc == idx
                best_j = -1
                best_overlap = 0
                for j, (other_id, other_mask) in enumerate(work):
                    if i == j or other_id == cid:
                        continue
                    overlap_px = int(np.count_nonzero(frag & other_mask))
                    if overlap_px > best_overlap:
                        best_overlap = overlap_px
                        best_j = j
                if best_j >= 0 and best_overlap >= OCCLUSION_FRAGMENT_ADOPT_OVERLAP_PX:
                    work[best_j] = (work[best_j][0], np.logical_or(work[best_j][1], frag))
                else:
                    keep_mask |= frag
            work[i] = (cid, keep_mask)
        out = [(cid, mask) for cid, mask in work if np.any(mask)]
    return out, log, fragment_removed


def colorize(lbl: np.ndarray) -> np.ndarray:
    lab = lbl.astype(np.int64)
    out = np.zeros((*lab.shape, 3), dtype=np.uint8)
    out[..., 0] = (lab * 53) % 256
    out[..., 1] = (lab * 97) % 256
    out[..., 2] = (lab * 193) % 256
    out[lab == 0] = 0
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_source:
        raw_dir = args.output / "raw_reconnect"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        shutil.copytree(args.input, raw_dir)

    src_label = find_label_file(args.input)
    lbl = skio.imread(str(src_label)).astype(np.int32)
    base = src_label.stem.replace("_reconnect_labels", "")
    layered_src = args.input / f"{base}_reconnect_multilabel.npz"
    if layered_src.exists():
        pieces = pieces_from_layers(load_layered_components(layered_src, lbl.shape))
    else:
        pieces = pieces_from_labels(lbl)
    drop_short_pieces(pieces, args.min_keep_len)

    out = np.zeros(lbl.shape, dtype=np.uint16)
    sem_gray = read_gray(args.background, out.shape) if args.smart_width and args.background.exists() else None
    layered_out: list[tuple[int, np.ndarray]] = []
    smart_width_log: list[dict] = []
    kept = [p for p in pieces if not p.dropped and p.skel_len >= args.min_keep_len]
    kept.sort(key=lambda p: (-p.skel_len, -p.area, p.new_id))
    for out_id, p in enumerate(kept, start=1):
        tmp = np.zeros_like(out)
        path = smooth_path(dominant_path(p.mask), args.smooth_window)
        if args.smart_width and sem_gray is not None:
            width_info = estimate_sem_guided_width(
                path,
                sem_gray,
                fallback_width=int(args.thicken_px),
                search_px=int(args.smart_width_search_px),
                min_width_px=int(args.smart_width_min_px),
                max_width_px=int(args.smart_width_max_px),
                min_edge_grad=float(args.smart_width_min_edge_grad),
                max_samples=int(args.smart_width_max_samples),
            )
        else:
            width_info = {
                "width_px": int(args.thicken_px),
                "used_fallback": True,
                "n_samples": 0,
                "n_valid_samples": 0,
                "median_width_px": None,
            }
        render_path(tmp, path, out_id, int(width_info["width_px"]))
        layer = tmp > 0
        if np.any(layer):
            layered_out.append((out_id, layer.copy()))
        smart_width_log.append(
            {
                "id": int(out_id),
                "source_area_px": int(p.mask.sum()),
                **width_info,
                "rendered_area_px": int(layer.sum()),
            }
        )

    # Cleanup passes run on the rendered layers. When smart width is enabled,
    # its SEM-guided masks intentionally affect absorb and occlusion decisions.
    overlap_absorb_log: list[tuple[int, int, float]] = []
    if args.overlap_absorb_thr > 0:
        layered_out, overlap_absorb_log = absorb_overlapping_layers(
            layered_out,
            float(args.overlap_absorb_thr),
        )
    occlusion_trim_log: list[tuple[int, int, float]] = []
    if args.occlusion_trim_thr > 0:
        layered_out, occlusion_trim_log, occlusion_fragment_removed = trim_occluded_layers(
            layered_out,
            float(args.occlusion_trim_thr),
            int(args.occlusion_trim_min_px),
        )
    else:
        occlusion_fragment_removed = 0

    # Flatten layered_out into the final label image. Layers are already in
    # priority order (longest first); first-write-wins preserves "longer-bundle
    # owns the disputed pixel".
    out[:] = 0
    for cid, layer in layered_out:
        out[np.logical_and(layer, out == 0)] = cid

    layered_summary = save_layered_components(args.output / f"{base}_post", layered_out)
    skio.imsave(str(args.output / f"{base}_post_labels.tif"), out.astype(np.uint16), check_contrast=False)
    skio.imsave(str(args.output / f"{base}_post_labels_preview.png"), colorize(out), check_contrast=False)
    if args.background.exists():
        bg = read_rgb(args.background, out.shape)
        color = colorize(out)
        overlay = bg.copy()
        m = out > 0
        overlay[m] = (args.overlay_alpha * color[m] + (1.0 - args.overlay_alpha) * bg[m]).astype(np.uint8)
        skio.imsave(str(args.output / f"{base}_post_overlay.png"), overlay, check_contrast=False)

    summary = {
        "source": str(src_label),
        "source_multilabel": str(layered_src) if layered_src.exists() else None,
        "raw_labels": int(len([v for v in np.unique(lbl) if v])),
        "input_pieces": len(pieces),
        "kept_labels": int(len([v for v in np.unique(out) if v])),
        "max_label_id": int(out.max()),
        "dropped_pieces": int(sum(p.dropped for p in pieces)),
        "thicken_px": int(args.thicken_px),
        "render_width_mode": "sem_edge" if args.smart_width and sem_gray is not None else "fixed",
        "smart_width_enabled": bool(args.smart_width),
        "smart_width_search_px": int(args.smart_width_search_px),
        "smart_width_min_px": int(args.smart_width_min_px),
        "smart_width_max_px": int(args.smart_width_max_px),
        "smart_width_min_edge_grad": float(args.smart_width_min_edge_grad),
        "smart_width_estimates": smart_width_log,
        "smooth_window": int(args.smooth_window),
        "overlap_absorb_thr": float(args.overlap_absorb_thr),
        "overlap_absorbed_pairs": [
            {"absorbed_id": int(a), "into_id": int(b), "overlap_ratio": round(float(r), 4)}
            for a, b, r in overlap_absorb_log
        ],
        "overlap_absorbed_count": len(overlap_absorb_log),
        "occlusion_trim_thr": float(args.occlusion_trim_thr),
        "occlusion_trim_min_px": int(args.occlusion_trim_min_px),
        "occlusion_fragment_removed_pixels": int(occlusion_fragment_removed),
        "occlusion_trimmed_layers": [
            {"id": int(cid), "hidden_px": int(px), "hidden_ratio": round(float(r), 4)}
            for cid, px, r in occlusion_trim_log
        ],
        "occlusion_trimmed_count": len(occlusion_trim_log),
        **layered_summary,
    }
    (args.output / "post_process_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
