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
# ── Overlap absorb / endpoint trim (after thickening) ─────────────────────
DEFAULT_OVERLAP_ABSORB_THR      = 0.0   # 0 = disabled. >=0.5 absorbs near-duplicate IDs into the larger.
DEFAULT_OCCLUSION_TRIM_THR      = 0.0   # 0 = disabled. Trim lower-priority layers mostly hidden by earlier ones.
DEFAULT_OCCLUSION_TRIM_MIN_PX   = 500   # minimum hidden pixels before occlusion trim can trigger
DEFAULT_TIP_TRIM_FRAC           = 0.0   # 0 = disabled. ~0.15 trims overlap pixels at an ID's skeleton tip.
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
    ap.add_argument("--overlap-absorb-thr", type=float, default=DEFAULT_OVERLAP_ABSORB_THR,
                    help="Pairs whose intersection covers >= this fraction of the smaller mask are merged "
                         "into the larger. 0 disables. Try 0.6-0.7.")
    ap.add_argument("--occlusion-trim-thr", type=float, default=DEFAULT_OCCLUSION_TRIM_THR,
                    help="Trim lower-priority rendered layers when this fraction is already covered by "
                         "earlier layers. 0 disables.")
    ap.add_argument("--occlusion-trim-min-px", type=int, default=DEFAULT_OCCLUSION_TRIM_MIN_PX,
                    help="Minimum covered pixels required before --occlusion-trim-thr can trigger.")
    ap.add_argument("--tip-trim-frac", type=float, default=DEFAULT_TIP_TRIM_FRAC,
                    help="If overlap pixels of A∩B sit within this fraction of A's skeleton-tip distance, "
                         "erase them from A so it 'merges into' B without redundant overlap. 0 disables. "
                         "Try 0.15.")
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

    Operates on rendered (thickened) per-piece masks, where duplicate-bundle
    overlap actually appears -- reconnect's component-level overlap-kill cannot
    see this because it runs before postprocess thickening.
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
    covered = np.zeros(layers[0][1].shape, dtype=bool)
    log: list[tuple[int, int, float]] = []  # (id, hidden_px, hidden_ratio)
    fragment_removed = 0
    split_trimmed_ids: set[int] = set()
    for cid, mask in layers:
        area = int(mask.sum())
        if area <= 0:
            continue
        hidden = np.logical_and(mask, covered)
        hidden_px = int(hidden.sum())
        hidden_ratio = hidden_px / max(1, area)
        if hidden_px >= int(min_px) and hidden_ratio >= float(thr):
            mask = np.logical_and(mask, ~covered)
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
        covered |= mask

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


def trim_endpoint_overlaps(
    layers: list[tuple[int, np.ndarray]], tip_frac: float
) -> tuple[list[tuple[int, np.ndarray]], int]:
    """For overlap pixels of A∩B that sit within tip_frac of A's skeleton tip,
    erase them from A's mask so A 'merges into' B without redundant overlap.

    Mid-bundle overlaps (legitimate crossings) are preserved -- only overlap
    pixels in the tip neighbourhood are removed.
    """
    if tip_frac <= 0 or len(layers) < 2:
        return layers, 0
    masks = [m.copy() for _, m in layers]
    ids = [cid for cid, _ in layers]
    bboxes = [_bbox(m) for m in masks]
    n = len(masks)

    # Per-id tip-distance map and tip neighbourhood radius
    tip_info: list[tuple[np.ndarray, float] | None] = []
    for m in masks:
        sk = skeletonize(m)
        if int(sk.sum()) < 4:
            tip_info.append(None)
            continue
        nb = convolve(sk.astype(np.uint8),
                      np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
                      mode="constant")
        eps = np.argwhere(sk & (nb == 1))
        if len(eps) < 1:
            tip_info.append(None)
            continue
        seed = np.zeros(m.shape, dtype=bool)
        seed[eps[:, 0], eps[:, 1]] = True
        d = distance_transform_edt(~seed)
        skel_len = int(sk.sum())
        radius = max(8.0, float(tip_frac) * float(skel_len))
        tip_info.append((d, radius))

    trimmed_total = 0
    for i in range(n):
        info_i = tip_info[i]
        if info_i is None:
            continue
        d_i, r_i = info_i
        for j in range(n):
            if i == j:
                continue
            if not _bbox_intersects(bboxes[i], bboxes[j]):
                continue
            inter = np.logical_and(masks[i], masks[j])
            if not inter.any():
                continue
            erase = inter & (d_i <= r_i)
            if not erase.any():
                continue
            # Only trim if the overlap is endpoint-dominant for A (>=50% near A's tip)
            if int(erase.sum()) * 2 < int(inter.sum()):
                continue
            masks[i] = masks[i] & ~erase
            bboxes[i] = _bbox(masks[i])
            trimmed_total += int(erase.sum())
    out = [(ids[k], masks[k]) for k in range(n) if masks[k].any()]
    return out, trimmed_total


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
    layered_out: list[tuple[int, np.ndarray]] = []
    kept = [p for p in pieces if not p.dropped and p.skel_len >= args.min_keep_len]
    kept.sort(key=lambda p: (-p.skel_len, -p.area, p.new_id))
    for out_id, p in enumerate(kept, start=1):
        tmp = np.zeros_like(out)
        for submask in [p.mask]:
            path = smooth_path(dominant_path(submask), args.smooth_window)
            render_path(tmp, path, out_id, args.thicken_px)
        layer = tmp > 0
        if np.any(layer):
            layered_out.append((out_id, layer.copy()))

    # Optional cleanup passes operating on the rendered (thickened) layers.
    # Order matters: absorption first (eliminates near-duplicate IDs entirely),
    # then endpoint trim (cleans residual tip overlaps where small ID merges into big).
    overlap_absorb_log: list[tuple[int, int, float]] = []
    if args.overlap_absorb_thr > 0:
        layered_out, overlap_absorb_log = absorb_overlapping_layers(
            layered_out,
            float(args.overlap_absorb_thr),
        )
    tip_trim_pixels = 0
    if args.tip_trim_frac > 0:
        layered_out, tip_trim_pixels = trim_endpoint_overlaps(layered_out, float(args.tip_trim_frac))
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
        "smooth_window": int(args.smooth_window),
        "overlap_absorb_thr": float(args.overlap_absorb_thr),
        "overlap_absorbed_pairs": [
            {"absorbed_id": int(a), "into_id": int(b), "overlap_ratio": round(float(r), 4)}
            for a, b, r in overlap_absorb_log
        ],
        "overlap_absorbed_count": len(overlap_absorb_log),
        "tip_trim_frac": float(args.tip_trim_frac),
        "tip_trim_pixels": int(tip_trim_pixels),
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
