from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, convolve
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
DEFAULT_ABSORB_LEN              = 28   # absorb bundles shorter than this into a longer neighbour
DEFAULT_ABSORB_RADIUS           = 5    # halo radius (px) for neighbour detection during absorb
DEFAULT_SAME_SOURCE_BRIDGE_RADIUS = 12 # merge disconnected pieces sharing the same source label within this radius
DEFAULT_OVERLAY_ALPHA           = 0.72 # blend weight for the coloured overlay (0 = background only, 1 = labels only)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Piece:
    src_id: int
    new_id: int
    mask: np.ndarray
    skel_len: int
    area: int
    absorbed_into: int | None = None
    same_source_into: int | None = None
    dropped: bool = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Post-process reconnect labels into smoother, thicker CNT bundles.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    ap.add_argument("--thicken-px", type=int, default=DEFAULT_THICKEN_PX)
    ap.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--min-keep-len", type=int, default=DEFAULT_MIN_KEEP_LEN)
    ap.add_argument("--absorb-len", type=int, default=DEFAULT_ABSORB_LEN)
    ap.add_argument("--absorb-radius", type=int, default=DEFAULT_ABSORB_RADIUS)
    ap.add_argument("--same-source-bridge-radius", type=int, default=DEFAULT_SAME_SOURCE_BRIDGE_RADIUS)
    ap.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
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



def split_pieces(lbl: np.ndarray) -> list[Piece]:
    pieces: list[Piece] = []
    nid = 1
    for src_id in [int(v) for v in np.unique(lbl) if v]:
        cc = cc_label(lbl == src_id, connectivity=2)
        for k in range(1, int(cc.max()) + 1):
            m = cc == k
            if not np.any(m):
                continue
            pieces.append(Piece(src_id, nid, m, int(np.count_nonzero(skeletonize(m))), int(m.sum())))
            nid += 1
    return pieces


def split_pieces_from_layers(layers: list[tuple[int, np.ndarray]]) -> list[Piece]:
    pieces: list[Piece] = []
    nid = 1
    for src_id, layer_mask in layers:
        cc = cc_label(layer_mask, connectivity=2)
        for k in range(1, int(cc.max()) + 1):
            m = cc == k
            if not np.any(m):
                continue
            pieces.append(Piece(src_id, nid, m, int(np.count_nonzero(skeletonize(m))), int(m.sum())))
            nid += 1
    return pieces


def bridge_points(mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(mask)
    eps = endpoints(sk)
    pts = np.asarray(eps, dtype=np.int32) if eps else np.argwhere(sk > 0).astype(np.int32)
    if len(pts) == 0:
        pts = np.argwhere(mask).astype(np.int32)
    return pts


def closest_points(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    pa, pb = bridge_points(a), bridge_points(b)
    if len(pa) == 0 or len(pb) == 0:
        return float("inf"), np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32)
    d2 = np.sum((pa[:, None, :] - pb[None, :, :]) ** 2, axis=2)
    ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return float(np.sqrt(d2[ia, ib])), pa[ia], pb[ib]


def line_bridge(shape: tuple[int, int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    cv2.line(out, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, 1, cv2.LINE_8)
    return out > 0


def refresh_piece(piece: Piece) -> None:
    piece.area = int(piece.mask.sum())
    piece.skel_len = int(np.count_nonzero(skeletonize(piece.mask)))


def bridge_same_source_pieces(pieces: list[Piece], radius: int) -> int:
    if radius <= 0:
        return 0
    merged = 0
    for src_id in sorted({p.src_id for p in pieces}):
        while True:
            active = [p for p in pieces if p.src_id == src_id and not p.dropped]
            if len(active) < 2:
                break
            best = None
            for i, p in enumerate(active):
                for q in active[i + 1:]:
                    dist, a, b = closest_points(p.mask, q.mask)
                    if dist <= radius and (best is None or dist < best[0]):
                        best = (dist, p, q, a, b)
            if best is None:
                break
            _, p, q, a, b = best
            keep, drop = sorted((p, q), key=lambda x: (-x.skel_len, -x.area, x.new_id))
            keep.mask = np.logical_or.reduce((keep.mask, drop.mask, line_bridge(keep.mask.shape, a, b)))
            refresh_piece(keep)
            drop.same_source_into = keep.new_id
            drop.dropped = True
            merged += 1
    return merged


def absorb_small_pieces(pieces: list[Piece], radius: int, absorb_len: int, min_keep_len: int) -> None:
    for p in pieces:
        if p.skel_len < min_keep_len:
            p.dropped = True
    active = [p for p in pieces if not p.dropped]
    for p in sorted(active, key=lambda x: x.skel_len):
        if p.skel_len >= absorb_len:
            continue
        halo = binary_dilation(p.mask, iterations=max(1, radius))
        best, best_touch = None, 0
        for q in active:
            if q is p or q.dropped or q.skel_len <= p.skel_len:
                continue
            touch = int(np.count_nonzero(np.logical_and(halo, q.mask)))
            if touch > best_touch:
                best, best_touch = q, touch
        if best is not None and best_touch > 0:
            best.mask = np.logical_or(best.mask, p.mask)
            best.area = int(best.mask.sum())
            best.skel_len = int(np.count_nonzero(skeletonize(best.mask)))
            p.absorbed_into = best.new_id
            p.dropped = True
        elif p.skel_len < min_keep_len:
            p.dropped = True


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
        pieces = split_pieces_from_layers(load_layered_components(layered_src, lbl.shape))
    else:
        pieces = split_pieces(lbl)
    same_source_merged = bridge_same_source_pieces(pieces, args.same_source_bridge_radius)
    absorb_small_pieces(pieces, args.absorb_radius, args.absorb_len, args.min_keep_len)

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
            out[np.logical_and(layer, out == 0)] = out_id

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
        "pieces_after_split": len(pieces),
        "same_source_merged_pieces": int(same_source_merged),
        "kept_labels": int(out.max()),
        "absorbed_pieces": int(sum(p.absorbed_into is not None for p in pieces)),
        "same_source_absorbed_pieces": int(sum(p.same_source_into is not None for p in pieces)),
        "dropped_pieces": int(sum(p.dropped and p.absorbed_into is None and p.same_source_into is None for p in pieces)),
        "thicken_px": int(args.thicken_px),
        "smooth_window": int(args.smooth_window),
        "same_source_bridge_radius": int(args.same_source_bridge_radius),
        **layered_summary,
    }
    (args.output / "post_process_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
