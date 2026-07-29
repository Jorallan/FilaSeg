"""Classical, mask-only baselines for the FilaSeg paper.

Both baselines below consume the SAME binary filament-centreline mask that
FilaSeg's own pipeline consumes, and produce an instance-label image. Neither
baseline looks at the SEM image, at ground truth, or calls any FilaSeg stage
(stringart / preprocess / reconnect / postprocess) -- they are pure functions
of the mask, included so the paper can quantify what "no junction handling"
(baseline A) and "no evidence-driven repair, just local geometry" (baseline B)
each cost relative to the full pipeline.

Baseline A -- connected components
-----------------------------------
Binarize, then 8-connected-component label the mask. No junction resolution,
no gap bridging. This quantifies the failure of pure foreground connectivity:
every crossing fuses two filaments into one component, and every break in a
low-contrast filament splits it into two.

Baseline B -- skeleton graph + minimum-turn continuation
----------------------------------------------------------
1. Skeletonize the mask (`skimage.morphology.skeletonize`).
2. Prune terminal spurs shorter than `prune_spur_px`.
3. Split the pruned skeleton into atomic branches by removing junction
   pixels (skeleton pixels with >= 3 eight-neighbours) and labelling the
   8-connected remainder; branches shorter than `min_branch_px` are dropped.
4. For each branch, order its pixels into a path and fit an outward unit
   tangent at each end by least-squares (PCA) over the last
   `tangent_window_px` pixels.
5. Candidate pairings are any two distinct branch-ends within `max_gap_px`
   of each other (this covers both true gaps and the near-zero distance
   between several branches meeting at one junction).
6. Each candidate is scored by its turning angle (the direction change
   needed to continue straight through the join); candidates whose angle
   exceeds `max_turn_deg` are rejected outright.
7. Remaining candidates are accepted greedily in increasing turning-angle
   order, each end usable at most once (a greedy maximum-continuation
   matching). No other gate exists: no Hough voting, no SEM width evidence,
   no topology repair, no orientation layer -- that is the point of this
   baseline.
8. Branches connected (directly or transitively) through accepted pairings
   are union-find grouped into one instance each. Every foreground mask
   pixel -- including junction pixels and pixels belonging to branches that
   were pruned/discarded -- is assigned to the group whose surviving
   skeleton is nearest, via `scipy.ndimage.distance_transform_edt(...,
   return_indices=True)` on the complement of the union of all kept groups'
   skeleton pixels. This guarantees full coverage: no foreground pixel is
   left unlabelled.

Threshold table (every free parameter, its default, and its role)
-------------------------------------------------------------------
Baseline A:
    min_area          15    px area below which a component is dropped as a
                              speck. The ONLY cleanup baseline A performs.

Baseline B:
    max_gap_px        15.0  px, max Euclidean distance between two branch
                              ends to be considered a candidate join. This
                              is a REAL free parameter -- sweep it (see
                              `sweep_baseline_b`).
    min_branch_px     6     px, atomic branches shorter than this are
                              dropped from the pairing graph (their pixels
                              still get painted into the nearest surviving
                              instance at render time).
    prune_spur_px     3     px, terminal skeleton spurs (dead ends hanging
                              off a junction) shorter than this are erased
                              before branch splitting.
    tangent_window_px 8     px (pixel count, not distance), number of path
                              pixels used in the least-squares tangent fit
                              at each branch end.
    max_turn_deg      90.0  degrees, max turning angle for a candidate join
                              to be accepted at all. This is the baseline's
                              OTHER real free parameter -- sweep it together
                              with max_gap_px.

Tunability note: only `max_gap_px` and `max_turn_deg` are meant to be swept
for the paper. Do this only on `input/synthetic_thick/cov*` development
scenes -- never on `input/synthetic_locked_v1`, which is a locked evaluation
set. `sweep_baseline_b` runs baseline B over a gap sweep and records instance
counts (a GT-free proxy) to help a human pick a value; it does not score
against ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize

__all__ = [
    "baseline_connected_components",
    "baseline_skeleton_minturn",
    "minimum_turn_group_fragments",
    "sweep_baseline_b",
    "main",
]

# 8-connectivity neighbour offsets (row, col), excluding the centre.
_NEIGH8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


# ═══════════════════════════ Baseline A ════════════════════════════════


def baseline_connected_components(mask: np.ndarray, min_area: int = 15) -> np.ndarray:
    """8-connected-component labelling of a binary mask. No junction/gap logic.

    Parameters
    ----------
    mask : array-like
        Binary (or 0/255, or bool) filament-centreline mask.
    min_area : int, default 15
        Components smaller than this (in pixels) are dropped as specks. This
        is the ONLY cleanup this baseline performs -- a single, transparent,
        documented parameter.

    Returns
    -------
    np.ndarray (int32)
        Label image, 0 = background, 1..K = instance ids (consecutive,
        assigned in raster-scan discovery order -- deterministic).
    """
    m = _binarize(mask)
    lbl = cc_label(m, connectivity=2).astype(np.int64)
    n_labels = int(lbl.max())
    out = np.zeros(m.shape, dtype=np.int32)
    if n_labels == 0:
        return out
    counts = np.bincount(lbl.ravel(), minlength=n_labels + 1)
    remap = np.zeros(n_labels + 1, dtype=np.int32)
    next_id = 1
    for i in range(1, n_labels + 1):
        if counts[i] >= min_area:
            remap[i] = next_id
            next_id += 1
    return remap[lbl].astype(np.int32)


# ═══════════════════════════ Baseline B ════════════════════════════════


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-connected skeleton neighbours (0 off-skeleton)."""
    k = np.ones((3, 3), dtype=np.int32)
    k[1, 1] = 0
    deg = ndi.convolve(skel.astype(np.int32), k, mode="constant", cval=0)
    deg = deg * skel.astype(np.int32)
    return deg


def _prune_spurs(skel: np.ndarray, prune_spur_px: int) -> np.ndarray:
    """Erase terminal skeleton spurs (endpoint -> junction) shorter than
    `prune_spur_px` pixels. Isolated branches with a free endpoint at BOTH
    ends (i.e. no junction reached) are never pruned -- they are whole
    filaments, not dangling spurs.
    """
    skel = skel.copy()
    if prune_spur_px <= 0:
        return skel
    H, W = skel.shape
    for _ in range(64):  # generous convergence cap
        deg = _degree_map(skel)
        endpoints = [tuple(p) for p in np.argwhere(skel & (deg == 1))]
        if not endpoints:
            break
        removed_any = False
        for (r0, c0) in endpoints:
            if not skel[r0, c0]:
                continue  # already erased while pruning another spur this pass
            r, c = r0, c0
            prev = None
            spur_pixels = [(r0, c0)]
            junction_found = False
            while True:
                d = int(deg[r, c])
                if d >= 3:
                    junction_found = True
                    break
                nbrs = [
                    (r + dr, c + dc) for dr, dc in _NEIGH8
                    if 0 <= r + dr < H and 0 <= c + dc < W
                    and skel[r + dr, c + dc] and (r + dr, c + dc) != prev
                ]
                if not nbrs:
                    break  # reached the branch's OTHER free endpoint: not a spur
                prev = (r, c)
                r, c = nbrs[0]
                if int(deg[r, c]) >= 3:
                    junction_found = True
                    break  # stop before stepping onto the junction pixel
                spur_pixels.append((r, c))
            if junction_found and len(spur_pixels) < prune_spur_px:
                for (rr, cc) in spur_pixels:
                    skel[rr, cc] = False
                removed_any = True
        if not removed_any:
            break
    return skel


def _order_piece(piece_set: set) -> List[Tuple[int, int]]:
    """Order a thin (degree<=2) pixel set into a path via double-BFS
    (tree-diameter trick): robust to simple chains, single pixels, and
    closed loops alike.
    """
    def neighbors(p):
        r, c = p
        return [
            (r + dr, c + dc) for dr, dc in _NEIGH8
            if (r + dr, c + dc) in piece_set
        ]

    def bfs(start):
        visited = {start: None}
        q = deque([start])
        last = start
        while q:
            cur = q.popleft()
            last = cur
            for nb in neighbors(cur):
                if nb not in visited:
                    visited[nb] = cur
                    q.append(nb)
        return last, visited

    start = sorted(piece_set)[0]
    a, _ = bfs(start)
    b, visited_from_a = bfs(a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = visited_from_a[cur]
    path.reverse()  # a -> b
    return path


def _fit_tangent(sub_pts: np.ndarray) -> np.ndarray:
    """Least-squares (PCA) unit tangent over `sub_pts`, oriented outward.

    `sub_pts` must be ordered starting at the branch end and proceeding
    inward (sub_pts[0] == the end pixel). The direction sign is fixed so it
    points from the interior toward (and past) the end -- i.e. outward.
    """
    pts = np.asarray(sub_pts, dtype=float)
    if len(pts) < 2:
        return np.zeros(2)
    crude = pts[0] - pts[-1]
    crude_norm = float(np.linalg.norm(crude))
    if crude_norm < 1e-9:
        return np.zeros(2)
    mean = pts.mean(axis=0)
    centered = pts - mean
    if np.allclose(centered, 0):
        return crude / crude_norm
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if np.dot(direction, crude) < 0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return crude / crude_norm
    return direction / norm


def _split_branches(
    skel: np.ndarray, min_branch_px: int, tangent_window_px: int
) -> List[dict]:
    """Split a (spur-pruned) skeleton into atomic branches.

    Junction pixels (degree >= 3) are removed to physically separate arms;
    the 8-connected remainder is labelled and each piece below
    `min_branch_px` pixels is dropped (its pixels are still painted into the
    nearest surviving instance at render time -- they just don't get an end
    of their own to pair with).
    """
    deg = _degree_map(skel)
    junction_mask = skel & (deg >= 3)
    rem = skel & ~junction_mask
    lbl = cc_label(rem, connectivity=2)
    n_pieces = int(lbl.max())
    branches: List[dict] = []
    w = max(2, int(tangent_window_px))
    for k in range(1, n_pieces + 1):
        coords = [tuple(int(v) for v in p) for p in np.argwhere(lbl == k)]
        if len(coords) < min_branch_px:
            continue
        piece_set = set(coords)
        path = _order_piece(piece_set)
        if len(path) < 2:
            continue  # cannot define an outward direction from one pixel
        sub_a = np.array(path[: min(w, len(path))], dtype=float)
        sub_b = np.array(list(reversed(path[-min(w, len(path)):])), dtype=float)
        tan_a = _fit_tangent(sub_a)
        tan_b = _fit_tangent(sub_b)
        branches.append({
            "id": k,
            "pixels": coords,
            "ends": [(path[0], tan_a), (path[-1], tan_b)],
        })
    return branches


def _match_ends(
    branches: List[dict], max_gap_px: float, max_turn_deg: float
) -> Dict[int, int]:
    """Greedy minimum-turning-angle matching between branch ends.

    Returns a union-find parent map {branch_id -> branch_id} after merging
    branches joined by an accepted pairing.
    """
    uf = {br["id"]: br["id"] for br in branches}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            uf[rx] = ry

    ends = []
    for br in branches:
        for ei, (coord, tan) in enumerate(br["ends"]):
            if float(np.linalg.norm(tan)) < 1e-9:
                continue  # degenerate tangent: cannot score a turning angle
            ends.append({
                "bid": br["id"], "end": ei,
                "coord": np.asarray(coord, dtype=float), "tan": tan,
            })
    if len(ends) < 2:
        return uf

    coords_arr = np.array([e["coord"] for e in ends])
    tree = cKDTree(coords_arr)
    pairs = tree.query_pairs(r=float(max_gap_px), output_type="ndarray")

    tol = 1e-6
    cands = []
    for i, j in pairs:
        i, j = int(i), int(j)
        ei, ej = ends[i], ends[j]
        if ei["bid"] == ej["bid"]:
            continue  # an end may not pair with the other end of its own branch
        dist = float(np.linalg.norm(ei["coord"] - ej["coord"]))
        cosang = float(np.dot(ei["tan"], -ej["tan"]))
        cosang = max(-1.0, min(1.0, cosang))
        angle_deg = float(np.degrees(np.arccos(cosang)))
        if angle_deg > float(max_turn_deg) + tol:
            continue
        lo, hi = (i, j) if i < j else (j, i)
        cands.append((angle_deg, dist, lo, hi))
    cands.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    used = set()
    for angle_deg, dist, i, j in cands:
        ei, ej = ends[i], ends[j]
        key_i = (ei["bid"], ei["end"])
        key_j = (ej["bid"], ej["end"])
        if key_i in used or key_j in used:
            continue
        used.add(key_i)
        used.add(key_j)
        union(ei["bid"], ej["bid"])

    return uf


def minimum_turn_group_fragments(
    fragment_masks: Sequence[np.ndarray],
    max_gap_px: float = 15.0,
    tangent_window_px: int = 8,
    max_turn_deg: float = 90.0,
) -> Dict[int, np.ndarray]:
    """Group a fixed fragment set without re-extraction or rendering.

    Each supplied mask remains one atomic unit. The returned
    ``group_id -> bool mask`` mapping contains unions of the original masks,
    preserving overlapping source fragments until their groups are formed
    and introducing no bridge or nearest-fill pixels.
    """
    masks = [np.asarray(mask, dtype=bool) for mask in fragment_masks]
    if not masks:
        return {}
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("all fragment masks must have the same shape")

    branches: List[dict] = []
    w = max(2, int(tangent_window_px))
    for fragment_id, mask in enumerate(masks, start=1):
        skel = skeletonize(mask)
        coords = [tuple(int(v) for v in point) for point in np.argwhere(skel)]
        if not coords:
            continue
        path = _order_piece(set(coords))
        if len(path) >= 2:
            sub_a = np.asarray(path[: min(w, len(path))], dtype=float)
            sub_b = np.asarray(
                list(reversed(path[-min(w, len(path)):])), dtype=float
            )
            ends = [
                (path[0], _fit_tangent(sub_a)),
                (path[-1], _fit_tangent(sub_b)),
            ]
        else:
            zero = np.zeros(2, dtype=float)
            ends = [(path[0], zero), (path[0], zero)]
        branches.append({
            "id": fragment_id,
            "pixels": coords,
            "ends": ends,
        })

    if not branches:
        return {}
    uf = _match_ends(branches, max_gap_px, max_turn_deg)

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    roots = sorted({find(branch["id"]) for branch in branches})
    root_to_group = {root: group for group, root in enumerate(roots, start=1)}
    grouped = {
        group: np.zeros(shape, dtype=bool)
        for group in root_to_group.values()
    }
    for branch in branches:
        group = root_to_group[find(branch["id"])]
        grouped[group] |= masks[branch["id"] - 1]
    return grouped


def baseline_skeleton_minturn(
    mask: np.ndarray,
    max_gap_px: float = 15.0,
    min_branch_px: int = 6,
    prune_spur_px: int = 3,
    tangent_window_px: int = 8,
    max_turn_deg: float = 90.0,
) -> np.ndarray:
    """Skeleton graph + minimum-turn continuation instance labelling.

    See the module docstring for the full algorithm description and the
    threshold table. `max_gap_px` and `max_turn_deg` are the two real free
    parameters meant to be swept (see `sweep_baseline_b`); the rest are
    fixed, documented defaults.

    Parameters
    ----------
    mask : array-like
        Binary (or 0/255, or bool) filament-centreline mask.
    max_gap_px : float, default 15.0
        Max Euclidean distance (px) between two branch ends to be a
        candidate join.
    min_branch_px : int, default 6
        Atomic branches shorter than this (px) are dropped from pairing.
    prune_spur_px : int, default 3
        Terminal skeleton spurs shorter than this (px) are erased first.
    tangent_window_px : int, default 8
        Pixel count used in the least-squares tangent fit at each end.
    max_turn_deg : float, default 90.0
        Max turning angle (degrees) for a candidate join to be accepted.

    Returns
    -------
    np.ndarray (int32)
        Label image, 0 = background, 1..K = instance ids. Every foreground
        mask pixel is assigned to exactly one instance.
    """
    m = _binarize(mask)
    skel = skeletonize(m)
    skel = _prune_spurs(skel, prune_spur_px)
    branches = _split_branches(skel, min_branch_px, tangent_window_px)

    out = np.zeros(m.shape, dtype=np.int32)
    if not branches:
        # Nothing survived pruning/splitting (e.g. all specks): fall back to
        # raw connectivity so no foreground pixel is left unlabelled.
        return baseline_connected_components(m, min_area=1)

    uf = _match_ends(branches, max_gap_px, max_turn_deg)

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    roots = sorted({find(br["id"]) for br in branches})
    root_to_gid = {root: gid for gid, root in enumerate(roots, start=1)}

    seed = np.zeros(m.shape, dtype=np.int32)
    for br in branches:
        gid = root_to_gid[find(br["id"])]
        for (r, c) in br["pixels"]:
            seed[r, c] = gid

    background = seed == 0
    _, (ind_r, ind_c) = ndi.distance_transform_edt(background, return_indices=True)
    nearest = seed[ind_r, ind_c]
    out[m] = nearest[m]
    return out


# ═══════════════════════════ Sweep helper ══════════════════════════════


def sweep_baseline_b(
    sample_dirs: Sequence[str],
    mask_name: str,
    gaps: Sequence[float],
    out_json: str,
    **kwargs,
) -> dict:
    """Run baseline B over `gaps` for each sample dir; record instance counts.

    This is a GT-free helper: it does NOT score against ground truth (that
    is a separate module's job). It only records, per development sample and
    per gap value, the number of instances baseline B produces -- a simple
    proxy a human can look at (fewer instances at larger gaps usually means
    more merging) to help choose a value for `max_gap_px`.

    Do not run this over `input/synthetic_locked_v1` -- it is a locked
    evaluation set. Use `input/synthetic_thick/cov*/` for development.

    Parameters
    ----------
    sample_dirs : sequence of str/Path
        Development sample directories, each containing `mask_name`.
    mask_name : str
        Mask filename to load from each sample dir (e.g. "mask_w1.png").
    gaps : sequence of float
        Values of `max_gap_px` to sweep.
    out_json : str
        Output JSON path.
    **kwargs
        Extra keyword arguments forwarded to `baseline_skeleton_minturn`
        (e.g. `max_turn_deg=`, `min_branch_px=`).

    Returns
    -------
    dict
        The same structure written to `out_json`.
    """
    from skimage import io as skio

    per_sample: Dict[str, Dict[str, int]] = {}
    for d in sample_dirs:
        d = Path(d)
        m = skio.imread(str(d / mask_name))
        if m.ndim == 3:
            m = m[..., :3].mean(axis=2)
        per_gap: Dict[str, int] = {}
        for g in gaps:
            lab = baseline_skeleton_minturn(m, max_gap_px=float(g), **kwargs)
            per_gap[str(g)] = int(np.count_nonzero(np.unique(lab)))
        per_sample[str(d)] = per_gap

    gap_mean_instances: Dict[str, float] = {}
    for g in gaps:
        vals = [per_sample[str(Path(d))][str(g)] for d in sample_dirs]
        gap_mean_instances[str(g)] = float(np.mean(vals)) if vals else None

    result = {
        "mask_name": mask_name,
        "gaps": [float(g) for g in gaps],
        "per_sample": per_sample,
        "gap_mean_instances": gap_mean_instances,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ═══════════════════════════ CLI ═══════════════════════════════════════


def _instance_count(lab: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(lab)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Classical mask-only instance-labelling baselines for FilaSeg "
            "(connected components, or skeleton graph + minimum-turn "
            "continuation)."
        )
    )
    ap.add_argument("--mask", required=True, help="Path to binary mask image.")
    ap.add_argument("--method", required=True, choices=["cc", "skeleton"])
    ap.add_argument("--out", required=True, help="Output int32 label .tif path.")
    ap.add_argument("--min-area", type=int, default=15,
                     help="Baseline A: min component area in px (default 15).")
    ap.add_argument("--max-gap-px", type=float, default=15.0,
                     help="Baseline B: max end-to-end gap in px (default 15.0).")
    ap.add_argument("--min-branch-px", type=int, default=6,
                     help="Baseline B: min atomic-branch length in px (default 6).")
    ap.add_argument("--prune-spur-px", type=int, default=3,
                     help="Baseline B: max spur length erased, in px (default 3).")
    ap.add_argument("--tangent-window-px", type=int, default=8,
                     help="Baseline B: tangent-fit window in px (default 8).")
    ap.add_argument("--max-turn-deg", type=float, default=90.0,
                     help="Baseline B: max accepted turning angle, deg (default 90).")
    args = ap.parse_args(argv)

    from skimage import io as skio
    m = skio.imread(args.mask)
    if m.ndim == 3:
        m = m[..., :3].mean(axis=2)

    if args.method == "cc":
        lab = baseline_connected_components(m, min_area=args.min_area)
    else:
        lab = baseline_skeleton_minturn(
            m,
            max_gap_px=args.max_gap_px,
            min_branch_px=args.min_branch_px,
            prune_spur_px=args.prune_spur_px,
            tangent_window_px=args.tangent_window_px,
            max_turn_deg=args.max_turn_deg,
        )

    import tifffile
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), lab.astype(np.int32))
    print(f"{args.method}: {_instance_count(lab)} instances -> {out_path}")


if __name__ == "__main__":
    main()
