"""Method-independent evaluation representation for FilaSeg vs. baselines.

The grouping metric in `eval/metrics.py` partitions "fragments" produced by
FilaSeg's OWN Stage 1 (stringart tiling) and Stage 2 (preprocess branches) --
see `instance_io.load_fragments`. That fragment set changes whenever Stage 1
changes, so it cannot be used to fairly compare FilaSeg against an external
baseline method that never produces those fragments at all.

This module builds a fragment set from the INPUT MASK ALONE (skeleton
branches split at junctions/endpoints -- `common_fragments`), then measures
any method (FilaSeg or a baseline) by assigning those fragments to that
method's output instances (`assign_fragments`) and scoring the two induced
partitions of the *same* fragment set against each other
(`pairwise_scores`), plus a pixel-level instance-recovery scorecard
(`instance_recovery`). Because the fragment set never depends on any
method's internal representation, every method compared this way is
measured on identical atomic units (see `tests/test_common_metric.py`,
case 6, for a direct proof).

The scoring implementation lives entirely in this module.  In particular it
does not rely on private helpers from `eval.metrics`: this keeps the common
metric independently usable and makes its treatment of missing assignments
auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Match the sys.path convention used by eval/eval_reconnect.py: eval/*.py
# modules import each other with bare names (e.g. `import metrics`), which
# only works if eval/ itself is on sys.path -- it is not a package (no
# __init__.py), so this insertion is required both when this file is run
# directly as a script and when it is imported as `eval.common_metric`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)

import instance_io as iio   # noqa: E402


Instances = Union[np.ndarray, Dict[int, np.ndarray], "iio.InstanceSet"]


# ── 1. method-independent fragment set ────────────────────────────────────


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """8-neighbour count for every skeleton pixel (0 elsewhere)."""
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    deg = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return deg * skel


def _prune_spurs(skel: np.ndarray, prune_spur_px: int, max_iters: int = 20) -> np.ndarray:
    """Remove short dead-end spurs hanging off a junction (skeletonization
    noise), so they don't fragment a real branch or fake an extra junction.

    A "spur" here is a connected piece of skeleton (with the junction pixels
    themselves excluded) that (a) touches an endpoint (degree-1 pixel) at one
    end, (b) is 8-adjacent to a junction (degree>=3) pixel at the other end,
    and (c) is no longer than `prune_spur_px` pixels. Iterates because
    removing one spur can change a former junction's degree, exposing a new
    endpoint or collapsing a former junction into an ordinary path pixel.
    """
    from skimage.measure import label as cc_label
    from scipy.ndimage import binary_dilation

    skel = skel.copy()
    if prune_spur_px <= 0:
        return skel

    struct = np.ones((3, 3), dtype=bool)
    for _ in range(max_iters):
        deg = _degree_map(skel)
        junctions = skel & (deg >= 3)
        endpoints = skel & (deg == 1)
        if not junctions.any():
            break
        non_junc = skel & ~junctions
        lbl = cc_label(non_junc, connectivity=2)
        n = int(lbl.max())
        if n == 0:
            break
        junc_dilated = binary_dilation(junctions, structure=struct)

        removed_any = False
        for k in range(1, n + 1):
            piece = lbl == k
            if int(piece.sum()) > prune_spur_px:
                continue
            if (piece & junc_dilated).any() and (piece & endpoints).any():
                skel[piece] = False
                removed_any = True
        if not removed_any:
            break
    return skel


def common_fragments(
    mask: np.ndarray,
    min_len_px: int = 6,
    prune_spur_px: int = 3,
) -> np.ndarray:
    """Atomic reference fragments derived ONLY from the input centreline mask.

    This is the method-independent unit that every evaluated method (FilaSeg
    or a baseline) is measured against. It does not look at any method's
    output at all -- it depends solely on `mask`. See
    `tests/test_common_metric.py::test_fragments_method_independent` for a
    direct proof that permuting an unrelated prediction's instance labels
    does not change this function's output.

    Steps: skeletonize the binary mask; prune short skeletonization spurs
    (<= `prune_spur_px`, see `_prune_spurs`); mark junction pixels (>=3
    8-neighbours) and remove them from the skeleton; label the remaining
    8-connected pieces -- each is an atomic branch, naturally split at every
    junction and endpoint; discard branches shorter than `min_len_px` pixels.

    Returns an int32 label image, same shape as `mask`, 0 = background,
    1..K = fragment ids (renumbered contiguously after the length filter).
    """
    from skimage.morphology import skeletonize
    from skimage.measure import label as cc_label

    mask = np.asarray(mask).astype(bool)
    skel = skeletonize(mask)
    skel = _prune_spurs(skel, prune_spur_px)

    deg = _degree_map(skel)
    junctions = skel & (deg >= 3)
    non_junc = skel & ~junctions

    lbl = cc_label(non_junc, connectivity=2)
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for k in range(1, int(lbl.max()) + 1):
        piece = lbl == k
        if int(piece.sum()) >= min_len_px:
            out[piece] = next_id
            next_id += 1
    return out


# ── shared instance-representation helper ─────────────────────────────────


def _as_masks(instances: Instances) -> Dict[int, np.ndarray]:
    """Normalise any accepted instance representation to {id: bool mask}.

    Accepts: a dict already in that shape, an `InstanceSet` (duck-typed via
    `.masks`), a 2-D int label image (0 = background), or a 3-D (K, H, W)
    boolean/int multi-label stack. For the stack form there is no id in the
    data itself, so instance id := index + 1 (1-based) along axis 0.
    """
    if isinstance(instances, dict):
        return {int(k): np.asarray(v).astype(bool) for k, v in instances.items()}
    if hasattr(instances, "masks"):
        return {int(k): np.asarray(v).astype(bool) for k, v in instances.masks.items()}
    arr = np.asarray(instances)
    if arr.ndim == 2:
        return {int(i): arr == i for i in np.unique(arr) if int(i) != 0}
    if arr.ndim == 3:
        return {k + 1: arr[k].astype(bool) for k in range(arr.shape[0])}
    raise ValueError(f"instances must be 2D label image or (K,H,W) stack, got shape {arr.shape}")


# ── 2. fragment -> instance assignment ────────────────────────────────────


def assign_fragments(
    frag_labels: np.ndarray,
    instances: Instances,
    min_overlap_frac: float = 0.3,
    max_dist_px: float = 6.0,
) -> Dict[int, int]:
    """Assign every common fragment to exactly one instance of `instances`.

    `instances` may be a 2-D int label image OR a 3-D (K, H, W) multi-label
    stack (see `_as_masks`) -- both a ground-truth labelling and any method's
    prediction are handled the same way.

    Assignment votes against each instance's FULL mask (rather than a
    flattened label image), so overlapping instances at crossings retain
    their ownership.  With insufficient direct overlap it falls back to the
    closest flattened instance within `max_dist_px`. Returns
    {fragment_label: instance_label_or_0} (0 = unassigned).
    """
    frag_labels = np.asarray(frag_labels)
    ids = [int(i) for i in np.unique(frag_labels) if int(i) != 0]
    if not ids:
        return {}
    masks = _as_masks(instances)
    if any(m.shape != frag_labels.shape for m in masks.values()):
        raise ValueError("instance masks must have the same shape as frag_labels")
    if not masks:
        return {i: 0 for i in ids}

    # A deterministic flattened field is only used for distance fallback;
    # direct-overlap voting above it remains fully overlap-aware.
    flat = np.zeros(frag_labels.shape, dtype=np.int64)
    order = sorted(masks, key=lambda k: (-int(masks[k].sum()), k))
    for instance_id in order:
        take = masks[instance_id] & (flat == 0)
        flat[take] = instance_id

    from scipy.ndimage import distance_transform_edt
    occupied = flat != 0
    if occupied.any() and not occupied.all():
        dist, nearest_idx = distance_transform_edt(~occupied, return_indices=True)
        nearest = flat[tuple(nearest_idx)]
    else:
        dist = np.zeros(flat.shape, dtype=float)
        nearest = flat

    out: Dict[int, int] = {}
    flat_masks = {k: m.ravel() for k, m in masks.items()}
    for fragment_id in ids:
        fragment = frag_labels == fragment_id
        idx = np.flatnonzero(fragment.ravel())
        best_id, best_count = 0, 0
        for instance_id in sorted(masks):
            count = int(flat_masks[instance_id][idx].sum())
            if count > best_count:
                best_id, best_count = instance_id, count
        if best_count >= min_overlap_frac * idx.size:
            out[fragment_id] = best_id
            continue
        candidates = nearest[fragment & (dist <= max_dist_px)]
        candidates = candidates[candidates != 0]
        if candidates.size:
            labels, counts = np.unique(candidates, return_counts=True)
            out[fragment_id] = int(labels[np.argmax(counts)])
        else:
            out[fragment_id] = 0
    return out


# ── 3. pairwise co-assignment scores ───────────────────────────────────────


def pairwise_scores(gt_assign: Dict[int, int], pred_assign: Dict[int, int]) -> dict:
    """Permutation-invariant pairwise co-assignment scores.

    Computed over exactly the fragments that have a VALID (non-zero) ground-
    truth assignment. With n_ij the number of
    fragments in gt-group i and pred-group j,
        TP = sum_ij C(n_ij, 2)
        same_pred = sum_j C(b_j, 2)   (b_j = pred-group sizes)
        same_gt   = sum_i C(a_i, 2)   (a_i = gt-group sizes)
        FP = same_pred - TP, FN = same_gt - TP, P = TP/(TP+FP), R = TP/(TP+FN).

    Unlike a naive contingency table over the raw label vectors, a fragment
    the PREDICTION leaves unassigned (pred_assign[k] == 0) is given its own
    unique singleton predicted cluster here, rather than being lumped
    together with every other unassigned fragment into one shared "cluster
    0" -- that lumping would manufacture pairwise co-clustering between
    fragments that the prediction never actually joined. The same
    singleton-per-unassigned-fragment treatment is applied on the
    ground-truth side for symmetry/robustness (normally a no-op here, since
    fragments with gt_assign == 0 are excluded from the comparison set
    entirely, per the paragraph above).
    """
    all_frag_ids = sorted(set(gt_assign) | set(pred_assign))
    raw_gt_all = [int(gt_assign.get(k, 0)) for k in all_frag_ids]
    raw_pred_all = [int(pred_assign.get(k, 0)) for k in all_frag_ids]
    frag_ids = [k for k in all_frag_ids if gt_assign.get(k, 0) != 0]
    raw_gt = [int(gt_assign[k]) for k in frag_ids]
    raw_pred = [int(pred_assign.get(k, 0)) for k in frag_ids]
    n = len(frag_ids)

    n_gt_groups = len({v for v in raw_gt if v != 0})
    n_pred_groups = len({v for v in raw_pred if v != 0})

    coverage = {
        "n_total_fragments": len(all_frag_ids),
        "n_gt_assigned_fragments": sum(v != 0 for v in raw_gt_all),
        "n_gt_unassigned_fragments": sum(v == 0 for v in raw_gt_all),
        "n_pred_assigned_fragments": sum(v != 0 for v in raw_pred_all),
        "n_pred_unassigned_fragments": sum(v == 0 for v in raw_pred_all),
    }

    if n < 2:
        return {
            **coverage, "n_fragments": n, "n_scored_fragments": n,
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            "pair_evidence_available": False, "pair_evidence_reason": "fewer_than_two_scored_fragments",
            "n_gt_groups": n_gt_groups, "n_pred_groups": n_pred_groups,
            "n_gt_pairs": 0.0, "n_pred_pairs": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0,
        }

    def _singletonize(vals: List[int]) -> np.ndarray:
        out = np.empty(len(vals), dtype=np.int64)
        next_id = -1
        for idx, v in enumerate(vals):
            if v == 0:
                out[idx] = next_id
                next_id -= 1
            else:
                out[idx] = v
        return out

    gt_vec = _singletonize(raw_gt)
    pred_vec = _singletonize(raw_pred)

    # Local contingency + choose-two implementation: never couple this public
    # score to private internals of eval.metrics.
    _, gi = np.unique(gt_vec, return_inverse=True)
    _, pi = np.unique(pred_vec, return_inverse=True)
    n_mat = np.zeros((gi.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(n_mat, (gi, pi), 1)
    a, b = n_mat.sum(axis=1), n_mat.sum(axis=0)
    comb2 = lambda x: np.asarray(x, dtype=float) * (np.asarray(x, dtype=float) - 1.0) / 2.0
    tp = float(comb2(n_mat).sum())
    same_pred = float(comb2(b).sum())
    same_gt = float(comb2(a).sum())
    fp = same_pred - tp
    fn = same_gt - tp
    precision = tp / same_pred if same_pred > 0 else float("nan")
    recall = tp / same_gt if same_gt > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else (0.0 if np.isfinite(precision) and np.isfinite(recall) else float("nan")))
    if same_gt == 0 and same_pred == 0:
        evidence_reason = "no_positive_pairs_in_either_partition"
    elif same_gt == 0:
        evidence_reason = "no_positive_ground_truth_pairs"
    elif same_pred == 0:
        evidence_reason = "no_positive_prediction_pairs"
    else:
        evidence_reason = "ok"

    return {
        **coverage,
        "n_fragments": n,
        "n_scored_fragments": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt_groups": n_gt_groups,
        "n_pred_groups": n_pred_groups,
        "pair_evidence_available": evidence_reason == "ok",
        "pair_evidence_reason": evidence_reason,
        "n_gt_pairs": same_gt,
        "n_pred_pairs": same_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ── 4. pixel-level instance recovery ───────────────────────────────────────


def fragment_instance_recovery(
    gt_assign: Dict[int, int],
    pred_assign: Dict[int, int],
    coverage_thr: float = 0.8,
    purity_thr: float = 0.8,
) -> dict:
    """Shared instance recovery on common-fragment assignments.

    This is the fair recovery score for comparing methods with different pixel
    representations (for example, a thin-axis baseline and thick synthetic
    GT).  Every count is over the method-independent fragment set.  A GT
    instance is well recovered only when its dominant predicted instance is
    mutual and meets both fragment-count coverage and purity thresholds.
    Missing predicted assignments are absorbed; predicted groups with no
    dominant GT are false instances.
    """
    fragment_ids = sorted(set(gt_assign) | set(pred_assign))
    gt_total: Dict[int, int] = {}
    pred_total: Dict[int, int] = {}
    cell: Dict[Tuple[int, int], int] = {}
    for fragment_id in fragment_ids:
        g = int(gt_assign.get(fragment_id, 0))
        p = int(pred_assign.get(fragment_id, 0))
        if g:
            gt_total[g] = gt_total.get(g, 0) + 1
        if p:
            pred_total[p] = pred_total.get(p, 0) + 1
        if g and p:
            cell[(g, p)] = cell.get((g, p), 0) + 1

    gt_ids = sorted(gt_total)
    pred_ids = sorted(pred_total)
    gt_dom = {
        g: max((p for (gg, p) in cell if gg == g), key=lambda p: cell[(g, p)], default=0)
        for g in gt_ids
    }
    pred_dom = {
        p: max((g for (g, pp) in cell if pp == p), key=lambda g: cell[(g, p)], default=0)
        for p in pred_ids
    }

    well = fragmented = absorbed = 0
    dominant_predictions = set()
    for g in gt_ids:
        p = gt_dom[g]
        if p == 0:
            absorbed += 1
            continue
        dominant_predictions.add(p)
        overlap = cell[(g, p)]
        coverage = overlap / gt_total[g]
        purity = overlap / pred_total[p]
        if pred_dom.get(p) != g:
            absorbed += 1
        elif coverage >= coverage_thr and purity >= purity_thr:
            well += 1
        else:
            fragmented += 1

    n_gt = len(gt_ids)
    return {
        "n_gt_instances": n_gt,
        "n_pred_instances": len(pred_ids),
        "well_recovered": well,
        "fragmented": fragmented,
        "absorbed": absorbed,
        "false_instances": sum(p not in dominant_predictions for p in pred_ids),
        "recovery_rate": well / max(1, n_gt),
    }


def instance_recovery(gt: Instances, pred: Instances,
                       coverage_thr: float = 0.8, purity_thr: float = 0.8) -> dict:
    """Representation-dependent diagnostic recovery, computed on pixels.

    `eval/metrics.py` already has an `instance_recovery`, but it operates on
    a pair of aligned FRAGMENT->instance vectors (fragment-level
    granularity) and reports well/deleted/fragmented/false_pred. This
    function is not a duplicate of that -- it is deliberately pixel-level,
    and therefore should only be used as a representation-dependent
    diagnostic when methods emit different mask widths. It is reimplemented
    here rather than wrapped; the
    categorisation logic (dominant-match + mutuality + coverage/purity
    thresholds) mirrors metrics.py's approach, renaming
    'deleted' -> 'absorbed' and 'false_pred' -> 'false_instances' per spec.

    For each GT instance g, find its dominant predicted instance p* (largest
    pixel intersection). coverage = |g & p*| / |g|, purity = |g & p*| / |p*|.
    - well_recovered: the match is mutual (p*'s own dominant GT is g) AND
      coverage >= coverage_thr AND purity >= purity_thr.
    - absorbed: not mutual (p* really "belongs" to some other GT instance --
      g got swallowed into a merge), OR g has no overlap with any prediction
      at all (fully missed).
    - fragmented: mutual match, but under one of the thresholds (g survives
      as its own predicted id but is split/contaminated).
    - false_instances: predicted instances that are nobody's dominant match.
    """
    gt_masks = _as_masks(gt)
    pred_masks = _as_masks(pred)

    gt_ids = sorted(gt_masks)
    pred_ids = sorted(pred_masks)
    gt_areas = {g: int(gt_masks[g].sum()) for g in gt_ids}
    pred_areas = {p: int(pred_masks[p].sum()) for p in pred_ids}

    inter: Dict[Tuple[int, int], int] = {}
    for g in gt_ids:
        gm = gt_masks[g]
        for p in pred_ids:
            c = int(np.count_nonzero(gm & pred_masks[p]))
            if c > 0:
                inter[(g, p)] = c

    gt_dom = {
        g: max((p for (gg, p) in inter if gg == g), key=lambda p: inter[(g, p)], default=0)
        for g in gt_ids
    }
    pred_dom = {
        p: max((g for (g, pp) in inter if pp == p), key=lambda g: inter[(g, p)], default=0)
        for p in pred_ids
    }

    well = fragmented = absorbed = 0
    dom_preds = set()
    for g in gt_ids:
        p = gt_dom[g]
        if p == 0:
            absorbed += 1  # no overlap with any prediction at all
            continue
        dom_preds.add(p)
        coverage = inter[(g, p)] / max(1, gt_areas[g])
        purity = inter[(g, p)] / max(1, pred_areas[p])
        if pred_dom.get(p) != g:
            absorbed += 1
        elif coverage >= coverage_thr and purity >= purity_thr:
            well += 1
        else:
            fragmented += 1

    false_instances = sum(1 for p in pred_ids if p not in dom_preds)
    n = len(gt_ids)
    return {
        "n_gt_instances": n,
        "well_recovered": well,
        "fragmented": fragmented,
        "absorbed": absorbed,
        "false_instances": false_instances,
        "recovery_rate": well / max(1, n),
    }


# ── 5. convenience entry point ─────────────────────────────────────────────


def _load_mask(mask: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        return mask.astype(bool)
    from skimage import io as skio

    arr = skio.imread(str(mask))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(np.float64)
    thr = arr.max() / 2.0 if arr.max() > 0 else 0.5
    return arr > thr


def score_method(
    mask_path: Union[str, Path, np.ndarray],
    gt_path: Union[str, Path],
    pred: Union[str, Path, np.ndarray, Dict[int, np.ndarray], "iio.InstanceSet"],
    min_len_px: int = 6,
    prune_spur_px: int = 3,
    min_overlap_frac: float = 0.3,
    max_dist_px: float = 6.0,
    coverage_thr: float = 0.8,
    purity_thr: float = 0.8,
) -> dict:
    """Load the input mask, build common fragments once, score `pred`.

    `gt_path` is loaded via `instance_io.load_gt` (label image or centerline
    JSON; auto-prefers a sibling `gt_multilabel.npz` for overlap-aware GT).
    `pred` may be a path (file or pipeline run directory, resolved via
    `instance_io.load_pred_instances`) or an already-loaded in-memory
    representation (`InstanceSet`, label image, multi-label stack, or dict).

    Returns pairwise scores, prefixed shared fragment recovery, prefixed
    representation-dependent pixel diagnostics, and fragment coverage.
    Foreground coverage is relative to the original (possibly thick) input
    mask, while skeleton coverage is relative to the spur-pruned skeleton
    actually used to construct fragments.
    """
    mask = _load_mask(mask_path)
    frag = common_fragments(mask, min_len_px=min_len_px, prune_spur_px=prune_spur_px)

    gt_iset, _gt_meta = iio.load_gt(Path(gt_path), shape=mask.shape)

    if isinstance(pred, (str, Path)):
        pred_iset: Instances = iio.load_pred_instances(Path(pred))
    else:
        pred_iset = pred

    gt_assign = assign_fragments(frag, gt_iset, min_overlap_frac, max_dist_px)
    pred_assign = assign_fragments(frag, pred_iset, min_overlap_frac, max_dist_px)

    foreground_pixels = int(mask.sum())
    common_fragment_pixels = int((frag != 0).sum())
    from skimage.morphology import skeletonize
    fragment_skeleton = _prune_spurs(skeletonize(mask), prune_spur_px)
    skeleton_pixels = int(fragment_skeleton.sum())
    out: dict = {
        "n_common_fragments": int(frag.max()),
        "n_foreground_pixels": foreground_pixels,
        "n_skeleton_pixels": skeleton_pixels,
        "n_common_fragment_pixels": common_fragment_pixels,
        "common_fragment_foreground_pixel_coverage": (
            common_fragment_pixels / foreground_pixels if foreground_pixels else float("nan")
        ),
        "common_fragment_skeleton_coverage": (
            common_fragment_pixels / skeleton_pixels if skeleton_pixels else float("nan")
        ),
    }
    out.update(pairwise_scores(gt_assign, pred_assign))
    fragment_recovery = fragment_instance_recovery(
        gt_assign, pred_assign, coverage_thr, purity_thr
    )
    pixel_recovery = instance_recovery(gt_iset, pred_iset, coverage_thr, purity_thr)
    out.update({f"fragment_recovery_{key}": value
                for key, value in fragment_recovery.items()})
    out.update({f"pixel_recovery_{key}": value
                for key, value in pixel_recovery.items()})
    return out


# ── 6. CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Method-independent fragment-clustering + instance-recovery scoring"
    )
    ap.add_argument("--mask", required=True, help="binary centreline mask (png/tif)")
    ap.add_argument("--gt", required=True, help="ground-truth label image / centerline JSON")
    ap.add_argument(
        "--pred", required=True,
        help="predicted label image (.tif) OR a pipeline run directory "
             "(resolved via instance_io.load_pred_instances)",
    )
    ap.add_argument("--json", help="write scores as JSON to this path")
    ap.add_argument("--min-len-px", type=int, default=6, dest="min_len_px")
    ap.add_argument("--prune-spur-px", type=int, default=3, dest="prune_spur_px")
    ap.add_argument("--min-overlap-frac", type=float, default=0.3, dest="min_overlap_frac")
    ap.add_argument("--max-dist-px", type=float, default=6.0, dest="max_dist_px")
    ap.add_argument("--coverage-thr", type=float, default=0.8, dest="coverage_thr")
    ap.add_argument("--purity-thr", type=float, default=0.8, dest="purity_thr")
    args = ap.parse_args()

    scores = score_method(
        args.mask, args.gt, args.pred,
        min_len_px=args.min_len_px,
        prune_spur_px=args.prune_spur_px,
        min_overlap_frac=args.min_overlap_frac,
        max_dist_px=args.max_dist_px,
        coverage_thr=args.coverage_thr,
        purity_thr=args.purity_thr,
    )
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.json:
        Path(args.json).write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
