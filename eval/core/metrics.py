"""Instance-grouping metrics for FilaSeg reconnect evaluation.

Two families of metric, by priority:

1. Fragment-clustering metrics (PRIMARY). After stage 1+2 the image is a set of
   atomic fragments. Ground truth assigns each fragment to a true filament;
   reconnect assigns each to an output id. Comparing those two partitions of the
   *same* fragment set is exactly "did the right filaments emerge", and it is:
     - permutation invariant (immune to id renumbering),
     - decomposable into over-merge vs under-merge,
     - dependent only on grouping GT, not pixel-perfect GT.
   Pairwise precision -> over-merge (false joins); pairwise recall -> under-merge
   (missed joins). VOI splits into the same two directions; split/merge counts
   give the plain-language version.

2. Pixel Panoptic Quality (SECONDARY). Standard matched-instance PQ/F1 at IoU
   thresholds. Useful when GT has real area, but harsh for 1-px-wide filaments,
   so treat it as a cross-check, not the objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from instance_io import InstanceSet


# ── fragment -> label assignment ──────────────────────────────────────────


def assign_fragments_to_labels(
    fragments: List[np.ndarray],
    lab: np.ndarray,
    min_overlap_frac: float = 0.2,
    max_dist_px: float = 12.0,
) -> np.ndarray:
    """Assign each fragment one label from `lab` (0 = unassigned).

    A fragment first tries direct overlap: the most-covering non-zero label, if
    it covers >= min_overlap_frac of the fragment. Failing that (thin GT, small
    gaps), it falls back to the nearest non-zero label within max_dist_px via a
    Voronoi nearest-label field. Returns an int array aligned to `fragments`.
    """
    from scipy.ndimage import distance_transform_edt

    out = np.zeros(len(fragments), dtype=np.int64)

    # Nearest-label field: for every pixel, the label of the closest non-zero
    # pixel in `lab`, plus the distance to it.
    nz = lab != 0
    if nz.any() and not nz.all():
        dist, (ir, ic) = distance_transform_edt(~nz, return_indices=True)
        nearest = lab[ir, ic]
    else:
        dist = np.zeros(lab.shape, dtype=np.float64)
        nearest = lab.copy()

    for k, frag in enumerate(fragments):
        if not frag.any():
            continue
        vals = lab[frag]
        nzvals = vals[vals != 0]
        assigned = 0
        if nzvals.size:
            ids, counts = np.unique(nzvals, return_counts=True)
            top = int(ids[int(np.argmax(counts))])
            if counts.max() >= min_overlap_frac * int(frag.sum()):
                assigned = top
        if assigned == 0:
            # Fall back to nearest label, capped by distance.
            near_vals = nearest[frag]
            near_dist = dist[frag]
            within = near_dist <= max_dist_px
            cand = near_vals[within]
            if cand.size:
                ids, counts = np.unique(cand[cand != 0], return_counts=True)
                if ids.size:
                    assigned = int(ids[int(np.argmax(counts))])
        out[k] = assigned
    return out


def assign_fragments_to_instances(
    fragments: List[np.ndarray],
    iset: InstanceSet,
    min_overlap_frac: float = 0.2,
    max_dist_px: float = 12.0,
) -> np.ndarray:
    """Assign each fragment the instance it overlaps most (0 = unassigned).

    Unlike the label-image variant, this votes against each instance's FULL mask,
    so instances that overlap at crossings (a filament that crosses others still
    owns its whole stroke) attribute fragments correctly. Falls back to nearest
    instance within max_dist_px via the flattened label field.
    """
    from scipy.ndimage import distance_transform_edt

    out = np.zeros(len(fragments), dtype=np.int64)
    ids = sorted(iset.masks)
    if not ids:
        return out
    flat = {i: iset.masks[i].ravel() for i in ids}

    lab = iset.label_image()
    nz = lab != 0
    if nz.any() and not nz.all():
        dist, (ir, ic) = distance_transform_edt(~nz, return_indices=True)
        nearest = lab[ir, ic]
    else:
        dist = np.zeros(lab.shape); nearest = lab.copy()

    for k, frag in enumerate(fragments):
        idx = np.flatnonzero(frag.ravel())
        if idx.size == 0:
            continue
        best_i, best_c = 0, 0
        for i in ids:
            c = int(flat[i][idx].sum())
            if c > best_c:
                best_c, best_i = c, i
        if best_c >= min_overlap_frac * idx.size:
            out[k] = best_i
            continue
        # Nearest-instance fallback.
        near_vals = nearest[frag]
        near_dist = dist[frag]
        cand = near_vals[near_dist <= max_dist_px]
        cand = cand[cand != 0]
        if cand.size:
            vids, counts = np.unique(cand, return_counts=True)
            out[k] = int(vids[int(np.argmax(counts))])
    return out


# ── contingency-table helpers ─────────────────────────────────────────────


def _contingency(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense contingency matrix n[i, j] = #fragments with gt=i and pred=j.

    Inputs are integer label vectors over the same fragment set (already
    filtered to fragments that have a valid gt label). Returns (n, a, b) with
    row sums a (gt cluster sizes) and col sums b (pred cluster sizes).
    """
    gids, gi = np.unique(gt, return_inverse=True)
    pids, pj = np.unique(pred, return_inverse=True)
    n = np.zeros((gids.size, pids.size), dtype=np.int64)
    np.add.at(n, (gi, pj), 1)
    return n, n.sum(axis=1), n.sum(axis=0)


def _comb2(x: np.ndarray | int) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


@dataclass
class ClusteringResult:
    n_fragments_with_gt: int
    n_spurious_fragments: int      # fragments with no GT (false-positive pieces)
    n_gt_instances: int
    n_pred_instances: int
    pairwise_precision: float      # high -> few false merges (over-merge low)
    pairwise_recall: float         # high -> few missed merges (under-merge low)
    pairwise_f1: float
    adjusted_rand: float
    voi: float                     # total variation of information (bits)
    voi_split: float               # over-seg component  H(pred|gt)
    voi_merge: float               # under-seg component H(gt|pred)
    n_gt_split: int                # GT filaments broken across >1 pred id
    excess_splits: int             # sum over GT of (pred ids - 1)
    n_pred_overmerged: int         # pred ids covering >1 GT filament
    excess_merges: int             # sum over pred of (gt ids - 1)

    def summary(self) -> str:
        return (
            f"fragments(GT)={self.n_fragments_with_gt} spurious={self.n_spurious_fragments}\n"
            f"instances: GT={self.n_gt_instances}  pred={self.n_pred_instances}\n"
            f"pairwise  P={self.pairwise_precision:.3f} (over-merge)  "
            f"R={self.pairwise_recall:.3f} (under-merge)  F1={self.pairwise_f1:.3f}\n"
            f"ARI={self.adjusted_rand:.3f}  "
            f"VOI={self.voi:.3f} (split={self.voi_split:.3f} merge={self.voi_merge:.3f})\n"
            f"splits: {self.n_gt_split} GT filaments fragmented "
            f"(+{self.excess_splits} extra ids)\n"
            f"merges: {self.n_pred_overmerged} pred ids over-merged "
            f"(+{self.excess_merges} extra GT)"
        )


def fragment_clustering_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> ClusteringResult:
    """Compare two partitions of the same fragment set.

    gt_labels / pred_labels are aligned int vectors; gt==0 marks a fragment
    with no ground truth (counted as spurious, excluded from the comparison).
    """
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)
    valid = gt_labels != 0
    n_spurious = int((~valid).sum())
    gt = gt_labels[valid]
    pred = pred_labels[valid]
    N = gt.size

    n_gt_instances = int(np.unique(gt).size) if N else 0
    n_pred_instances = int(np.unique(pred_labels[pred_labels != 0]).size)

    if N < 2:
        return ClusteringResult(
            N, n_spurious, n_gt_instances, n_pred_instances,
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0,
        )

    n, a, b = _contingency(gt, pred)

    # Pairwise (Rand-style) counts.
    tp = float(_comb2(n).sum())
    same_pred = float(_comb2(b).sum())
    same_gt = float(_comb2(a).sum())
    total_pairs = float(_comb2(N))
    fp = same_pred - tp
    fn = same_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Adjusted Rand Index.
    expected = same_gt * same_pred / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (same_gt + same_pred)
    ari = (tp - expected) / (max_index - expected) if (max_index - expected) != 0 else 1.0

    # Variation of information (bits), split into over/under-seg directions.
    Nf = float(N)
    pa = a / Nf
    pb = b / Nf
    pij = n / Nf
    h_gt = -float(np.sum(pa[pa > 0] * np.log2(pa[pa > 0])))
    h_pred = -float(np.sum(pb[pb > 0] * np.log2(pb[pb > 0])))
    nz = pij > 0
    mi = float(np.sum(pij[nz] * np.log2(pij[nz] / (pa[:, None] * pb[None, :])[nz])))
    voi_merge = h_gt - mi    # H(gt|pred): true filaments mixed into one pred id
    voi_split = h_pred - mi  # H(pred|gt): one filament scattered over pred ids
    voi = max(0.0, voi_split) + max(0.0, voi_merge)

    # Plain-language split/merge counts.
    n_gt_split = int(np.sum((n > 0).sum(axis=1) > 1))
    excess_splits = int(np.sum(np.maximum(0, (n > 0).sum(axis=1) - 1)))
    n_pred_overmerged = int(np.sum((n > 0).sum(axis=0) > 1))
    excess_merges = int(np.sum(np.maximum(0, (n > 0).sum(axis=0) - 1)))

    return ClusteringResult(
        N, n_spurious, n_gt_instances, n_pred_instances,
        precision, recall, f1, ari, voi, max(0.0, voi_split), max(0.0, voi_merge),
        n_gt_split, excess_splits, n_pred_overmerged, excess_merges,
    )


# ── instance-level recovery (co-primary) ──────────────────────────────────


@dataclass
class InstanceRecoveryResult:
    """Instance-level scorecard, closer to the goal (counting distinct filaments)
    than fragment-pairwise F1. Computed from the same fragment->instance vectors.

    A GT filament is exactly one of:
      well_recovered  its dominant pred id covers >= frac of it AND is >= frac pure
      deleted         its best id is dominated by ANOTHER GT (lost as a distinct
                      instance — a wrong-connect swallowed it)
      fragmented      survives as its own id but split / contaminated below frac
    Plus false_pred: pred ids that are nobody's dominant (spurious extra instances).
    """
    n_gt_present: int
    well_recovered: int
    deleted: int
    fragmented: int
    false_pred: int
    recovery_rate: float

    def summary(self) -> str:
        return (
            f"instances: {self.well_recovered}/{self.n_gt_present} well-recovered "
            f"({self.recovery_rate:.2f})  deleted={self.deleted}  "
            f"fragmented={self.fragmented}  false_pred={self.false_pred}"
        )


def instance_recovery(
    gt_frag: np.ndarray,
    pred_frag: np.ndarray,
    recover_frac: float = 0.8,
) -> InstanceRecoveryResult:
    """Per-instance recovery from aligned fragment->id vectors (0 = unassigned)."""
    cell: Dict[Tuple[int, int], int] = {}
    gt_tot: Dict[int, int] = {}
    pred_tot: Dict[int, int] = {}
    for g, p in zip(gt_frag, pred_frag):
        g, p = int(g), int(p)
        if g and p:
            cell[(g, p)] = cell.get((g, p), 0) + 1
            gt_tot[g] = gt_tot.get(g, 0) + 1
            pred_tot[p] = pred_tot.get(p, 0) + 1

    gts = sorted(gt_tot)
    preds = sorted(pred_tot)
    gt_dom = {g: max((p for (gg, p) in cell if gg == g),
                     key=lambda p: cell[(g, p)], default=0) for g in gts}
    pred_dom = {p: max((g for (g, pp) in cell if pp == p),
                       key=lambda g: cell[(g, p)], default=0) for p in preds}

    well = deleted = fragmented = 0
    dom_preds = set()
    for g in gts:
        p = gt_dom[g]
        dom_preds.add(p)
        recovery = cell[(g, p)] / max(1, gt_tot[g])
        purity = cell[(g, p)] / max(1, pred_tot[p])
        if pred_dom.get(p) != g:
            deleted += 1
        elif recovery >= recover_frac and purity >= recover_frac:
            well += 1
        else:
            fragmented += 1
    false_pred = sum(1 for p in preds if p not in dom_preds)
    n = len(gts)
    return InstanceRecoveryResult(n, well, deleted, fragmented, false_pred,
                                  well / max(1, n))


# ── pixel panoptic quality (secondary) ────────────────────────────────────


def _iou_matrix(pred_lab: np.ndarray, gt_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pids = np.unique(pred_lab[pred_lab != 0])
    gids = np.unique(gt_lab[gt_lab != 0])
    if pids.size == 0 or gids.size == 0:
        return np.zeros((pids.size, gids.size)), pids, gids
    pidx = {int(v): k for k, v in enumerate(pids)}
    gidx = {int(v): k for k, v in enumerate(gids)}
    inter = np.zeros((pids.size, gids.size), dtype=np.int64)
    both = (pred_lab != 0) & (gt_lab != 0)
    pr = np.vectorize(pidx.get)(pred_lab[both])
    gr = np.vectorize(gidx.get)(gt_lab[both])
    np.add.at(inter, (pr, gr), 1)
    p_area = np.array([int((pred_lab == v).sum()) for v in pids])
    g_area = np.array([int((gt_lab == v).sum()) for v in gids])
    union = p_area[:, None] + g_area[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou, pids, gids


@dataclass
class PanopticResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    f1_by_iou: Dict[float, float] = field(default_factory=dict)

    def summary(self) -> str:
        f1s = "  ".join(f"F1@{t:.1f}={v:.3f}" for t, v in sorted(self.f1_by_iou.items()))
        return (
            f"PQ={self.pq:.3f} (SQ={self.sq:.3f} RQ={self.rq:.3f})  "
            f"TP={self.tp} FP={self.fp} FN={self.fn}\n{f1s}"
        )


def panoptic_quality(
    pred: InstanceSet,
    gt: InstanceSet,
    extra_iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
) -> PanopticResult:
    """Standard PQ at IoU>0.5 plus F1 at several IoU thresholds (greedy match)."""
    pred_lab = pred.label_image()
    gt_lab = gt.label_image()
    iou, pids, gids = _iou_matrix(pred_lab, gt_lab)
    n_pred, n_gt = pids.size, gids.size

    # PQ: at IoU > 0.5 the match is unique.
    matched_iou = iou > 0.5
    tp_pairs = np.argwhere(matched_iou)
    tp = int(tp_pairs.shape[0])
    sq = float(iou[matched_iou].mean()) if tp else 0.0
    fp = n_pred - tp
    fn = n_gt - tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    f1_by_iou: Dict[float, float] = {}
    for thr in extra_iou_thresholds:
        # Greedy descending-IoU matching for thresholds <= 0.5.
        order = np.argsort(-iou, axis=None)
        used_p, used_g = set(), set()
        m = 0
        for flat in order:
            i, j = divmod(int(flat), max(1, n_gt))
            if n_gt == 0 or iou[i, j] < thr:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j); m += 1
        tpx = m
        fpx = n_pred - m
        fnx = n_gt - m
        f1_by_iou[thr] = (2 * tpx) / (2 * tpx + fpx + fnx) if (2 * tpx + fpx + fnx) > 0 else 0.0

    return PanopticResult(pq, sq, rq, tp, fp, fn, f1_by_iou)


# ── top-level driver ──────────────────────────────────────────────────────


@dataclass
class EvalReport:
    clustering: ClusteringResult
    panoptic: Optional[PanopticResult]
    n_fragments: int
    gt_kind: str
    instance: Optional[InstanceRecoveryResult] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=== Fragment-clustering (PRIMARY) ===", self.clustering.summary()]
        if self.instance is not None:
            lines += ["", "=== Instance recovery (co-primary) ===", self.instance.summary()]
        if self.panoptic is not None:
            lines += ["", "=== Pixel panoptic (secondary) ===", self.panoptic.summary()]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {x}" for x in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {"gt_kind": self.gt_kind, "n_fragments": self.n_fragments,
             "clustering": self.clustering.__dict__, "notes": self.notes}
        if self.instance is not None:
            d["instance"] = self.instance.__dict__
        if self.panoptic is not None:
            pd = dict(self.panoptic.__dict__)
            pd["f1_by_iou"] = {str(k): v for k, v in pd["f1_by_iou"].items()}
            d["panoptic"] = pd
        return d


def evaluate(
    pred: InstanceSet,
    gt: InstanceSet,
    fragments: List[np.ndarray],
    gt_kind: str = "label",
    min_overlap_frac: float = 0.2,
    max_dist_px: float = 12.0,
    do_panoptic: bool = True,
) -> EvalReport:
    """Run the full measurement on one image."""
    notes: List[str] = []

    # Overlap-aware assignment: vote against each instance's full mask so that
    # crossings (overlapping instances) attribute fragments to the filament they
    # actually run along, not whichever one owns the crossing in a flat image.
    gt_frag = assign_fragments_to_instances(fragments, gt, min_overlap_frac, max_dist_px)
    pred_frag = assign_fragments_to_instances(fragments, pred, min_overlap_frac, max_dist_px)
    clustering = fragment_clustering_metrics(gt_frag, pred_frag)
    instance = instance_recovery(gt_frag, pred_frag)

    panoptic = None
    if do_panoptic:
        if gt_kind == "centerline":
            notes.append(
                "GT is centerline-derived (thick raster); pixel IoU/PQ is "
                "approximate. Trust the fragment-clustering block."
            )
        panoptic = panoptic_quality(pred, gt)

    if pred.n > 3 * max(1, gt.n):
        notes.append(
            f"pred has {pred.n} instances vs GT {gt.n}: strong under-merge "
            "(over-segmentation) signal."
        )

    return EvalReport(clustering, panoptic, len(fragments), gt_kind,
                      instance=instance, notes=notes)
