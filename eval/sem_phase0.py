"""SEM Phase 0 — does SEM evidence separate good bridges from bad ones?

The kill-switch experiment before building any SEM decision logic. It enumerates
the *geometrically-plausible* candidate bridges between fragment endpoints (the
ones gap-bridge / reconnect would accept: close, facing, collinear), labels each
by ground truth (same true filament = good, different = bad), and reports how
well each feature separates good from bad (AUC).

The point: on this geometrically-plausible set, GEOMETRY features should score
~0.5 AUC (they can't separate the residual — that's why these are the hard
cases). If SEM features score notably > 0.5, SEM is the missing signal and the
full build is justified. If not, stop.

Works for both pipelines — point --run at a tiled run or a skeleton run; the
fragments are that run's stage-1/2 pieces.

Usage
-----
    python eval/sem_phase0.py --run <run> --sem <sem.png> --gt <gt> [--max-dist 25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import instance_io as iio          # noqa: E402
from sem_features import sample_bridge_sem, SEM_FEATURE_NAMES  # noqa: E402


def _fragment_tips(frag: np.ndarray):
    """Two extreme endpoints + inward unit tangents from the fragment's PCA axis."""
    pts = np.argwhere(frag)
    if pts.shape[0] < 2:
        return []
    ctr = pts.mean(axis=0)
    X = (pts - ctr).astype(np.float64)
    w, v = np.linalg.eigh((X.T @ X) / max(1, X.shape[0] - 1))
    axis = v[:, int(np.argmax(w))]
    proj = X @ axis
    a = pts[int(np.argmin(proj))].astype(np.float64)
    b = pts[int(np.argmax(proj))].astype(np.float64)
    ab = b - a
    inward_a = axis if float(ab @ axis) > 0 else -axis   # a -> b
    return [(a, inward_a), (b, -inward_a)]


def _nearest_label_field(lab: np.ndarray):
    from scipy.ndimage import distance_transform_edt
    nz = lab != 0
    if nz.any() and not nz.all():
        dist, (ir, ic) = distance_transform_edt(~nz, return_indices=True)
        return lab[ir, ic], dist
    return lab.copy(), np.zeros(lab.shape)


def _label_at(field, dist, r, c, max_dist):
    rr = int(min(max(round(r), 0), field.shape[0] - 1))
    cc = int(min(max(round(c), 0), field.shape[1] - 1))
    return int(field[rr, cc]) if dist[rr, cc] <= max_dist else 0


def _auc(good: np.ndarray, bad: np.ndarray) -> float:
    """Rank-based AUC = P(feature(good) > feature(bad)); 0.5 = no separation."""
    if good.size == 0 or bad.size == 0:
        return float("nan")
    allv = np.concatenate([good, bad])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    rg = ranks[:good.size].sum()
    return float((rg - good.size * (good.size + 1) / 2.0) / (good.size * bad.size))


def main() -> int:
    ap = argparse.ArgumentParser(description="SEM Phase 0 separation analysis")
    ap.add_argument("--run", required=True)
    ap.add_argument("--sem", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max-dist", type=float, default=25.0, dest="max_dist")
    ap.add_argument("--fwd-thr", type=float, default=0.7, dest="fwd_thr")
    ap.add_argument("--opp-thr", type=float, default=-0.7, dest="opp_thr")
    ap.add_argument("--min-area", type=int, default=15, dest="min_area")
    ap.add_argument("--out", default=None, help="write the AUC results as JSON")
    args = ap.parse_args()

    from skimage import io as skio
    sem = skio.imread(str(Path(args.sem)))
    if sem.ndim == 3:
        sem = sem[..., :3].mean(axis=2)
    sem = sem.astype(np.float32)
    if sem.max() > 1.5:
        sem /= 255.0
    gt, _ = iio.load_gt(Path(args.gt), shape=sem.shape)
    gt_lab = gt.label_image()
    gfield, gdist = _nearest_label_field(gt_lab)
    frags = iio.load_fragments(Path(args.run), min_area=args.min_area)

    tips = []   # (frag_idx, end_rc, inward_tan)
    for k, f in enumerate(frags):
        for end, inn in _fragment_tips(f):
            tips.append((k, end, inn))

    geom_names = ("dist", "forward_min", "opposition_pos")
    rows = {nm: [] for nm in geom_names + SEM_FEATURE_NAMES}
    labels = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            if tips[i][0] == tips[j][0]:
                continue
            ei, ej = tips[i][1], tips[j][1]
            d = float(np.hypot(*(ei - ej)))
            if d < 1e-6 or d > args.max_dist:
                continue
            u = (ej - ei) / d
            fwd_i = float((-tips[i][2]) @ u)
            fwd_j = float((-tips[j][2]) @ (-u))
            opp = float(tips[i][2] @ tips[j][2])
            # geometrically PLAUSIBLE candidates only (what the gates pass)
            if fwd_i < args.fwd_thr or fwd_j < args.fwd_thr or opp > args.opp_thr:
                continue
            gi = _label_at(gfield, gdist, ei[0], ei[1], 8.0)
            gj = _label_at(gfield, gdist, ej[0], ej[1], 8.0)
            if gi == 0 or gj == 0:
                continue
            labels.append(1 if gi == gj else 0)
            rows["dist"].append(d)
            rows["forward_min"].append(min(fwd_i, fwd_j))
            rows["opposition_pos"].append(-opp)
            sem_f = sample_bridge_sem(sem, ei, ej)
            for nm in SEM_FEATURE_NAMES:
                rows[nm].append(sem_f[nm])

    labels = np.asarray(labels)
    n_good = int(labels.sum())
    n_bad = int((labels == 0).sum())
    print(f"run: {Path(args.run).name}")
    print(f"plausible candidates: {labels.size}  (good={n_good}, bad={n_bad})\n")
    if n_good == 0 or n_bad == 0:
        print("  not enough of both classes to compute AUC")
        return 0

    def block(title, names):
        print(title)
        print(f"  {'feature':<20}{'AUC':>7}{'|AUC-0.5|':>11}  separation")
        scored = {}
        for nm in names:
            v = np.asarray(rows[nm], dtype=np.float64)
            scored[nm] = _auc(v[labels == 1], v[labels == 0])
        for nm, auc in sorted(scored.items(), key=lambda x: -abs(x[1] - 0.5)):
            strength = abs(auc - 0.5)
            bar = "strong" if strength > 0.20 else "some" if strength > 0.10 else "weak"
            print(f"  {nm:<20}{auc:>7.3f}{strength:>11.3f}  {bar}")
        print()
        return {k: round(float(v), 4) for k, v in scored.items()}

    geom_auc = block("GEOMETRY (expected ~0.5 on the plausible set):", geom_names)
    sem_auc = block("SEM (>0.5 = the missing signal):", SEM_FEATURE_NAMES)
    print("AUC>0.5 -> higher feature value favors a TRUE join; <0.5 -> lower does.")

    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "run": Path(args.run).name, "sem": str(args.sem), "gt": str(args.gt),
            "max_dist": args.max_dist, "n_good": n_good, "n_bad": n_bad,
            "geometry_auc": geom_auc, "sem_auc": sem_auc,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
