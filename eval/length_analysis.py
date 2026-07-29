"""Per-filament length: how much of the short-bias is recoverable?

Length is biased short because residual SPLITS chop filaments. Compare the
extracted per-filament length distribution to GT under three groupings:
  current      pipeline output as-is
  +recover     merge split pieces of one filament that have NO third filament in
               the join (the curve/junction over-blocks #2 targets; realizable)
  ceiling      merge ALL same-filament splits (perfect grouping = GT lengths)
Report median length and KS vs GT for each, so we see the realizable gain vs the
geometric ceiling.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.draw import line as skline
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_io as iio   # noqa: E402
import metrics as M         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("real_crop", "output/audit/real_crop",
     "output/full_pipeline/sem00000_crop512_manual/modified/sem_full_00000_1p66_crop512_manual_multilabel.npz"),
    ("synth_0001", "output/audit/synth_0001", "output/synthetic_val/synth_0001/gt_multilabel.npz"),
    ("synth_0002", "output/audit/synth_0002", "output/synthetic_val/synth_0002/gt_multilabel.npz"),
    ("synth_0003", "output/audit/synth_0003", "output/synthetic_val/synth_0003/gt_multilabel.npz"),
]


class UF:
    def __init__(s, it): s.p = {x: x for x in it}
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)


def skel_len(mask):
    return int(skeletonize(mask).sum())


def ks(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    xs = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def lengths(masks):
    return np.array([skel_len(m) for m in masks if int(m.sum()) >= 6 and skel_len(m) >= 6], float)


def main():
    print(f"{'case':<11} | {'medGT':>6} | {'cur med/KS':>12} | {'+rec med/KS':>13} | {'ceil med/KS':>13}")
    agg = defaultdict(list)
    for name, run, gtp in CASES:
        pred = iio.load_pred_instances(ROOT / run)
        gt, _ = iio.load_gt(ROOT / gtp, shape=pred.shape)
        frags = iio.load_fragments(ROOT / run, min_area=15)
        shape = pred.shape
        gf = M.assign_fragments_to_instances(frags, gt)
        pf = M.assign_fragments_to_instances(frags, pred)
        cell = defaultdict(list); pred_gt = defaultdict(lambda: defaultdict(int))
        for k, (g, p) in enumerate(zip(gf, pf)):
            if g and p:
                cell[(int(g), int(p))].append(k); pred_gt[int(p)][int(g)] += 1
        pred_dom = {p: max(d, key=d.get) for p, d in pred_gt.items()}
        # split pieces of each GT = pred ids whose dominant GT is g
        gt_splits = defaultdict(list)
        for p, g in pred_dom.items():
            gt_splits[g].append(p)

        L_gt = lengths(list(gt.masks.values()))
        L_cur = lengths(list(pred.masks.values()))

        # build merge oracles on pred ids
        pred_ids = list(pred.masks)
        uf_rec = UF(pred_ids); uf_ceil = UF(pred_ids)
        for g, ps in gt_splits.items():
            if len(ps) < 2:
                continue
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    p, q = ps[i], ps[j]
                    uf_ceil.union(p, q)
                    mp, mq = pred.masks[p], pred.masks[q]
                    dq = distance_transform_edt(~mq); ap = np.argwhere(mp)
                    pa = ap[int(np.argmin(dq[mp]))]
                    dp = distance_transform_edt(~mp); aq = np.argwhere(mq)
                    qa = aq[int(np.argmin(dp[mq]))]
                    rr, cc = skline(int(pa[0]), int(pa[1]), int(qa[0]), int(qa[1]))
                    band = np.zeros(shape, bool); band[rr, cc] = True
                    band = binary_dilation(band, iterations=4)
                    third = any(int(h) != g and (gt.masks[h] & band).any() for h in gt.masks)
                    if not third:
                        uf_rec.union(p, q)

        def merged_lengths(uf):
            groups = defaultdict(list)
            for p in pred_ids:
                groups[uf.find(p)].append(p)
            ms = []
            for ps in groups.values():
                u = np.zeros(shape, bool)
                for p in ps:
                    u |= pred.masks[p]
                ms.append(u)
            return lengths(ms)

        L_rec = merged_lengths(uf_rec); L_ceil = merged_lengths(uf_ceil)
        mg = np.median(L_gt)
        agg["cur"].append(np.median(L_cur) / mg); agg["rec"].append(np.median(L_rec) / mg)
        agg["ceil"].append(np.median(L_ceil) / mg)
        agg["ks_cur"].append(ks(L_cur, L_gt)); agg["ks_rec"].append(ks(L_rec, L_gt)); agg["ks_ceil"].append(ks(L_ceil, L_gt))
        print(f"{name:<11} | {mg:>6.0f} | {np.median(L_cur):>5.0f}/{ks(L_cur,L_gt):>4.2f} | "
              f"{np.median(L_rec):>6.0f}/{ks(L_rec,L_gt):>4.2f} | {np.median(L_ceil):>6.0f}/{ks(L_ceil,L_gt):>4.2f}")

    m = np.mean
    print(f"\nMEAN median/GT  current={m(agg['cur']):.2f}  +recover={m(agg['rec']):.2f}  ceiling={m(agg['ceil']):.2f}")
    print(f"MEAN length-KS  current={m(agg['ks_cur']):.2f}  +recover={m(agg['ks_rec']):.2f}  ceiling={m(agg['ks_ceil']):.2f}")
    print("\n(median/GT -> 1.0 = unbiased length; KS -> 0 = matching distribution)")


if __name__ == "__main__":
    main()
