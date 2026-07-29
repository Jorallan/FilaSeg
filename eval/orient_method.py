"""Orientation distribution: per-instance PCA vs per-pixel local tangent.

Hypothesis: the modest per-instance-PCA overlap (0.74) is a MEASUREMENT artifact
(one angle per instance is corrupted by splits and crossing-fusions), not a real
pipeline failure. Fiber orientation is a LOCAL property — measured per skeleton
pixel (local tangent) it should be near grouping-independent, so GT vs extracted
should agree much better. If so, orientation is already sufficient when measured
locally.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_io as iio   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "quantify_out"
CASES = [
    ("real_crop", "output/audit/real_crop",
     "output/full_pipeline/sem00000_crop512_manual/modified/sem_full_00000_1p66_crop512_manual_multilabel.npz"),
    ("synth_0001", "output/audit/synth_0001", "output/synthetic_val/synth_0001/gt_multilabel.npz"),
    ("synth_0002", "output/audit/synth_0002", "output/synthetic_val/synth_0002/gt_multilabel.npz"),
    ("synth_0003", "output/audit/synth_0003", "output/synthetic_val/synth_0003/gt_multilabel.npz"),
]
R = 6   # local-tangent window radius (px)


def union(iset):
    u = np.zeros(iset.shape, bool)
    for m in iset.masks.values():
        u |= m
    return u


def per_instance_pca(iset):
    angs, wts = [], []
    for m in iset.masks.values():
        sk = skeletonize(m); L = int(sk.sum())
        if L < 6:
            continue
        pts = np.argwhere(m).astype(float); pts -= pts.mean(0)
        w, v = np.linalg.eigh(np.cov(pts.T)); mj = v[:, int(np.argmax(w))]
        angs.append(np.degrees(np.arctan2(mj[0], mj[1])) % 180.0); wts.append(L)
    return np.array(angs), np.array(wts)


def per_pixel_tangent(mask_union):
    sk = skeletonize(mask_union)
    coords = np.argwhere(sk)
    if len(coords) < 3:
        return np.zeros(0)
    tree = cKDTree(coords)
    angs = []
    for p in coords:
        idx = tree.query_ball_point(p, R)
        if len(idx) < 3:
            continue
        nb = coords[idx].astype(float); nb -= nb.mean(0)
        w, v = np.linalg.eigh(np.cov(nb.T)); mj = v[:, int(np.argmax(w))]
        angs.append(np.degrees(np.arctan2(mj[0], mj[1])) % 180.0)
    return np.array(angs)


def overlap(a, wa, b, wb, bins=18):
    e = np.linspace(0, 180, bins + 1)
    ha, _ = np.histogram(a, e, weights=wa); hb, _ = np.histogram(b, e, weights=wb)
    ha = ha / max(1e-9, ha.sum()); hb = hb / max(1e-9, hb.sum())
    return float(np.minimum(ha, hb).sum())


def main():
    print(f"{'case':<11} {'per-instance PCA':>17} {'per-pixel tangent':>18}")
    pi, pp = [], []
    for name, run, gtp in CASES:
        pred = iio.load_pred_instances(ROOT / run)
        gt, _ = iio.load_gt(ROOT / gtp, shape=pred.shape)
        ga, gw = per_instance_pca(gt); pa, pw = per_instance_pca(pred)
        o_inst = overlap(ga, gw, pa, pw)
        gpix = per_pixel_tangent(union(gt)); ppix = per_pixel_tangent(union(pred))
        o_pix = overlap(gpix, np.ones_like(gpix), ppix, np.ones_like(ppix))
        pi.append(o_inst); pp.append(o_pix)
        print(f"{name:<11} {o_inst:>17.2f} {o_pix:>18.2f}")
    print(f"\nMEAN        {np.mean(pi):>17.2f} {np.mean(pp):>18.2f}")
    print("\n(overlap 1.0 = identical orientation distribution; per-pixel is grouping-independent)")


if __name__ == "__main__":
    main()
