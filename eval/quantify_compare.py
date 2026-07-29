"""Downstream-sufficiency check: does the pipeline reproduce the SCIENTIFIC
measurements (count, length, orientation) of the ground truth, despite the
remaining grouping errors?

The reconnect F1 measures grouping; this measures what the science actually
uses. For each case it compares EXTRACTED vs GT instances on:
  - count               n_pred vs n_gt
  - total length        sum of per-instance skeleton arc-length (px)
  - per-filament length  mean / median + KS distance of the distributions
  - orientation         length-weighted angle histogram (similarity)
  - width               area/length per instance (mean)
Overlaid histograms (length, orientation) are written per case.

Verdict logic: grouping errors that SPLIT/MERGE filaments of similar orientation
preserve total length and the orientation distribution but bias count and
per-filament length; DROPPED filaments / wrong bridges break total length too.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
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
MIN_LEN = 6   # ignore sub-this-px specks


def instance_stats(iset, min_len=MIN_LEN):
    """Per-instance (length_px, orientation_deg, width_px)."""
    out = []
    for m in iset.masks.values():
        area = int(m.sum())
        if area < min_len:
            continue
        sk = skeletonize(m)
        L = int(sk.sum())
        if L < min_len:
            continue
        pts = np.argwhere(m).astype(np.float32)
        pts -= pts.mean(0)
        cov = np.cov(pts.T)
        ang = 0.0
        if np.all(np.isfinite(cov)):
            w, v = np.linalg.eigh(cov)
            mj = v[:, int(np.argmax(w))]
            ang = np.degrees(np.arctan2(mj[0], mj[1])) % 180.0
        out.append((float(L), float(ang), area / max(1, L)))
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 3), np.float32)


def ks(a, b):
    """KS distance between two 1-D samples (0=identical, 1=disjoint)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    xs = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def orient_similarity(gt, pr, gtw, prw, bins=18):
    """Length-weighted orientation histogram intersection (1=identical)."""
    edges = np.linspace(0, 180, bins + 1)
    hg, _ = np.histogram(gt, bins=edges, weights=gtw)
    hp, _ = np.histogram(pr, bins=edges, weights=prw)
    hg = hg / max(1e-9, hg.sum()); hp = hp / max(1e-9, hp.sum())
    return float(np.minimum(hg, hp).sum())


def save_hists(name, G, P):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    le = np.linspace(0, max(G[:, 0].max(), P[:, 0].max()) + 1, 25)
    ax[0].hist(G[:, 0], le, alpha=0.55, label=f"GT (n={len(G)})", color="tab:blue")
    ax[0].hist(P[:, 0], le, alpha=0.55, label=f"extracted (n={len(P)})", color="tab:orange")
    ax[0].set_title(f"{name}: length (px)"); ax[0].set_xlabel("skeleton length px"); ax[0].legend()
    oe = np.linspace(0, 180, 19)
    ax[1].hist(G[:, 1], oe, weights=G[:, 0], alpha=0.55, label="GT", color="tab:blue", density=True)
    ax[1].hist(P[:, 1], oe, weights=P[:, 0], alpha=0.55, label="extracted", color="tab:orange", density=True)
    ax[1].set_title(f"{name}: orientation (length-weighted)"); ax[1].set_xlabel("deg"); ax[1].legend()
    fig.tight_layout(); fig.savefig(str(OUT / f"{name}_quant.png"), dpi=110); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'case':<11} {'n GT':>5} {'n ex':>5} {'cnt':>5} | {'totL GT':>8} {'totL ex':>8} {'ratio':>6} | "
          f"{'medL GT':>7} {'medL ex':>7} {'KS':>5} | {'orientOv':>7}")
    agg = []
    for name, run, gtp in CASES:
        pred = iio.load_pred_instances(ROOT / run)
        gt, _ = iio.load_gt(ROOT / gtp, shape=pred.shape)
        G = instance_stats(gt); P = instance_stats(pred)
        save_hists(name, G, P)
        tg, tp = G[:, 0].sum(), P[:, 0].sum()
        row = dict(name=name, ng=len(G), npred=len(P),
                   cnt=len(P) / max(1, len(G)),
                   totg=tg, totp=tp, totr=tp / max(1e-9, tg),
                   medg=np.median(G[:, 0]), medp=np.median(P[:, 0]),
                   ks=ks(G[:, 0], P[:, 0]),
                   osim=orient_similarity(G[:, 1], P[:, 1], G[:, 0], P[:, 0]))
        agg.append(row)
        print(f"{name:<11} {row['ng']:>5} {row['npred']:>5} {row['cnt']:>5.2f} | "
              f"{tg:>8.0f} {tp:>8.0f} {row['totr']:>6.2f} | "
              f"{row['medg']:>7.0f} {row['medp']:>7.0f} {row['ks']:>5.2f} | {row['osim']:>7.2f}")
    m = lambda k: np.mean([r[k] for r in agg])
    print(f"\nMEAN  count ratio (ex/GT) = {m('cnt'):.2f}   total-length ratio = {m('totr'):.2f}   "
          f"length KS = {m('ks'):.2f}   orientation-overlap = {m('osim'):.2f}")
    print(f"\nReadout: total-length ratio ~1 and orientation-overlap ~1 => those measurements are SUFFICIENT")
    print(f"         count ratio >1 / high length-KS => per-filament count & length biased by splits")
    print(f"\nhistograms -> {OUT}")


if __name__ == "__main__":
    main()
