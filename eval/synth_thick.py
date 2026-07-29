"""Thick-bundle / thin-axis synthetic CNT generator.

Promoted from `sandbox/synth_thick.py` into `eval/` on 2026-07-29. This is the
canonical generator for the paper's thick-bundle synthetic dataset: it is the
generator named by `LOCKED_GENERATOR_COMMAND` in `eval/build_manifest.py`, and
`input/synthetic_thick/` -- which backs `fig_width_sweep` and
`fig_clean_vs_degraded` in the manuscript -- cannot be regenerated without it.
It extends `eval/synth_generator.py` rather than replacing it.

Motivation (2026-07-20, user request for physical accuracy)
-----------------------------------------------------------
Real CNT bundles are *thick* in the SEM, and the U-Net upstream of FilaSeg does
not reproduce the full bundle silhouette -- it reconstructs the bundle *axis*,
so the mask FilaSeg actually consumes is a thin (1-3 px) centreline ribbon, not a
filament-width band. The original `eval/synth_generator.py` thresholds a
filament-width ridge, giving masks roughly as wide as the stroke; that conflates
the SEM appearance with the mask the pipeline sees.

This generator fixes that:
  * bundles are rendered THICK for the SEM (width_range default 7-16 px);
  * areal density is defined and measured on those thick bundles (target
    foreground-coverage fraction, filaments added until reached);
  * the pipeline INPUT mask is a THIN axis of configurable width w in {1,2,3} px,
    produced by degrading a thin-axis ridge with the SAME realistic UNet-like
    degradation model (content-correlated faint-spot breaks, hard cuts, crossing
    occlusion, false positives, roughening) as production;
  * for a width sensitivity sweep, all three widths (w=1,2,3) are written for
    every sample (mask_w1.png / mask_w2.png / mask_w3.png) from the SAME
    degradation draw, so they differ ONLY in axis thickness.

Ground truth (`gt_labels.tif`, `gt_multilabel.npz`) is the THICK bundle geometry
-- that is the physical object FilaSeg must recover. `mask_clean.png` is the 1px
zero-degradation axis (topology-only reference), unchanged in spirit.

Output layout is run_batch.py-compatible:
  <out>/<density>/synth_XXXX/{mask_w1.png,mask_w2.png,mask_w3.png,mask_clean.png,
                              sem.png,gt_labels.tif,gt_multilabel.npz,gt_meta.json,
                              preview.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path

import numpy as np

# Import the production rendering machinery (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import synth_generator as G  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402


def _smooth_path_uniform(shape, rng, curviness, min_len, max_len,
                         curve_smooth_px=20.0, edge_margin=0.30):
    """Correlated-curvature centreline anchored uniformly across the frame.

    Unlike the production walk (which starts at a random point and truncates at
    the first frame exit -- so edge/corner filaments are discarded and survivors
    cluster centrally), this anchors the path at a uniform point in a slightly
    enlarged frame and extends the walk in BOTH directions from that anchor, then
    keeps the longest contiguous in-frame run. Filaments crossing the boundary
    are kept (their visible portion), so coverage is spatially uniform including
    corners, like a real micrograph window into a larger mat.
    """
    H, W = shape
    step = 2.0
    length = float(rng.uniform(min_len, max_len))
    n = max(4, int(length / step))
    turns = rng.normal(0.0, curviness, size=n).astype(np.float32)
    sigma = curve_smooth_px / step
    if sigma >= 0.5:
        turns = gaussian_filter1d(turns, sigma=sigma, mode="nearest")
    headings = float(rng.uniform(0, 2 * np.pi)) + np.cumsum(turns)
    dx = step * np.cos(headings)
    dy = step * np.sin(headings)
    # cumulative positions relative to the path start
    px = np.concatenate(([0.0], np.cumsum(dx)))
    py = np.concatenate(([0.0], np.cumsum(dy)))
    # anchor: a random point along the path placed uniformly over the (enlarged)
    # frame, so the filament extends both ways from an unbiased location.
    a = int(rng.integers(0, px.size))
    ax = float(rng.uniform(-edge_margin * W, (1 + edge_margin) * W))
    ay = float(rng.uniform(-edge_margin * H, (1 + edge_margin) * H))
    xs = ax + (px - px[a])
    ys = ay + (py - py[a])
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    idx = np.flatnonzero(inside)
    if idx.size < 4:
        return np.empty((0, 2), dtype=np.float32)
    # longest contiguous in-frame run
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(groups, key=len)
    if run.size < 4:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack([xs[run], ys[run]], axis=1).astype(np.float32)


def place_to_coverage(shape, rng, target_cov, width_range, bright_range,
                      curviness, len_frac, break_depth, curve_smooth_px,
                      max_fils=400):
    """Place thick filaments until union thick-stroke coverage >= target_cov."""
    H, W = shape
    diag = float(np.hypot(H, W))
    fils = []
    cover = np.zeros(shape, dtype=bool)
    attempts = 0
    while cover.mean() < target_cov and len(fils) < max_fils and attempts < max_fils * 3:
        attempts += 1
        pts = _smooth_path_uniform(shape, rng, curviness,
                                   min_len=len_frac[0] * diag, max_len=len_frac[1] * diag,
                                   curve_smooth_px=curve_smooth_px)
        if pts.shape[0] < 4:
            continue
        width = float(rng.uniform(*width_range))
        bright = float(rng.uniform(*bright_range))
        lm = G._longitudinal_modulation(pts.shape[0], rng, break_depth)
        fil = G.Filament(len(fils) + 1, pts, width, bright, lm)
        s = np.zeros(shape, dtype=bool)
        G._stamp_along(s, fil, None, combine="max")
        cover |= (s > 0)
        fils.append(fil)
    return fils, float(cover.mean())


def thin_axis_ridge(shape, fils, w):
    """Ridge with each bundle stamped at a THIN axis width `w` (with long. mod.)."""
    ridge = np.zeros(shape, dtype=np.float32)
    for fil in fils:
        thin = _dc_replace(fil, width=float(w))
        vals = fil.brightness * fil.long_mod
        G._stamp_along(ridge, thin, vals, combine="max", soft_edge_px=0.6)
    return ridge


def degrade_axis(shape, fils, gt_lab, w, seed, params):
    """Degrade a thin-axis ridge at width w with a fixed per-width seed."""
    rng = np.random.default_rng(seed)  # same seed across widths -> comparable
    ridge = thin_axis_ridge(shape, fils, w)
    return G.degrade_to_mask(
        ridge, fils, gt_lab, rng,
        mask_thr=params["mask_thr"], edge_jagged=params["edge_jagged"],
        break_prob=params["break_prob"], false_pos=params["false_pos"],
        occlusion_prob=params["occlusion_prob"],
    )


def generate_sample(shape, seed, target_cov, params):
    rng = np.random.default_rng(seed)
    fils, cov = place_to_coverage(
        shape, rng, target_cov,
        width_range=params["width_range"], bright_range=params["bright_range"],
        curviness=params["curviness"], len_frac=params["len_frac"],
        break_depth=params["break_depth"], curve_smooth_px=params["curve_smooth_px"],
    )
    gt_lab = G.render_gt_labels(shape, fils)          # THICK bundle GT
    ridge = G.render_ridge(shape, fils)               # THICK ridge -> SEM
    sem = G.make_sem(ridge, rng, params["bg_level"], params["bg_amp"],
                     params["grain"], params["psf"])
    mask_clean = G.render_clean_skeleton_mask(shape, fils)  # 1px axis
    # Thin degraded masks at w=1,2,3 from a fixed degradation seed (comparable).
    deg_seed = int(seed) * 1000 + 7
    masks = {w: degrade_axis(shape, fils, gt_lab, w, deg_seed, params)
             for w in (1, 2, 3)}
    return dict(filaments=fils, coverage=cov, gt_labels=gt_lab, sem=sem,
                mask_clean=mask_clean, masks=masks)


def save_sample(out_dir, sample, params, seed, target_cov):
    from skimage import io as skio
    import tifffile
    out_dir.mkdir(parents=True, exist_ok=True)
    for w, m in sample["masks"].items():
        skio.imsave(str(out_dir / f"mask_w{w}.png"),
                    (m.astype(np.uint8) * 255), check_contrast=False)
    # `mask.png` default = the chosen production width (w=2) for convenience.
    skio.imsave(str(out_dir / "mask.png"),
                (sample["masks"][2].astype(np.uint8) * 255), check_contrast=False)
    skio.imsave(str(out_dir / "mask_clean.png"),
                (sample["mask_clean"].astype(np.uint8) * 255), check_contrast=False)
    skio.imsave(str(out_dir / "sem.png"),
                (np.clip(sample["sem"], 0, 1) * 255).astype(np.uint8), check_contrast=False)
    tifffile.imwrite(str(out_dir / "gt_labels.tif"), sample["gt_labels"].astype(np.uint16))
    gt_masks = G.render_gt_instance_masks(sample["gt_labels"].shape, sample["filaments"])
    G.save_gt_multilabel(out_dir / "gt_multilabel.npz", gt_masks, sample["gt_labels"].shape)
    meta = {
        "seed": int(seed),
        "n_filaments": int(len(sample["filaments"])),
        "target_coverage": float(target_cov),
        "measured_coverage": round(float(sample["coverage"]), 4),
        "axis_widths_px": [1, 2, 3],
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
    }
    meta["params"]["um_per_px"] = G.REF_UM_PER_PX
    (out_dir / "gt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if params.get("preview"):
        gt_c = G._color_labels(sample["gt_labels"])
        def gray3(a): return np.repeat((np.clip(a, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)
        def bin3(a): return np.repeat((a[..., None].astype(np.uint8) * 255), 3, axis=2)
        gap = np.full((gt_c.shape[0], 4, 3), 64, dtype=np.uint8)
        combo = np.concatenate(
            [gt_c, gap, gray3(sample["sem"]), gap, bin3(sample["masks"][2]),
             gap, bin3(sample["mask_clean"])], axis=1)
        skio.imsave(str(out_dir / "preview.png"), combo, check_contrast=False)


def thick_params():
    p = G.default_params()
    p.update(dict(
        width_range=(7.0, 16.0),   # THICK bundles (px at ref scale)
        occlusion_prob=0.15,       # thin axes are fragile; slightly gentler
        false_pos=2.5,
        preview=True,
    ))
    return p


def main():
    ap = argparse.ArgumentParser(description="Thick-bundle/thin-axis synthetic generator")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--coverages", default="0.20,0.30,0.40,0.50,0.60",
                    help="comma list of target areal coverages")
    ap.add_argument("--n", type=int, default=4, help="number of samples per coverage")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first sample index; use with --n to append a predefined extension without regenerating earlier scenes")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed0", type=int, default=20260720)
    args = ap.parse_args()
    if args.n < 1 or args.start_index < 0:
        ap.error("--n must be positive and --start-index must be non-negative")

    shape = (args.size, args.size)
    params = thick_params()
    covs = [float(x) for x in args.coverages.split(",")]
    root = Path(args.out)
    for cov in covs:
        pct = int(round(cov * 100))
        cdir = root / f"cov{pct}"
        print(f"=== coverage target {pct}% -> {cdir} ===", flush=True)
        for i in range(args.start_index, args.start_index + args.n):
            seed = args.seed0 + pct * 100 + i
            s = generate_sample(shape, seed, cov, params)
            sd = cdir / f"synth_{i:04d}"
            save_sample(sd, s, params, seed, cov)
            print(f"  {sd.name}: {len(s['filaments'])} bundles, "
                  f"measured cov={s['coverage']*100:.1f}%", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
