"""Synthetic filament data generator for FilaSeg evaluation/calibration.

Produces paired samples that mimic the real pipeline input plus ground truth:

    <out>/<sample_id>/
        mask.png        degraded "UNet-like" binary mask  (pipeline --mask)
        mask_clean.png  thin (1px), zero-degradation true centerline network --
                         no breaks/cuts/occlusion/false-pos/roughening, so it
                         isolates instance-grouping/topology performance from
                         mask-degradation recovery. Same pipeline --mask
                         contract (binary); width is not encoded, estimate it
                         downstream from sem.png or gt_multilabel.npz instead.
        sem.png         synthetic grayscale SEM            (pipeline --background)
        gt_labels.tif   ground-truth instance labels       (eval --gt)
        gt_meta.json    parameters + centerlines (provenance)
        preview.png     GT | mask | mask_clean | SEM side-by-side (with --preview)

Design principle (realism): the mask is NOT corrupted with i.i.d. pixel noise.
Instead each filament gets a synthetic intensity ridge with low-frequency
*longitudinal* modulation; the mask is obtained by thresholding that ridge plus
edge noise. Breaks therefore appear where the filament is faint (content-
correlated, like a real UNet), with realistic gap-length distributions, and
jaggedness/holes emerge from the same field. Crossings, spurious fragments and
explicit cuts are layered on top. Defaults are tuned to SEM08 crops: 512 px,
thin filaments (~3-5 px), ~15% foreground.

Degradation knobs (CLI):
    --break-depth     longitudinal modulation depth -> faint-spot breaks
    --break-prob      chance of an explicit hard cut per filament
    --edge-jagged     boundary noise amplitude -> ragged contours
    --false-pos       expected spurious background blobs per image
    --occlusion-prob  at a crossing, drop the lower-priority filament locally
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from skimage.draw import line as skline, disk as skdisk


# SEM08 reference: all px-size defaults below are tuned for this scale, so at
# the default um/px the generated data matches what the reconnect params expect.
REF_UM_PER_PX = 1.66 / 1536  # ~0.0010807 um/px


# ── filament geometry ─────────────────────────────────────────────────────


@dataclass
class Filament:
    fid: int
    pts: np.ndarray          # (K, 2) centerline as [x=col, y=row]
    width: float
    brightness: float
    long_mod: np.ndarray = field(default=None)  # (K,) per-point intensity 0..1


def _smooth_path(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    curviness: float,
    min_len: float,
    max_len: float,
    curve_smooth_px: float = 20.0,
) -> np.ndarray:
    """A correlated-curvature random walk -> one smooth filament centerline.

    Turn increments are Gaussian-smoothed along the path (correlation length
    `curve_smooth_px`) so bending happens at long wavelengths, like real
    filaments, instead of per-step jitter. The smoothing kernel is normalized,
    so the cumulative heading drift over distances >> curve_smooth_px matches
    the unsmoothed walk: `curviness` keeps its large-scale meaning.
    """
    H, W = shape
    length = float(rng.uniform(min_len, max_len))
    step = 2.0
    n = max(4, int(length / step))
    turns = rng.normal(0.0, curviness, size=n).astype(np.float32)
    sigma = curve_smooth_px / step
    if sigma >= 0.5:
        turns = gaussian_filter1d(turns, sigma=sigma, mode="nearest")
    headings = float(rng.uniform(0, 2 * np.pi)) + np.cumsum(turns)
    xs = float(rng.uniform(0.05 * W, 0.95 * W)) + np.concatenate(
        ([0.0], np.cumsum(step * np.cos(headings))))
    ys = float(rng.uniform(0.05 * H, 0.95 * H)) + np.concatenate(
        ([0.0], np.cumsum(step * np.sin(headings))))
    # Truncate at the first exit from the frame (matches old walk behavior).
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not inside.all():
        k = int(np.argmin(inside))
        xs, ys = xs[:k], ys[:k]
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _longitudinal_modulation(k: int, rng: np.random.Generator, depth: float) -> np.ndarray:
    """Smooth 0..1 intensity profile along a filament; dips -> faint segments."""
    if k < 2:
        return np.ones(max(1, k), dtype=np.float32)
    raw = rng.normal(0.0, 1.0, size=k)
    sigma = max(1.0, k / 8.0)
    smooth = gaussian_filter(raw, sigma=sigma, mode="nearest")
    smooth = (smooth - smooth.min()) / max(1e-6, float(np.ptp(smooth)))
    return (1.0 - depth) + depth * smooth   # in [1-depth, 1]


def place_filaments(
    shape: Tuple[int, int],
    n: int,
    rng: np.random.Generator,
    width_range: Tuple[float, float],
    bright_range: Tuple[float, float],
    curviness: float,
    len_frac: Tuple[float, float],
    break_depth: float,
    curve_smooth_px: float = 20.0,
) -> List[Filament]:
    H, W = shape
    diag = float(np.hypot(H, W))
    out: List[Filament] = []
    for i in range(n):
        pts = _smooth_path(
            shape, rng, curviness,
            min_len=len_frac[0] * diag, max_len=len_frac[1] * diag,
            curve_smooth_px=curve_smooth_px,
        )
        if pts.shape[0] < 4:
            continue
        width = float(rng.uniform(*width_range))
        bright = float(rng.uniform(*bright_range))
        lm = _longitudinal_modulation(pts.shape[0], rng, break_depth)
        out.append(Filament(len(out) + 1, pts, width, bright, lm))
    return out


# ── rendering ─────────────────────────────────────────────────────────────


def _densify_path(
    pts: np.ndarray,
    values: Optional[np.ndarray],
    step: float = 0.35,
):
    """Resample a polyline (and per-point values) at sub-pixel arc-length steps."""
    if pts.shape[0] == 1:
        v = None if values is None else np.asarray(values[:1], dtype=np.float32)
        return pts.astype(np.float32), v
    seg = np.diff(pts.astype(np.float64), axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    total = float(s[-1])
    t = np.linspace(0.0, total, max(2, int(np.ceil(total / step)) + 1))
    xy = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)
    v = None
    if values is not None:
        vals = np.asarray(values, dtype=np.float64)
        if vals.shape[0] != pts.shape[0]:   # tolerate per-segment values
            vals = np.concatenate([vals, vals[-1:]])[: pts.shape[0]]
        v = np.interp(t, s, vals).astype(np.float32)
    return xy.astype(np.float32), v


def _stamp_along(
    canvas: np.ndarray,
    fil: Filament,
    value_per_pt: Optional[np.ndarray],
    combine: str = "max",
    label_value: Optional[int] = None,
    soft_edge_px: float = 0.0,
) -> None:
    """Paint a stroke along a filament's centerline into `canvas` (in place).

    Sub-pixel: pixels are included by true distance to the densified polyline
    (fractional radii work; no integer-disk stair-stepping). With
    `soft_edge_px` > 0 the stroke value ramps linearly across the nominal
    boundary (cover 1 -> 0 over d in [r - soft/2, r + soft/2]), i.e. an
    anti-aliased edge whose 0.5-crossing sits at the same radius as the hard
    binary stroke.
    """
    H, W = canvas.shape
    r = max(0.75, fil.width / 2.0)
    xy, vals = _densify_path(fil.pts, value_per_pt)
    # Candidate pixels: rasterized centerline dilated by ceil(r + soft) + 1.
    core = np.zeros((H, W), dtype=bool)
    core[np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1),
         np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)] = True
    pad = int(np.ceil(r + soft_edge_px)) + 1
    oy, ox = np.ogrid[-pad:pad + 1, -pad:pad + 1]
    cand = binary_dilation(core, structure=(oy * oy + ox * ox) <= pad * pad)
    pr, pc = np.nonzero(cand)
    d, idx = cKDTree(xy).query(
        np.column_stack([pc, pr]).astype(np.float32), k=1, workers=-1)
    if soft_edge_px > 0:
        cover = np.clip(0.5 + (r - d) / soft_edge_px, 0.0, 1.0)
    else:
        cover = (d <= r).astype(np.float32)
    hit = cover > 0
    pr, pc, cover, idx = pr[hit], pc[hit], cover[hit], idx[hit]
    if label_value is not None:
        # First-write-wins keeps earlier filaments visible at crossings.
        empty = canvas[pr, pc] == 0
        canvas[pr[empty], pc[empty]] = label_value
    else:
        v = cover if vals is None else vals[idx] * cover
        if combine == "max":
            canvas[pr, pc] = np.maximum(canvas[pr, pc], v.astype(canvas.dtype))
        else:
            canvas[pr, pc] += v.astype(canvas.dtype)


def render_gt_labels(shape: Tuple[int, int], fils: List[Filament]) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.int32)
    for fil in fils:
        _stamp_along(lab, fil, None, label_value=fil.fid)
    return lab


def render_clean_skeleton_mask(shape: Tuple[int, int], fils: List[Filament]) -> np.ndarray:
    """Thin (1px), topologically-perfect binary mask: the true centerlines only.

    No width, no threshold noise, no breaks, no crossing-occlusion, no false
    positives, no roughening -- every filament is fully connected end-to-end and
    every true crossing is a real intersection in the raster. This isolates
    instance-grouping/topology performance from mask-degradation recovery,
    which `mask.png`'s realistic UNet-like degradation deliberately conflates.
    Width is NOT encoded here; estimate it downstream from `sem.png` or
    `gt_multilabel.npz` (matching the pipeline's own smart-width convention).
    """
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for fil in fils:
        pts = fil.pts
        for i in range(len(pts) - 1):
            r0, c0 = int(round(pts[i, 1])), int(round(pts[i, 0]))
            r1, c1 = int(round(pts[i + 1, 1])), int(round(pts[i + 1, 0]))
            rr, cc = skline(min(max(r0, 0), H - 1), min(max(c0, 0), W - 1),
                            min(max(r1, 0), H - 1), min(max(c1, 0), W - 1))
            mask[rr, cc] = True
    return mask


def render_gt_instance_masks(shape: Tuple[int, int], fils: List[Filament]):
    """Per-filament FULL-stroke masks (overlapping at crossings).

    Unlike the flattened label image, each filament owns its entire stroke, so
    downstream fragment attribution is not corrupted by crossings.
    """
    out = []
    for fil in fils:
        m = np.zeros(shape, dtype=np.int32)
        _stamp_along(m, fil, None, label_value=1)
        if m.any():
            out.append((fil.fid, m > 0))
    return out


def save_gt_multilabel(path: Path, masks, shape: Tuple[int, int]) -> None:
    """Write per-instance masks in the pipeline's CSR multilabel format."""
    ids, indptr, chunks = [], [0], []
    for fid, m in masks:
        idx = np.flatnonzero(m.ravel()).astype(np.uint64)
        ids.append(int(fid))
        chunks.append(idx)
        indptr.append(indptr[-1] + int(idx.size))
    np.savez_compressed(
        str(path),
        shape=np.asarray(shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.uint64),
        indices=np.concatenate(chunks) if chunks else np.array([], dtype=np.uint64),
    )


def render_ridge(shape: Tuple[int, int], fils: List[Filament]) -> np.ndarray:
    """Filament-only intensity field (no background), with longitudinal dips."""
    ridge = np.zeros(shape, dtype=np.float32)
    for fil in fils:
        vals = fil.brightness * fil.long_mod
        _stamp_along(ridge, fil, vals, combine="max", soft_edge_px=1.0)
    return ridge


def make_sem(
    ridge: np.ndarray,
    rng: np.random.Generator,
    bg_level: float,
    bg_amp: float,
    grain: float,
    psf: float,
) -> np.ndarray:
    """Synthetic grayscale SEM in [0,1]: blurred ridge + low-freq bg + grain."""
    H, W = ridge.shape
    low = gaussian_filter(rng.normal(0, 1, size=ridge.shape).astype(np.float32),
                          sigma=max(H, W) / 16.0, mode="reflect")
    low = (low - low.min()) / max(1e-6, float(np.ptp(low)))
    blurred = gaussian_filter(ridge, sigma=psf, mode="reflect")
    sem = bg_level + bg_amp * low + blurred
    sem = sem + rng.normal(0, grain, size=ridge.shape).astype(np.float32)
    return np.clip(sem, 0.0, 1.0)


def degrade_to_mask(
    ridge: np.ndarray,
    fils: List[Filament],
    gt_lab: np.ndarray,
    rng: np.random.Generator,
    mask_thr: float,
    edge_jagged: float,
    break_prob: float,
    false_pos: float,
    occlusion_prob: float,
) -> np.ndarray:
    """Threshold the ridge (+edge noise) into a UNet-like binary mask.

    Content-correlated breaks come for free from faint ridge segments; on top we
    add explicit hard cuts, crossing occlusion, spurious blobs and a light
    morphological roughening.
    """
    H, W = ridge.shape
    field_noise = gaussian_filter(
        rng.normal(0, 1, size=ridge.shape).astype(np.float32), sigma=1.0, mode="reflect"
    )
    mask = (ridge + edge_jagged * field_noise) >= mask_thr

    # Explicit hard cuts: zero a short window along some filaments.
    for fil in fils:
        if rng.random() >= break_prob:
            continue
        k = fil.pts.shape[0]
        if k < 8:
            continue
        cut = int(rng.integers(2, k - 2))
        half = int(rng.integers(2, 5))
        seg = np.zeros(ridge.shape, dtype=bool)
        sub = Filament(fil.fid, fil.pts[max(0, cut - half):cut + half],
                       fil.width + 2.0, 1.0, None)
        _stamp_along(seg, sub, None, combine="max")
        mask &= ~(seg > 0)

    # Crossing occlusion: where >=2 filaments overlap, sometimes drop the
    # lower-priority one's pixels locally (simulates UNet losing the dimmer
    # filament under a brighter crossing one).
    if occlusion_prob > 0:
        cover = np.zeros(ridge.shape, dtype=np.int32)
        for fil in fils:
            s = np.zeros(ridge.shape, dtype=bool)
            _stamp_along(s, fil, None, combine="max")
            cover += s.astype(np.int32)
        crossings = cover >= 2
        if crossings.any():
            from skimage.measure import label as cc_label
            cl = cc_label(crossings, connectivity=2)
            for k in range(1, int(cl.max()) + 1):
                if rng.random() < occlusion_prob:
                    blob = binary_dilation(cl == k, iterations=2)
                    mask &= ~blob

    # Spurious background fragments (false positives).
    n_fp = rng.poisson(false_pos)
    for _ in range(int(n_fp)):
        r = int(rng.integers(0, H)); c = int(rng.integers(0, W))
        rad = int(rng.integers(1, 4))
        length = int(rng.integers(2, 12))
        ang = float(rng.uniform(0, np.pi))
        r1 = int(np.clip(r + length * np.sin(ang), 0, H - 1))
        c1 = int(np.clip(c + length * np.cos(ang), 0, W - 1))
        rr, cc = skline(r, c, r1, c1)
        for pr, pc in zip(rr, cc):
            dr, dc = skdisk((pr, pc), rad, shape=mask.shape)
            mask[dr, dc] = True

    # Light morphological roughening so contours look UNet-ish, not geometric.
    rough = gaussian_filter(rng.normal(0, 1, size=mask.shape).astype(np.float32),
                            sigma=1.2, mode="reflect")
    add = (rough > 1.3) & binary_dilation(mask, iterations=1)
    rem = (rough < -1.3) & mask
    mask = (mask | add) & ~rem
    return mask


# ── sample assembly ───────────────────────────────────────────────────────


def generate_sample(shape, rng, params) -> dict:
    fils = place_filaments(
        shape, params["n_filaments"], rng,
        width_range=params["width_range"], bright_range=params["bright_range"],
        curviness=params["curviness"], len_frac=params["len_frac"],
        break_depth=params["break_depth"],
        curve_smooth_px=params.get("curve_smooth_px", 20.0),
    )
    gt_lab = render_gt_labels(shape, fils)
    ridge = render_ridge(shape, fils)
    sem = make_sem(ridge, rng, params["bg_level"], params["bg_amp"],
                   params["grain"], params["psf"])
    mask = degrade_to_mask(
        ridge, fils, gt_lab, rng,
        mask_thr=params["mask_thr"], edge_jagged=params["edge_jagged"],
        break_prob=params["break_prob"], false_pos=params["false_pos"],
        occlusion_prob=params["occlusion_prob"],
    )
    mask_clean = render_clean_skeleton_mask(shape, fils)
    return {"filaments": fils, "gt_labels": gt_lab, "sem": sem, "mask": mask,
            "mask_clean": mask_clean}


def _color_labels(lab: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (lab * 53) % 256
    rgb[..., 1] = (lab * 97) % 256
    rgb[..., 2] = (lab * 193) % 256
    rgb[lab == 0] = 0
    return rgb


def save_sample(out_dir: Path, sample: dict, params: dict, seed: int) -> None:
    from skimage import io as skio
    import tifffile

    out_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(str(out_dir / "mask.png"),
                (sample["mask"].astype(np.uint8) * 255), check_contrast=False)
    skio.imsave(str(out_dir / "mask_clean.png"),
                (sample["mask_clean"].astype(np.uint8) * 255), check_contrast=False)
    skio.imsave(str(out_dir / "sem.png"),
                (np.clip(sample["sem"], 0, 1) * 255).astype(np.uint8), check_contrast=False)
    tifffile.imwrite(str(out_dir / "gt_labels.tif"),
                     sample["gt_labels"].astype(np.uint16))
    # Overlap-aware GT (each filament owns its full stroke) for robust eval.
    gt_masks = render_gt_instance_masks(sample["gt_labels"].shape, sample["filaments"])
    save_gt_multilabel(out_dir / "gt_multilabel.npz", gt_masks, sample["gt_labels"].shape)
    meta = {
        "seed": int(seed),
        "n_filaments": int(len(sample["filaments"])),
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "filaments": [
            {"id": f.fid, "width_px": f.width, "brightness": f.brightness,
             "centerline_xy": f.pts.round(2).tolist()}
            for f in sample["filaments"]
        ],
    }
    (out_dir / "gt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if params.get("preview"):
        gt_c = _color_labels(sample["gt_labels"])
        mask_c = np.repeat((sample["mask"][..., None].astype(np.uint8) * 255), 3, axis=2)
        clean_c = np.repeat((sample["mask_clean"][..., None].astype(np.uint8) * 255), 3, axis=2)
        sem_c = np.repeat((np.clip(sample["sem"], 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)
        gap = np.full((gt_c.shape[0], 4, 3), 64, dtype=np.uint8)
        combo = np.concatenate([gt_c, gap, mask_c, gap, clean_c, gap, sem_c], axis=1)
        skio.imsave(str(out_dir / "preview.png"), combo, check_contrast=False)


def default_params() -> dict:
    """Defaults tuned for SEM08 scale (REF_UM_PER_PX), gentle degradation.

    Linear px sizes (width_range, psf, break/spurious geometry) are multiplied
    by px_scale = REF_UM_PER_PX / um_per_px in main(), so the *physical* scene is
    held constant when um/px changes. Dimensionless knobs (depths, thresholds,
    probabilities) are scale-invariant.
    """
    return dict(
        um_per_px=REF_UM_PER_PX,
        n_filaments=25,          # moderate density; raise as a stress axis (crossings get hard)
        width_range=(2.5, 5.5),  # px at ref scale
        bright_range=(0.55, 0.98),
        curviness=0.035,         # rad/step; real CNT bundles are fairly straight
        curve_smooth_px=20.0,    # curvature correlation length (px); kills per-step jitter
        len_frac=(0.25, 0.85),   # filament length as fraction of image diagonal
        break_depth=0.30,        # longitudinal dip depth -> faint-spot breaks (gentler)
        mask_thr=0.27,           # ridge threshold for the UNet-like mask
        edge_jagged=0.06,        # boundary noise amplitude (gentler)
        break_prob=0.12,         # chance of an explicit hard cut per filament (gentler)
        false_pos=3.0,           # expected spurious blobs per image (fewer)
        occlusion_prob=0.20,     # chance of dropping a filament at a crossing (gentler)
        bg_level=0.06, bg_amp=0.10, grain=0.04, psf=1.0,
        preview=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="FilaSeg synthetic filament generator")
    ap.add_argument("--out", required=True, help="output root folder")
    ap.add_argument("--n", type=int, default=10, help="number of samples")
    ap.add_argument("--size", type=int, default=512, help="image side (px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--um-per-px", type=float, default=None, dest="um_per_px",
                    help=f"physical scale (default REF {REF_UM_PER_PX:.6g}). "
                         "Linear sizes scale by REF/um_per_px so the physical "
                         "scene is constant across resolutions.")
    ap.add_argument("--n-filaments", type=int, default=None, dest="n_filaments")
    ap.add_argument("--break-depth", type=float, default=None, dest="break_depth")
    ap.add_argument("--break-prob", type=float, default=None, dest="break_prob")
    ap.add_argument("--edge-jagged", type=float, default=None, dest="edge_jagged")
    ap.add_argument("--false-pos", type=float, default=None, dest="false_pos")
    ap.add_argument("--occlusion-prob", type=float, default=None, dest="occlusion_prob")
    ap.add_argument("--curviness", type=float, default=None)
    ap.add_argument("--curve-smooth", type=float, default=None, dest="curve_smooth_px",
                    help="curvature correlation length in px (default 20); "
                         "lower -> wigglier, 0 -> legacy per-step jitter")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    params = default_params()
    for k in ("um_per_px", "n_filaments", "break_depth", "break_prob",
              "edge_jagged", "false_pos", "occlusion_prob", "curviness",
              "curve_smooth_px"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v
    params["preview"] = args.preview

    # Hold the physical scene constant as resolution changes: scale every linear
    # px size by px_scale = REF / um_per_px (1.0 at the reference scale).
    px_scale = REF_UM_PER_PX / float(params["um_per_px"])
    params["px_scale"] = px_scale
    params["width_range"] = tuple(w * px_scale for w in params["width_range"])
    params["psf"] = params["psf"] * px_scale
    params["curve_smooth_px"] = params["curve_smooth_px"] * px_scale
    if abs(px_scale - 1.0) > 1e-6:
        print(f"[scale] um_per_px={params['um_per_px']:.6g}  px_scale={px_scale:.3f} "
              f"(width->{params['width_range'][0]:.1f}-{params['width_range'][1]:.1f}px)")

    out_root = Path(args.out)
    shape = (args.size, args.size)
    master = np.random.default_rng(args.seed)
    for i in range(args.n):
        seed_i = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_i)
        sample = generate_sample(shape, rng, params)
        sample_dir = out_root / f"synth_{i:04d}"
        save_sample(sample_dir, sample, params, seed_i)
        fg = float(sample["mask"].mean())
        print(f"  {sample_dir.name}: {len(sample['filaments'])} filaments, "
              f"fg={fg:.3f}, seed={seed_i}")
    print(f"wrote {args.n} samples to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
