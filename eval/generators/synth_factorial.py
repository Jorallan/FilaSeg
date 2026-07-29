"""Development-only centreline-density by physical-width generator."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label as ndi_label
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
for module_path in (EVAL_ROOT, HERE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
import _evalpath  # noqa: F401,E402
import synth_generator as G  # noqa: E402
import synth_thick as T  # noqa: E402

DEFAULT_LENGTH_DENSITIES = (0.020, 0.040, 0.060)
DEFAULT_WIDTH_CENTRES_PX = (6.0, 11.0, 16.0)
DEFAULT_SEEDS = (20262001, 20262002, 20262003, 20262004)
DEFAULT_WIDTH_HALF_RANGE_PX = 0.75


def reject_locked(path: Path | str) -> None:
    parts = [part.lower() for part in Path(path).resolve().parts]
    if "synthetic_locked_v1" in parts or any(
        part.startswith("locked_evaluation") for part in parts
    ):
        raise ValueError("locked data paths are forbidden for this development study")


def _arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(pts, dtype=float), axis=0), axis=1).sum())


def _content_hash(shape: Tuple[int, int], masks: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(np.asarray(shape, dtype=np.int64).tobytes())
    for index, mask in enumerate(masks, start=1):
        digest.update(np.asarray(index, dtype=np.int32).tobytes())
        digest.update(np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_geometry_sequence(
    shape: Tuple[int, int],
    seed: int,
    max_length_density: float,
    params: Dict[str, Any],
) -> Tuple[List[G.Filament], List[float]]:
    """Generate one ordered path sequence reaching the largest density."""
    if max_length_density <= 0:
        raise ValueError("max_length_density must be positive")
    rng = np.random.default_rng(int(seed))
    diagonal = float(np.hypot(*shape))
    target_length = float(max_length_density) * int(np.prod(shape))
    filaments: List[G.Filament] = []
    width_offsets: List[float] = []
    total = 0.0
    attempts = 0
    while total < target_length and len(filaments) < 400 and attempts < 1200:
        attempts += 1
        pts = T._smooth_path_uniform(
            shape,
            rng,
            params["curviness"],
            min_len=params["len_frac"][0] * diagonal,
            max_len=params["len_frac"][1] * diagonal,
            curve_smooth_px=params["curve_smooth_px"],
        )
        if len(pts) < 4:
            continue
        brightness = float(rng.uniform(*params["bright_range"]))
        modulation = G._longitudinal_modulation(len(pts), rng, params["break_depth"])
        filaments.append(
            G.Filament(len(filaments) + 1, pts, 1.0, brightness, modulation)
        )
        width_offsets.append(float(rng.uniform(-1.0, 1.0)))
        total += _arc_length(pts)
    if total < target_length:
        raise RuntimeError(f"failed to reach centreline density {max_length_density}")
    return filaments, width_offsets


def prefix_for_density(
    filaments: Sequence[G.Filament],
    width_offsets: Sequence[float],
    shape: Tuple[int, int],
    target_length_density: float,
) -> Tuple[List[G.Filament], List[float], float]:
    target = float(target_length_density) * int(np.prod(shape))
    total = 0.0
    count = 0
    for count, filament in enumerate(filaments, start=1):
        total += _arc_length(filament.pts)
        if total >= target:
            break
    if total < target:
        raise ValueError("geometry sequence does not reach target length density")
    return list(filaments[:count]), list(width_offsets[:count]), total


def physical_filaments(
    centreline_filaments: Sequence[G.Filament],
    width_offsets: Sequence[float],
    width_centre_px: float,
    width_half_range_px: float = DEFAULT_WIDTH_HALF_RANGE_PX,
) -> List[G.Filament]:
    widths = [
        max(1.0, float(width_centre_px) + float(width_half_range_px) * float(offset))
        for offset in width_offsets
    ]
    return [
        replace(filament, width=width)
        for filament, width in zip(centreline_filaments, widths)
    ]


def centreline_instance_masks(
    shape: Tuple[int, int], filaments: Sequence[G.Filament]
) -> List[Tuple[int, np.ndarray]]:
    return [
        (int(filament.fid), G.render_clean_skeleton_mask(shape, [filament]))
        for filament in filaments
    ]


def crossing_statistics(
    instance_masks: Sequence[Tuple[int, np.ndarray]], shape: Tuple[int, int]
) -> Dict[str, float | int]:
    """Count pairwise and unique raster centreline crossing regions."""
    owners: Dict[int, List[int]] = {}
    coverage = np.zeros(shape, dtype=np.uint16)
    for fid, mask in instance_masks:
        binary = np.asarray(mask, dtype=bool)
        coverage += binary.astype(np.uint16)
        for flat_index in np.flatnonzero(binary):
            owners.setdefault(int(flat_index), []).append(int(fid))

    pair_pixels: Dict[Tuple[int, int], List[int]] = {}
    for flat_index, ids in owners.items():
        unique_ids = sorted(set(ids))
        for i, left in enumerate(unique_ids):
            for right in unique_ids[i + 1 :]:
                pair_pixels.setdefault((left, right), []).append(flat_index)

    pairwise_events = 0
    image_width = int(shape[1])
    for pixels in pair_pixels.values():
        rows = np.asarray(pixels, dtype=np.int64) // image_width
        cols = np.asarray(pixels, dtype=np.int64) % image_width
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        crop[rows - r0, cols - c0] = True
        pairwise_events += int(
            ndi_label(crop, structure=np.ones((3, 3), dtype=np.uint8))[1]
        )
    unique_events = int(
        ndi_label(coverage >= 2, structure=np.ones((3, 3), dtype=np.uint8))[1]
    )
    return {
        "pairwise_crossing_events": pairwise_events,
        "unique_crossing_regions": unique_events,
        "crossing_density_per_megapixel": float(
            pairwise_events / np.prod(shape) * 1_000_000
        ),
    }


def gap_statistics(
    degraded_mask: np.ndarray, filaments: Sequence[G.Filament]
) -> Dict[str, Any]:
    """Summarize missing runs along ordered true centrelines."""
    support = maximum_filter(
        np.asarray(degraded_mask, dtype=np.uint8), size=3, mode="constant"
    ) > 0
    gaps: List[float] = []
    missing_samples = 0
    total_samples = 0
    no_gap_count = 0
    height, width = support.shape
    for filament in filaments:
        dense, _ = G._densify_path(filament.pts, None, step=1.0)
        rows = np.clip(np.rint(dense[:, 1]).astype(int), 0, height - 1)
        cols = np.clip(np.rint(dense[:, 0]).astype(int), 0, width - 1)
        missing = ~support[rows, cols]
        total_samples += int(missing.size)
        missing_samples += int(missing.sum())
        padded = np.concatenate(([False], missing, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        lengths = (stops - starts).astype(float)
        if lengths.size:
            gaps.extend(lengths.tolist())
        else:
            no_gap_count += 1
    values = np.asarray(gaps, dtype=float)

    def q(value: float) -> float:
        return float(np.quantile(values, value)) if values.size else 0.0

    return {
        "gap_count": int(values.size),
        "missing_centreline_sample_fraction": (
            float(missing_samples / total_samples) if total_samples else 0.0
        ),
        "mean_gap_length_px": float(values.mean()) if values.size else 0.0,
        "median_gap_length_px": q(0.5),
        "p90_gap_length_px": q(0.9),
        "max_gap_length_px": float(values.max()) if values.size else 0.0,
        "filaments_with_no_detected_gap": int(no_gap_count),
        "total_centreline_samples": int(total_samples),
        "missing_centreline_samples": int(missing_samples),
    }


def _save_binary(path: Path, mask: np.ndarray) -> None:
    from skimage import io as skio

    skio.imsave(str(path), np.asarray(mask, dtype=np.uint8) * 255, check_contrast=False)


def _save_common(
    out_dir: Path,
    clean_mask: np.ndarray,
    degraded_mask: np.ndarray,
    centreline_masks: Sequence[Tuple[int, np.ndarray]],
    shape: Tuple[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_binary(out_dir / "mask.png", degraded_mask)
    _save_binary(out_dir / "mask_w1.png", degraded_mask)
    _save_binary(out_dir / "mask_clean.png", clean_mask)
    G.save_gt_multilabel(
        out_dir / "gt_centerlines_multilabel.npz", centreline_masks, shape
    )


def generate_factorial(
    out_root: Path | str,
    length_densities: Sequence[float] = DEFAULT_LENGTH_DENSITIES,
    width_centres_px: Sequence[float] = DEFAULT_WIDTH_CENTRES_PX,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    shape: Tuple[int, int] = (512, 512),
    width_half_range_px: float = DEFAULT_WIDTH_HALF_RANGE_PX,
    params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Generate the paired grid and return per-cell metadata."""
    out_root = Path(out_root).resolve()
    reject_locked(out_root)
    if not length_densities or not width_centres_px or not seeds:
        raise ValueError("factor levels and seeds must be non-empty")
    densities = tuple(sorted({float(value) for value in length_densities}))
    widths = tuple(sorted({float(value) for value in width_centres_px}))
    if densities[0] <= 0 or widths[0] <= 0:
        raise ValueError("factor levels must be positive")
    active_params = dict(T.thick_params() if params is None else params)
    active_params["preview"] = False
    out_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for seed in [int(value) for value in seeds]:
        sequence, offsets = generate_geometry_sequence(
            shape, seed, max(densities), active_params
        )
        for density_index, density in enumerate(densities):
            centre_fils, centre_offsets, total_length = prefix_for_density(
                sequence, offsets, shape, density
            )
            centre_masks = centreline_instance_masks(shape, centre_fils)
            clean_mask = np.any(
                np.stack([mask for _, mask in centre_masks], axis=0), axis=0
            )
            thin_fils = [replace(filament, width=1.0) for filament in centre_fils]
            thin_ridge = T.thin_axis_ridge(shape, thin_fils, 1)
            degradation_seed = int(seed + 100_000 + density_index * 10_000)
            degraded_mask = G.degrade_to_mask(
                thin_ridge,
                thin_fils,
                G.render_gt_labels(shape, thin_fils),
                np.random.default_rng(degradation_seed),
                mask_thr=active_params["mask_thr"],
                edge_jagged=active_params["edge_jagged"],
                break_prob=active_params["break_prob"],
                false_pos=active_params["false_pos"],
                occlusion_prob=active_params["occlusion_prob"],
            )
            crossings = crossing_statistics(centre_masks, shape)
            gaps = gap_statistics(degraded_mask, centre_fils)
            common_hash = _content_hash(shape, [mask for _, mask in centre_masks])
            achieved_density = float(total_length / np.prod(shape))
            final_length = _arc_length(centre_fils[-1].pts)
            first_common_dir: Path | None = None

            for width_centre in widths:
                scene_dir = (
                    out_root
                    / f"ld{int(round(density * 1000)):03d}"
                    / f"w{int(round(width_centre)):02d}"
                    / f"seed_{seed}"
                )
                if first_common_dir is None:
                    _save_common(
                        scene_dir, clean_mask, degraded_mask, centre_masks, shape
                    )
                    first_common_dir = scene_dir
                else:
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "mask.png",
                        "mask_w1.png",
                        "mask_clean.png",
                        "gt_centerlines_multilabel.npz",
                    ):
                        shutil.copyfile(first_common_dir / name, scene_dir / name)

                thick_fils = physical_filaments(
                    centre_fils, centre_offsets, width_centre, width_half_range_px
                )
                thick_labels = G.render_gt_labels(shape, thick_fils)
                thick_masks = G.render_gt_instance_masks(shape, thick_fils)
                ridge = G.render_ridge(shape, thick_fils)
                sem_seed = int(seed + 200_000 + density_index * 10_000)
                sem = G.make_sem(
                    ridge,
                    np.random.default_rng(sem_seed),
                    active_params["bg_level"],
                    active_params["bg_amp"],
                    active_params["grain"],
                    active_params["psf"],
                )
                import tifffile
                from skimage import io as skio

                tifffile.imwrite(
                    str(scene_dir / "gt_labels.tif"), thick_labels.astype(np.uint16)
                )
                G.save_gt_multilabel(
                    scene_dir / "gt_thick_multilabel.npz", thick_masks, shape
                )
                skio.imsave(
                    str(scene_dir / "sem.png"),
                    (np.clip(sem, 0, 1) * 255).astype(np.uint8),
                    check_contrast=False,
                )
                achieved_widths = np.asarray(
                    [filament.width for filament in thick_fils], dtype=float
                )
                meta: Dict[str, Any] = {
                    "dataset_role": "development",
                    "study": "factorial_centreline_density_x_bundle_width",
                    "seed": seed,
                    "shape": list(shape),
                    "target_length_density_px_per_px2": density,
                    "achieved_length_density_px_per_px2": achieved_density,
                    "achieved_length_density_um_per_um2": achieved_density
                    / G.REF_UM_PER_PX,
                    "total_visible_centreline_length_px": float(total_length),
                    "last_filament_length_px": float(final_length),
                    "clean_1px_union_coverage": float(clean_mask.mean()),
                    "degraded_input_coverage": float(degraded_mask.mean()),
                    "bundle_width_level_px": width_centre,
                    "bundle_width_half_range_px": float(width_half_range_px),
                    "achieved_bundle_width_px": {
                        "mean": float(achieved_widths.mean()),
                        "sd": float(achieved_widths.std(ddof=1))
                        if achieved_widths.size > 1
                        else 0.0,
                        "min": float(achieved_widths.min()),
                        "max": float(achieved_widths.max()),
                    },
                    "achieved_thick_mask_coverage": float((thick_labels > 0).mean()),
                    "n_filaments": int(len(centre_fils)),
                    **crossings,
                    **gaps,
                    "geometry_seed": seed,
                    "degradation_seed": degradation_seed,
                    "sem_noise_seed": sem_seed,
                    "centreline_gt_content_sha256": common_hash,
                    "params": {
                        key: list(value) if isinstance(value, tuple) else value
                        for key, value in active_params.items()
                    },
                }
                (scene_dir / "gt_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                meta["scene_dir"] = str(scene_dir)
                meta["artifact_sha256"] = {
                    name: _file_hash(scene_dir / name)
                    for name in (
                        "mask.png",
                        "mask_clean.png",
                        "gt_centerlines_multilabel.npz",
                        "sem.png",
                        "gt_thick_multilabel.npz",
                    )
                }
                records.append(meta)

            paired = records[-len(widths) :]
            for name in ("mask.png", "mask_clean.png", "gt_centerlines_multilabel.npz"):
                if len({record["artifact_sha256"][name] for record in paired}) != 1:
                    raise RuntimeError(
                        f"width negative-control invariant failed for {name}, "
                        f"seed={seed}, density={density}"
                    )

    manifest = {
        "schema_version": 1,
        "study": "development_only_factorial_centreline_density_x_bundle_width",
        "length_density_levels_px_per_px2": list(densities),
        "bundle_width_levels_px": list(widths),
        "seeds": [int(value) for value in seeds],
        "shape": list(shape),
        "n_cells": len(records),
        "records": records,
    }
    (out_root / "factorial_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return records


def _csv_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="output/development_factorial_density_width/scenes"
    )
    parser.add_argument("--length-densities", default="0.020,0.040,0.060")
    parser.add_argument("--widths", default="6,11,16")
    parser.add_argument("--seeds", default="20262001,20262002,20262003,20262004")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    records = generate_factorial(
        args.out,
        length_densities=_csv_floats(args.length_densities),
        width_centres_px=_csv_floats(args.widths),
        seeds=_csv_ints(args.seeds),
        shape=(args.size, args.size),
    )
    print(f"wrote {len(records)} development cells to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
