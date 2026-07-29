"""Validate and summarize the development density-by-width factorial grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ACHIEVED_METRICS = (
    "achieved_length_density_px_per_px2",
    "achieved_length_density_um_per_um2",
    "total_visible_centreline_length_px",
    "clean_1px_union_coverage",
    "degraded_input_coverage",
    "achieved_thick_mask_coverage",
    "n_filaments",
    "pairwise_crossing_events",
    "unique_crossing_regions",
    "crossing_density_per_megapixel",
    "gap_count",
    "missing_centreline_sample_fraction",
    "mean_gap_length_px",
    "median_gap_length_px",
    "p90_gap_length_px",
    "max_gap_length_px",
    "filaments_with_no_detected_gap",
)


def reject_locked(path: Path | str) -> None:
    parts = [part.lower() for part in Path(path).resolve().parts]
    if "synthetic_locked_v1" in parts or any(
        part.startswith("locked_evaluation") for part in parts
    ):
        raise ValueError("locked data paths are forbidden for this development study")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_records(root: Path | str) -> List[Dict[str, Any]]:
    root = Path(root).resolve()
    reject_locked(root)
    records: List[Dict[str, Any]] = []
    for meta_path in sorted(root.rglob("gt_meta.json")):
        reject_locked(meta_path)
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("dataset_role") != "development":
            raise ValueError(f"non-development record: {meta_path}")
        record["scene_dir"] = str(meta_path.parent)
        record["artifact_sha256"] = {
            name: _sha256(meta_path.parent / name)
            for name in (
                "mask.png",
                "mask_clean.png",
                "gt_centerlines_multilabel.npz",
                "sem.png",
                "gt_thick_multilabel.npz",
            )
        }
        records.append(record)
    if not records:
        raise ValueError(f"no factorial records found under {root}")
    return records


def _key(record: Mapping[str, Any]) -> Tuple[int, float]:
    return int(record["seed"]), float(record["target_length_density_px_per_px2"])


def _width(record: Mapping[str, Any]) -> float:
    return float(record["bundle_width_level_px"])


def validate_grid(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Enforce orthogonality and the structural Stage-3 negative control."""
    widths = sorted({_width(record) for record in records})
    densities = sorted(
        {float(record["target_length_density_px_per_px2"]) for record in records}
    )
    seeds = sorted({int(record["seed"]) for record in records})
    expected = {
        (seed, density, width)
        for seed in seeds
        for density in densities
        for width in widths
    }
    observed = {
        (
            int(record["seed"]),
            float(record["target_length_density_px_per_px2"]),
            _width(record),
        )
        for record in records
    }
    if observed != expected or len(records) != len(expected):
        raise ValueError(
            f"incomplete or duplicated grid: expected {len(expected)} cells, "
            f"found {len(records)} records and {len(observed)} unique cells"
        )

    groups: Dict[Tuple[int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_key(record)].append(record)
        target = float(record["target_length_density_px_per_px2"])
        achieved = float(record["achieved_length_density_px_per_px2"])
        maximum = target + float(record["last_filament_length_px"]) / np.prod(
            record["shape"]
        )
        if not target <= achieved <= maximum + 1e-12:
            raise ValueError(
                f"length-density stopping rule failed for {record['scene_dir']}"
            )
        if float(record["achieved_thick_mask_coverage"]) >= 0.75:
            raise ValueError(f"thick-mask saturation in {record['scene_dir']}")
        if int(record["n_filaments"]) < 1:
            raise ValueError(f"empty geometry in {record['scene_dir']}")
        for metric in ACHIEVED_METRICS:
            if metric not in record or not np.isfinite(float(record[metric])):
                raise ValueError(f"missing or non-finite {metric} in {record['scene_dir']}")

    invariant_names = (
        "mask.png",
        "mask_clean.png",
        "gt_centerlines_multilabel.npz",
    )
    invariant_groups = []
    for key, group in sorted(groups.items()):
        checks: Dict[str, str] = {}
        if len({str(record["centreline_gt_content_sha256"]) for record in group}) != 1:
            raise ValueError(f"centreline GT content varies across widths for {key}")
        for name in invariant_names:
            hashes = {record["artifact_sha256"][name] for record in group}
            if len(hashes) != 1:
                raise ValueError(f"{name} varies across widths for {key}")
            checks[name] = next(iter(hashes))
        for metric in (
            "achieved_length_density_px_per_px2",
            "clean_1px_union_coverage",
            "degraded_input_coverage",
            "n_filaments",
            "pairwise_crossing_events",
            "gap_count",
        ):
            if len({float(record[metric]) for record in group}) != 1:
                raise ValueError(f"{metric} varies across widths for {key}")
        invariant_groups.append(
            {
                "seed": key[0],
                "target_length_density_px_per_px2": key[1],
                "n_width_cells": len(group),
                "identical_artifact_sha256": checks,
            }
        )

    for seed in seeds:
        ordered = sorted(
            (
                float(record["target_length_density_px_per_px2"]),
                float(record["achieved_length_density_px_per_px2"]),
                float(record["clean_1px_union_coverage"]),
            )
            for record in records
            if int(record["seed"]) == seed and _width(record) == widths[0]
        )
        if not all(left[1] < right[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"achieved length density is not increasing for seed {seed}")
        if not all(left[2] < right[2] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"clean one-pixel coverage is not increasing for seed {seed}")

    return {
        "n_cells": len(records),
        "length_density_levels_px_per_px2": densities,
        "bundle_width_levels_px": widths,
        "seeds": seeds,
        "stage3_width_negative_control": {
            "status": "pass",
            "basis": (
                "mask, clean centreline, and overlap-aware centreline GT are "
                "byte-identical across physical-width cells"
            ),
            "n_geometry_blocks": len(invariant_groups),
            "blocks": invariant_groups,
        },
    }


def validate_stage3_results(
    records: Sequence[Mapping[str, Any]],
    stage3_rows: Sequence[Mapping[str, Any]],
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Validate optional empirical Stage-3 common-F1 negative-control rows."""
    expected = {
        (
            int(record["seed"]),
            float(record["target_length_density_px_per_px2"]),
            _width(record),
        )
        for record in records
    }
    values = {
        (
            int(row["seed"]),
            float(row["target_length_density_px_per_px2"]),
            float(row["bundle_width_level_px"]),
        ): float(row["common_f1"])
        for row in stage3_rows
    }
    if set(values) != expected:
        raise ValueError("Stage-3 rows do not match the factorial grid")
    blocks: Dict[Tuple[int, float], List[float]] = defaultdict(list)
    for (seed, density, _), value in values.items():
        blocks[(seed, density)].append(value)
    maximum_range = max(max(group) - min(group) for group in blocks.values())
    if maximum_range > float(tolerance):
        raise ValueError(
            f"Stage-3 width negative control failed: maximum score range "
            f"{maximum_range} > {tolerance}"
        )
    return {
        "status": "pass",
        "n_rows": len(stage3_rows),
        "tolerance": float(tolerance),
        "maximum_within_geometry_score_range": float(maximum_range),
    }


def _summary(values: Iterable[float]) -> Dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_cells(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                float(record["target_length_density_px_per_px2"]),
                _width(record),
            )
        ].append(record)
    return [
        {
            "target_length_density_px_per_px2": density,
            "bundle_width_level_px": width,
            "n": len(group),
            "metrics": {
                metric: _summary(float(record[metric]) for record in group)
                for metric in ACHIEVED_METRICS
            },
        }
        for (density, width), group in sorted(groups.items())
    ]


def build_report(
    scenes_root: Path | str,
    output_path: Path | str,
    stage3_results: Path | str | None = None,
) -> Dict[str, Any]:
    scenes_root = Path(scenes_root).resolve()
    output_path = Path(output_path).resolve()
    reject_locked(scenes_root)
    reject_locked(output_path)
    records = discover_records(scenes_root)
    validation = validate_grid(records)
    empirical = {"status": "not_run"}
    if stage3_results:
        stage3_path = Path(stage3_results).resolve()
        reject_locked(stage3_path)
        document = json.loads(stage3_path.read_text(encoding="utf-8"))
        empirical = validate_stage3_results(
            records, document.get("per_scene", document.get("rows", []))
        )
    report = {
        "schema_version": 1,
        "study": "development_only_factorial_centreline_density_x_bundle_width",
        "scope": (
            "Exploratory development evidence only. The existing locked set is "
            "forbidden and was not read."
        ),
        "scenes_root": str(scenes_root),
        "validation": validation,
        "empirical_stage3_width_negative_control": empirical,
        "factorial_cells": summarize_cells(records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", default="output/development_factorial_density_width/scenes"
    )
    parser.add_argument(
        "--out",
        default="output/development_factorial_density_width/factorial_report.json",
    )
    parser.add_argument("--stage3-results", default="")
    args = parser.parse_args()
    report = build_report(
        args.scenes, args.out, stage3_results=args.stage3_results or None
    )
    print(
        f"validated {report['validation']['n_cells']} development cells; "
        f"report: {Path(args.out).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
