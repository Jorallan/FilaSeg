"""Non-overwriting representation audit for a completed locked evaluation.

This module derives a new report from stored inputs and method artifacts. It
does not run FilaSeg, tune a parameter, or modify the source locked report.

Primary endpoint
----------------
All predictions are converted to overlap-aware per-instance centrelines, and
the common-fragment partition is scored with GT-unassigned fragments retained
as singleton evidence. FilaSeg is taken from Stage 3, before width rendering.

Secondary endpoints retain the Stage-4 rendered result and a skeletonized
Stage-4 result so the effect of postprocessing and representation can be
reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from skimage import io as skio
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402
import stats_util as SU  # noqa: E402


SCHEMA_VERSION = 1
PRIMARY_FILASEG = "filaseg_stage3_centerline"
PRIMARY_BASELINE = "minimum_turn_centerline"
METHODS = (
    "connected_components_centerline",
    PRIMARY_BASELINE,
    PRIMARY_FILASEG,
    "filaseg_stage3_native",
    "filaseg_stage4_centerline",
    "filaseg_stage4_rendered",
)
SUMMARY_METRICS = (
    "f1",
    "precision",
    "recall",
    "adjusted_rand_index",
    "vi_merge_bits",
    "vi_split_bits",
    "fragment_recovery_recovery_rate",
    "fragment_recovery_false_instances",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def centerline_instances(instances: IIO.InstanceSet) -> IIO.InstanceSet:
    """Skeletonize each instance independently while preserving overlap layers."""
    masks: Dict[int, np.ndarray] = {}
    for instance_id, mask in sorted(instances.masks.items()):
        centreline = skeletonize(np.asarray(mask, dtype=bool))
        if centreline.any():
            masks[int(instance_id)] = centreline
    return IIO.InstanceSet(
        shape=instances.shape,
        masks=masks,
        source=f"per-instance-centreline({instances.source})",
    )


def resolve_stage3_artifact(final_artifact: Path) -> Path:
    """Resolve the stored Stage-3 multilabel paired with a final artifact."""
    final_artifact = Path(final_artifact)
    if final_artifact.parent.name != "final":
        raise ValueError(
            f"FilaSeg locked output is not under a final/ directory: {final_artifact}"
        )
    run_root = final_artifact.parent.parent
    candidate = run_root / "3.reconnect" / final_artifact.name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"paired Stage-3 artifact is missing for {final_artifact}: {candidate}"
        )
    return candidate


def _read_binary_mask(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return np.asarray(image) > 0


def _coverage(mask: np.ndarray, fragments: np.ndarray) -> Dict[str, Any]:
    fragment_skeleton = CM._prune_spurs(skeletonize(mask), 3)
    foreground_pixels = int(mask.sum())
    fragment_pixels = int((fragments != 0).sum())
    skeleton_pixels = int(fragment_skeleton.sum())
    return {
        "n_common_fragments": int(fragments.max()),
        "n_foreground_pixels": foreground_pixels,
        "n_skeleton_pixels": skeleton_pixels,
        "n_common_fragment_pixels": fragment_pixels,
        "common_fragment_foreground_pixel_coverage": (
            fragment_pixels / foreground_pixels if foreground_pixels else float("nan")
        ),
        "common_fragment_skeleton_coverage": (
            fragment_pixels / skeleton_pixels if skeleton_pixels else float("nan")
        ),
    }


def _score_assignment(
    coverage: Mapping[str, Any],
    gt_assignment: Mapping[int, int],
    pred_assignment: Mapping[int, int],
    *,
    gt_unassigned_policy: str = "singleton",
) -> Dict[str, Any]:
    out = dict(coverage)
    out.update(CM.pairwise_scores(
        dict(gt_assignment),
        dict(pred_assignment),
        gt_unassigned_policy=gt_unassigned_policy,
    ))
    recovery = CM.fragment_instance_recovery(
        dict(gt_assignment), dict(pred_assignment), 0.8, 0.8
    )
    out.update({
        f"fragment_recovery_{key}": value
        for key, value in recovery.items()
    })
    return out


def _artifact_record(path: Path, stage: str, representation: str) -> Dict[str, str]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "stage": stage,
        "representation": representation,
    }


def _method_rows(source: Mapping[str, Any]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    by_scene: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in source.get("per_scene", []):
        if row.get("status") != "ok":
            raise ValueError(
                "representation audit requires a complete successful checkpoint; "
                f"found {row.get('geometry_id')} {row.get('method')}={row.get('status')}"
            )
        scene_id = str(row["geometry_id"])
        method = str(row["method"])
        if method in by_scene.setdefault(scene_id, {}):
            raise ValueError(f"duplicate source row for {scene_id} and {method}")
        by_scene[scene_id][method] = row
    expected = {"baseline_cc", "baseline_skeleton", "filaseg"}
    for scene_id, rows in by_scene.items():
        if set(rows) != expected:
            raise ValueError(
                f"{scene_id} methods must be {sorted(expected)}, got {sorted(rows)}"
            )
    return by_scene


def score_scene(
    scene_id: str,
    rows: Mapping[str, Mapping[str, Any]],
    samples_root: Path,
    mask_name: str,
) -> Dict[str, Any]:
    """Score one stored scene under every declared representation."""
    started = time.perf_counter()
    scene_dir = Path(samples_root) / Path(scene_id)
    mask_path = scene_dir / mask_name
    gt_path = scene_dir / "gt_labels.tif"
    if not mask_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"missing locked input or GT under {scene_dir}")

    cc_path = Path(str(rows["baseline_cc"]["output_path"]))
    mt_path = Path(str(rows["baseline_skeleton"]["output_path"]))
    stage4_path = Path(str(rows["filaseg"]["output_path"]))
    stage3_path = resolve_stage3_artifact(stage4_path)
    gt_overlap_path = gt_path.with_name("gt_multilabel.npz")
    for artifact in (cc_path, mt_path, stage3_path, stage4_path):
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    cc_path = cc_path if cc_path.is_absolute() else ROOT / cc_path
    mt_path = mt_path if mt_path.is_absolute() else ROOT / mt_path
    stage3_path = stage3_path if stage3_path.is_absolute() else ROOT / stage3_path
    stage4_path = stage4_path if stage4_path.is_absolute() else ROOT / stage4_path

    mask = _read_binary_mask(mask_path)
    fragments = CM.common_fragments(mask)
    coverage = _coverage(mask, fragments)
    gt_instances, _ = IIO.load_gt(gt_path, shape=mask.shape)
    gt_assignment = CM.assign_fragments(fragments, gt_instances)

    cc_native = IIO.load_pred_instances(cc_path)
    mt_native = IIO.load_pred_instances(mt_path)
    stage3_native = IIO.load_pred_instances(stage3_path)
    stage4_native = IIO.load_pred_instances(stage4_path)
    predictions = {
        "connected_components_centerline": centerline_instances(cc_native),
        "minimum_turn_centerline": centerline_instances(mt_native),
        "filaseg_stage3_centerline": centerline_instances(stage3_native),
        "filaseg_stage3_native": stage3_native,
        "filaseg_stage4_centerline": centerline_instances(stage4_native),
        "filaseg_stage4_rendered": stage4_native,
    }
    scores: Dict[str, Dict[str, Any]] = {}
    assignments: Dict[str, Dict[int, int]] = {}
    for method, prediction in predictions.items():
        if prediction.shape != mask.shape:
            raise ValueError(
                f"{scene_id} {method} shape {prediction.shape} != {mask.shape}"
            )
        assignment = CM.assign_fragments(fragments, prediction)
        assignments[method] = assignment
        scores[method] = _score_assignment(
            coverage, gt_assignment, assignment, gt_unassigned_policy="singleton"
        )

    stored = rows["filaseg"]["scores"]["common_metric"]
    legacy = _score_assignment(
        coverage,
        gt_assignment,
        assignments["filaseg_stage4_rendered"],
        gt_unassigned_policy="exclude",
    )
    for metric in ("precision", "recall", "f1"):
        if not np.isclose(
            float(legacy[metric]), float(stored[metric]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"legacy Stage-4 validation drift for {scene_id} {metric}: "
                f"{legacy[metric]} != {stored[metric]}"
            )

    density = int(rows["filaseg"]["density"])
    return _jsonable({
        "geometry_id": scene_id,
        "density": density,
        "seed": int(rows["filaseg"]["seed"]),
        "scores": scores,
        "legacy_stage4_validation": {
            "stored_common_f1": stored["f1"],
            "rescored_common_f1": legacy["f1"],
            "absolute_difference": abs(float(stored["f1"]) - float(legacy["f1"])),
            "gt_unassigned_policy": "exclude",
        },
        "artifacts": {
            "input_mask": _artifact_record(mask_path, "input", "binary_axis_mask"),
            "ground_truth": _artifact_record(
                gt_path, "ground_truth", "flat_label_sibling"
            ),
            "ground_truth_selected": _artifact_record(
                gt_overlap_path if gt_overlap_path.is_file() else gt_path,
                "ground_truth",
                ("overlap_aware_multilabel" if gt_overlap_path.is_file()
                 else "flat_labels"),
            ),
            "connected_components": _artifact_record(
                cc_path, "baseline", "flat_thin_labels"
            ),
            "minimum_turn": _artifact_record(
                mt_path, "baseline", "flat_thin_labels"
            ),
            "filaseg_stage3": _artifact_record(
                stage3_path, "3.reconnect", "overlap_aware_multilabel"
            ),
            "filaseg_stage4": _artifact_record(
                stage4_path, "final", "rendered_overlap_aware_multilabel"
            ),
        },
        "audit_runtime_seconds": time.perf_counter() - started,
    })


def _values_by_method(
    per_scene: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, Dict[str, float]]:
    return {
        method: {
            str(row["geometry_id"]): float(row["scores"][method][metric])
            for row in per_scene
            if row["scores"][method].get(metric) is not None
        }
        for method in METHODS
    }


def _stratified_summary(
    values: Mapping[str, float], strata: Mapping[str, str]
) -> Dict[str, Any]:
    grouped: Dict[str, list[float]] = {}
    for scene_id, value in values.items():
        grouped.setdefault(strata[scene_id], []).append(value)
    return SU.stratified_summarize(grouped).as_dict()


def _aggregate(per_scene: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    strata = {
        str(row["geometry_id"]): str(row["density"])
        for row in per_scene
    }
    by_method: Dict[str, Dict[str, Any]] = {method: {} for method in METHODS}
    value_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in SUMMARY_METRICS:
        values = _values_by_method(per_scene, metric)
        value_cache[metric] = values
        for method in METHODS:
            by_method[method][metric] = _stratified_summary(
                values[method], strata
            )

    comparisons = (
        (PRIMARY_FILASEG, PRIMARY_BASELINE, "primary_filaseg_minus_minimum_turn"),
        (PRIMARY_FILASEG, "connected_components_centerline",
         "primary_filaseg_minus_connected_components"),
        ("filaseg_stage4_rendered", PRIMARY_FILASEG,
         "stage4_rendered_minus_stage3_centerline"),
        ("filaseg_stage4_centerline", PRIMARY_FILASEG,
         "stage4_centerline_minus_stage3_centerline"),
        ("filaseg_stage4_rendered", "filaseg_stage4_centerline",
         "stage4_rendered_minus_stage4_centerline"),
        ("filaseg_stage3_native", PRIMARY_FILASEG,
         "stage3_native_minus_stage3_centerline"),
        ("filaseg_stage4_rendered", PRIMARY_BASELINE,
         "stage4_rendered_minus_minimum_turn"),
    )
    paired: Dict[str, Dict[str, Any]] = {}
    for left, right, name in comparisons:
        paired[name] = {
            "method_a": left,
            "method_b": right,
            **{
                metric: SU.paired_stratified_summarize_by_scene(
                    value_cache[metric][left],
                    value_cache[metric][right],
                    strata,
                ).as_dict()
                for metric in SUMMARY_METRICS
                if set(value_cache[metric][left]) == set(value_cache[metric][right])
            },
        }

    by_density: Dict[str, Any] = {}
    for density in sorted(set(strata.values()), key=int):
        scene_ids = sorted(
            scene_id for scene_id, value in strata.items() if value == density
        )
        density_methods: Dict[str, Any] = {}
        for method in METHODS:
            density_methods[method] = {
                metric: SU.summarize([
                    value_cache[metric][method][scene_id]
                    for scene_id in scene_ids
                    if scene_id in value_cache[metric][method]
                ]).as_dict()
                for metric in SUMMARY_METRICS
            }
        density_pairs: Dict[str, Any] = {}
        for left, right, name in comparisons:
            density_pairs[name] = {
                "method_a": left,
                "method_b": right,
                **{
                    metric: SU.paired_summarize(
                        [value_cache[metric][left][scene_id] for scene_id in scene_ids],
                        [value_cache[metric][right][scene_id] for scene_id in scene_ids],
                    ).as_dict()
                    for metric in SUMMARY_METRICS
                    if all(
                        scene_id in value_cache[metric][left]
                        and scene_id in value_cache[metric][right]
                        for scene_id in scene_ids
                    )
                },
            }
        by_density[density] = {
            "n_scenes": len(scene_ids),
            "by_method": density_methods,
            "paired": density_pairs,
        }
    return _jsonable({
        "bootstrap": {
            "unit": "independent_scene",
            "overall_resampling": "within_density_then_scene_weighted_pooling",
            "seed": SU.BOOTSTRAP_SEED,
            "n_resamples": SU.N_BOOTSTRAP,
        },
        "by_method": by_method,
        "paired": paired,
        "by_density": by_density,
    })


def select_qualitative_scenes(
    per_scene: Sequence[Mapping[str, Any]]
) -> Dict[str, list[Dict[str, Any]]]:
    """Select wins, losses and near ties from the corrected primary F1."""
    ranked = []
    for row in per_scene:
        fila = float(row["scores"][PRIMARY_FILASEG]["f1"])
        minturn = float(row["scores"][PRIMARY_BASELINE]["f1"])
        ranked.append({
            "geometry_id": str(row["geometry_id"]),
            "density": int(row["density"]),
            "filaseg_f1": fila,
            "minimum_turn_f1": minturn,
            "filaseg_minus_minimum_turn_f1": fila - minturn,
        })
    ranked.sort(key=lambda row: (
        row["filaseg_minus_minimum_turn_f1"], row["geometry_id"]
    ))
    minimum_turn_wins = ranked[:3]
    filaseg_wins = list(reversed(ranked[-3:]))
    excluded = {
        row["geometry_id"] for row in minimum_turn_wins + filaseg_wins
    }
    near_ties = sorted(
        (row for row in ranked if row["geometry_id"] not in excluded),
        key=lambda row: (
            abs(row["filaseg_minus_minimum_turn_f1"]), row["geometry_id"]
        ),
    )[:2]
    return {
        "selection_metric": (
            "filaseg_stage3_centerline.f1 - minimum_turn_centerline.f1"
        ),
        "three_largest_filaseg_wins": filaseg_wins,
        "three_largest_minimum_turn_wins": minimum_turn_wins,
        "two_near_ties": near_ties,
    }


def build_report(
    source_json: Path,
    out_json: Path,
    samples_root: Path | None = None,
) -> Dict[str, Any]:
    source_json = Path(source_json).resolve()
    out_json = Path(out_json).resolve()
    if source_json == out_json:
        raise ValueError("derived audit report must not overwrite the source report")
    if out_json.exists():
        raise FileExistsError(
            f"derived audit report already exists; refusing to overwrite: {out_json}"
        )
    source_bytes = source_json.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    if not source.get("configuration", {}).get("configuration_locked"):
        raise ValueError("source report is not marked configuration_locked")
    samples = Path(samples_root or source["samples_root"]).resolve()
    rows_by_scene = _method_rows(source)
    mask_name = str(source["configuration"]["mask_name"])
    started = time.perf_counter()
    per_scene = [
        score_scene(scene_id, rows_by_scene[scene_id], samples, mask_name)
        for scene_id in sorted(rows_by_scene)
    ]
    if len(per_scene) != 125:
        raise ValueError(f"expected 125 locked scenes, found {len(per_scene)}")

    code_paths = (
        Path(__file__),
        ROOT / "eval" / "core" / "common_metric.py",
        ROOT / "eval" / "core" / "instance_io.py",
        ROOT / "eval" / "stats" / "stats_util.py",
    )
    report = _jsonable({
        "schema_version": SCHEMA_VERSION,
        "report_kind": "locked_evaluation_representation_and_metric_correction",
        "source_locked_report": str(source_json),
        "source_locked_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_locked_report_preserved": True,
        "configuration": {
            "primary_endpoint": (
                "common-fragment partition scores on equivalent per-instance "
                "centreline representations before width rendering"
            ),
            "primary_filaseg_method": PRIMARY_FILASEG,
            "primary_baseline_method": PRIMARY_BASELINE,
            "gt_unassigned_policy": "singleton",
            "unassigned_cluster_policy": "singleton_per_fragment",
            "centreline_transform": (
                "skimage.morphology.skeletonize independently per instance layer"
            ),
            "samples_root": str(samples),
            "mask_name": mask_name,
            "source_configuration": source["configuration"],
        },
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "code_hashes": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in code_paths
            },
            "source_config_hashes": source.get("config_hashes", {}),
            "audit_runtime_seconds": time.perf_counter() - started,
        },
        "per_scene": per_scene,
        "aggregate": _aggregate(per_scene),
        "qualitative_selection": select_qualitative_scenes(per_scene),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive a non-overwriting equivalent-representation audit"
    )
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-root")
    args = parser.parse_args(argv)
    build_report(
        Path(args.source_json),
        Path(args.out),
        Path(args.samples_root) if args.samples_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
