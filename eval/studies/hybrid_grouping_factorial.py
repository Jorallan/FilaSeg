"""Development-only 2x2 fragment-source by grouping-kernel study.

This study varies two factors while scoring every cell on one fixed,
input-derived common-fragment population:

* fragment source: input-skeleton atoms or stored Stage-2 Hough atoms;
* grouper: minimum-turn matching or the staged FilaSeg reconnect gates.

It is intentionally not a production-pipeline score. All source fragments are
skeletonized before grouping, Stage-4 rendering is absent, and FilaSeg overlap
suppression is disabled so the supplied fragment population is not trimmed or
deleted. No threshold is tuned here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml
from skimage import io as skio
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import baselines  # noqa: E402
import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402
import stats_util as ST  # noqa: E402

sys.path.insert(0, str(ROOT / "3.reconnect"))
import reconnect_run as RR  # noqa: E402
import reconnect_utils_straight as RU  # noqa: E402


CELLS = (
    "skeleton_minimum_turn",
    "skeleton_filaseg_gates",
    "hough_minimum_turn",
    "hough_filaseg_gates",
)
METRICS = (
    "precision",
    "recall",
    "f1",
    "adjusted_rand_index",
    "vi_merge_bits",
    "vi_split_bits",
    "fragment_recovery_recovery_rate",
)
FORBIDDEN_PATH_PARTS = frozenset({"synthetic_locked_v1", "locked_evaluation"})
ORIENTATION_BINS = 12
STAGE_KEYS = ("stage_clear", "stage_strict", "stage_relaxed")


def _reject_locked_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    lowered = {part.lower() for part in resolved.parts}
    forbidden = sorted(lowered & FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise ValueError(
            f"locked evaluation paths are forbidden in this study: {resolved}"
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _read_mask(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return np.asarray(image) > 0


def discover_development_scenes(
    samples_root: Path | str,
    hough_root: Path | str,
    mask_name: str = "mask_w2.png",
    expected_scenes: int = 20,
) -> list[dict]:
    samples = _reject_locked_path(samples_root)
    hough = _reject_locked_path(hough_root)
    scenes = []
    for density_dir in sorted(samples.glob("cov*")):
        if not density_dir.is_dir():
            continue
        suffix = density_dir.name[3:]
        if not suffix.isdigit():
            continue
        density = int(suffix)
        for sample_dir in sorted(density_dir.glob("synth_*")):
            if not sample_dir.is_dir():
                continue
            mask_path = sample_dir / mask_name
            gt_path = sample_dir / "gt_multilabel.npz"
            meta_path = sample_dir / "gt_meta.json"
            run_root = (
                hough / "filaseg" / density_dir.name / sample_dir.name
                / "2" / "filaseg"
            )
            branches = run_root / "2.preprocess" / "branches"
            for required in (mask_path, gt_path, meta_path, branches):
                _reject_locked_path(required)
                if not required.exists():
                    raise FileNotFoundError(required)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            scenes.append({
                "geometry_id": f"{density_dir.name}/{sample_dir.name}",
                "density": density,
                "seed": int(meta["seed"]),
                "sample_dir": sample_dir,
                "mask_path": mask_path,
                "gt_path": gt_path,
                "meta_path": meta_path,
                "run_root": run_root,
                "branches_dir": branches,
            })
    if len(scenes) != int(expected_scenes):
        raise ValueError(
            f"expected exactly {expected_scenes} development scenes, "
            f"found {len(scenes)} under {samples}"
        )
    return scenes


def _canonical_fragments(masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    canonical = []
    for mask in masks:
        thin = skeletonize(np.asarray(mask, dtype=bool))
        if np.any(thin):
            canonical.append(thin)
    return canonical


def load_skeleton_fragments(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = CM.common_fragments(mask)
    return labels, [
        labels == fragment_id
        for fragment_id in range(1, int(labels.max()) + 1)
    ]


def load_hough_fragments(run_root: Path) -> list[np.ndarray]:
    fragments = _canonical_fragments(IIO.load_fragments(run_root, min_area=15))
    if not fragments:
        raise ValueError(f"no Stage-2 Hough fragments found under {run_root}")
    return fragments


def fragment_orientation_layer(mask: np.ndarray, n_bins: int = ORIENTATION_BINS) -> int:
    """Assign a shared 0-to-180 degree PCA orientation bin."""
    points_rc = np.argwhere(mask)
    if len(points_rc) < 2:
        return 0
    points_xy = points_rc[:, ::-1].astype(float)
    centered = points_xy - points_xy.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return min(int(n_bins) - 1, int(angle / (180.0 / int(n_bins))))


def group_minimum_turn(fragments: Sequence[np.ndarray]) -> IIO.InstanceSet:
    grouped = baselines.minimum_turn_group_fragments(
        fragments,
        max_gap_px=15.0,
        tangent_window_px=8,
        max_turn_deg=90.0,
    )
    shape = tuple(int(v) for v in fragments[0].shape)
    return IIO.InstanceSet(shape, grouped, source="minimum_turn_fixed_fragments")


def grouping_kernel_config(source_config: Mapping[str, Any]) -> dict:
    """Create the fixed study config without mutating the production YAML."""
    config = copy.deepcopy(dict(source_config))
    config.setdefault("overlap", {})["kill_thr"] = 1.01
    config["overlap"]["trim_dilate_px"] = 0
    config.setdefault("runtime", {})["verbose"] = False
    config.setdefault("debug", {})["rejection_log_path"] = None
    return config


def group_filaseg_gates(
    fragments: Sequence[np.ndarray],
    source_config: Mapping[str, Any],
) -> IIO.InstanceSet:
    components = [
        RU.Component(
            id=index,
            layer=fragment_orientation_layer(mask),
            mask=np.asarray(mask, dtype=bool).copy(),
        )
        for index, mask in enumerate(fragments, start=1)
    ]
    config = grouping_kernel_config(source_config)
    for stage_key in STAGE_KEYS:
        stage_config = RR._make_stage_cfg(config, stage_key)
        stage_config.setdefault("runtime", {})["verbose"] = False
        stage_config.setdefault("debug", {})["rejection_log_path"] = None
        components = RU.reconnect_components(components, stage_config)
    masks = {
        index: component.mask.copy()
        for index, component in enumerate(
            sorted(
                (component for component in components if component.exist),
                key=lambda component: component.id,
            ),
            start=1,
        )
    }
    shape = tuple(int(v) for v in fragments[0].shape)
    return IIO.InstanceSet(shape, masks, source="filaseg_gates_fixed_fragments")


def _fixed_fragment_scores(
    common_fragments: np.ndarray,
    gt: IIO.InstanceSet,
    prediction: IIO.InstanceSet,
) -> dict:
    gt_assignment = CM.assign_fragments(common_fragments, gt)
    pred_assignment = CM.assign_fragments(common_fragments, prediction)
    scores = CM.pairwise_scores(
        gt_assignment,
        pred_assignment,
        gt_unassigned_policy="singleton",
    )
    recovery = CM.fragment_instance_recovery(gt_assignment, pred_assignment)
    scores.update({
        f"fragment_recovery_{key}": value
        for key, value in recovery.items()
    })
    return scores


def _source_diagnostics(
    common_fragments: np.ndarray,
    source_fragments: Sequence[np.ndarray],
) -> dict:
    source = IIO.InstanceSet(
        common_fragments.shape,
        {index: mask for index, mask in enumerate(source_fragments, start=1)},
        source="source_fragments",
    )
    assignment = CM.assign_fragments(common_fragments, source)
    assigned = sum(int(value) != 0 for value in assignment.values())
    union = np.logical_or.reduce(source_fragments)
    common_support = common_fragments != 0
    return {
        "n_source_fragments": len(source_fragments),
        "n_common_fragments": int(common_fragments.max()),
        "n_common_fragments_assigned": assigned,
        "common_fragment_assignment_rate": assigned / max(1, len(assignment)),
        "source_union_pixels": int(union.sum()),
        "common_support_pixels": int(common_support.sum()),
        "common_support_pixel_recall": (
            int((union & common_support).sum()) / max(1, int(common_support.sum()))
        ),
        "source_union_pixel_precision": (
            int((union & common_support).sum()) / max(1, int(union.sum()))
        ),
    }


def _save_multilabel(path: Path, instances: IIO.InstanceSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(sorted(instances.masks), dtype=np.int32)
    flats = [
        np.flatnonzero(instances.masks[int(instance_id)].ravel()).astype(np.int64)
        for instance_id in ids
    ]
    lengths = np.asarray([len(flat) for flat in flats], dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    indices = (
        np.concatenate(flats)
        if flats
        else np.asarray([], dtype=np.int64)
    )
    np.savez_compressed(
        path,
        shape=np.asarray(instances.shape, dtype=np.int64),
        ids=ids,
        indptr=indptr,
        indices=indices,
    )


def _summaries(per_scene: Sequence[Mapping[str, Any]]) -> dict:
    stratum_by_scene = {
        row["geometry_id"]: str(row["density"])
        for row in per_scene
    }
    values = {
        cell: {
            metric: {
                row["geometry_id"]: float(row["scores"][cell][metric])
                for row in per_scene
            }
            for metric in METRICS
        }
        for cell in CELLS
    }
    by_cell = {}
    for cell in CELLS:
        by_cell[cell] = {}
        for metric in METRICS:
            by_density: Dict[str, list[float]] = {}
            for scene_id, value in values[cell][metric].items():
                by_density.setdefault(stratum_by_scene[scene_id], []).append(value)
            by_cell[cell][metric] = ST.stratified_summarize(
                by_density
            ).as_dict()

    contrast_pairs = {
        "filaseg_gate_effect_with_skeleton_fragments": (
            "skeleton_filaseg_gates", "skeleton_minimum_turn"
        ),
        "filaseg_gate_effect_with_hough_fragments": (
            "hough_filaseg_gates", "hough_minimum_turn"
        ),
        "hough_source_effect_under_minimum_turn": (
            "hough_minimum_turn", "skeleton_minimum_turn"
        ),
        "hough_source_effect_under_filaseg_gates": (
            "hough_filaseg_gates", "skeleton_filaseg_gates"
        ),
    }
    contrasts = {}
    for name, (left, right) in contrast_pairs.items():
        contrasts[name] = {
            metric: ST.paired_stratified_summarize_by_scene(
                values[left][metric],
                values[right][metric],
                stratum_by_scene,
            ).as_dict()
            for metric in METRICS
        }

    interaction = {}
    for metric in METRICS:
        hough_effect = {
            scene_id: (
                values["hough_filaseg_gates"][metric][scene_id]
                - values["hough_minimum_turn"][metric][scene_id]
            )
            for scene_id in stratum_by_scene
        }
        skeleton_effect = {
            scene_id: (
                values["skeleton_filaseg_gates"][metric][scene_id]
                - values["skeleton_minimum_turn"][metric][scene_id]
            )
            for scene_id in stratum_by_scene
        }
        interaction[metric] = ST.paired_stratified_summarize_by_scene(
            hough_effect,
            skeleton_effect,
            stratum_by_scene,
        ).as_dict()
    contrasts[
        "fragment_source_by_grouper_interaction"
    ] = interaction
    return {"by_cell": by_cell, "contrasts": contrasts}


def run_study(
    samples_root: Path | str,
    hough_root: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    expected_scenes: int = 20,
) -> dict:
    samples = _reject_locked_path(samples_root)
    hough = _reject_locked_path(hough_root)
    output = _reject_locked_path(output_dir)
    config_file = _reject_locked_path(config_path)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing study directory: {output}"
        )
    scenes = discover_development_scenes(
        samples, hough, expected_scenes=expected_scenes
    )
    source_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output.mkdir(parents=True)

    source_files = (
        ROOT / "eval" / "studies" / "hybrid_grouping_factorial.py",
        ROOT / "eval" / "baseline" / "baselines.py",
        ROOT / "eval" / "core" / "common_metric.py",
        ROOT / "eval" / "core" / "instance_io.py",
        ROOT / "eval" / "stats" / "stats_util.py",
        ROOT / "3.reconnect" / "reconnect_run.py",
        ROOT / "3.reconnect" / "reconnect_utils_straight.py",
        config_file,
    )
    per_scene = []
    for scene_index, scene in enumerate(scenes, start=1):
        print(
            f"[{scene_index:02d}/{len(scenes):02d}] {scene['geometry_id']}",
            flush=True,
        )
        mask = _read_mask(scene["mask_path"])
        common_fragments, skeleton_fragments = load_skeleton_fragments(mask)
        hough_fragments = load_hough_fragments(scene["run_root"])
        gt, gt_meta = IIO.load_gt(scene["gt_path"], shape=mask.shape)
        predictions = {
            "skeleton_minimum_turn": group_minimum_turn(skeleton_fragments),
            "skeleton_filaseg_gates": group_filaseg_gates(
                skeleton_fragments, source_config
            ),
            "hough_minimum_turn": group_minimum_turn(hough_fragments),
            "hough_filaseg_gates": group_filaseg_gates(
                hough_fragments, source_config
            ),
        }

        artifact_records = {}
        for cell, prediction in predictions.items():
            artifact = (
                output / "artifacts" / scene["geometry_id"]
                / f"{cell}.npz"
            )
            _save_multilabel(artifact, prediction)
            artifact_records[cell] = {
                "path": str(artifact),
                "sha256": _sha256(artifact),
                "n_instances": prediction.n,
            }
        branch_files = sorted(
            scene["branches_dir"].glob("*branch_*.png")
        )
        per_scene.append({
            "geometry_id": scene["geometry_id"],
            "density": scene["density"],
            "seed": scene["seed"],
            "ground_truth_kind": gt_meta["kind"],
            "inputs": {
                "mask": {
                    "path": str(scene["mask_path"]),
                    "sha256": _sha256(scene["mask_path"]),
                },
                "ground_truth": {
                    "path": str(scene["gt_path"]),
                    "sha256": _sha256(scene["gt_path"]),
                },
                "metadata": {
                    "path": str(scene["meta_path"]),
                    "sha256": _sha256(scene["meta_path"]),
                },
                "hough_branch_files": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in branch_files
                ],
            },
            "source_diagnostics": {
                "skeleton": _source_diagnostics(
                    common_fragments, skeleton_fragments
                ),
                "hough": _source_diagnostics(
                    common_fragments, hough_fragments
                ),
            },
            "orientation_layer_counts": {
                "skeleton": np.bincount(
                    [
                        fragment_orientation_layer(fragment)
                        for fragment in skeleton_fragments
                    ],
                    minlength=ORIENTATION_BINS,
                ).tolist(),
                "hough": np.bincount(
                    [
                        fragment_orientation_layer(fragment)
                        for fragment in hough_fragments
                    ],
                    minlength=ORIENTATION_BINS,
                ).tolist(),
            },
            "scores": {
                cell: _fixed_fragment_scores(common_fragments, gt, prediction)
                for cell, prediction in predictions.items()
            },
            "artifacts": artifact_records,
        })

    report = {
        "schema_version": 1,
        "study": "development_hybrid_grouping_factorial",
        "development_only": True,
        "exploratory": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "n_scenes": len(per_scene),
            "density_strata": sorted({row["density"] for row in per_scene}),
            "cells": list(CELLS),
            "fixed_scoring_population": (
                "input-derived common fragments, computed once per scene"
            ),
            "gt_unassigned_policy": "singleton",
            "orientation_bins": ORIENTATION_BINS,
            "orientation_assignment": "shared per-fragment PCA, 0 to 180 degrees",
            "minimum_turn_parameters": {
                "max_gap_px": 15.0,
                "tangent_window_px": 8,
                "max_turn_deg": 90.0,
            },
            "filaseg_stage_order": list(STAGE_KEYS),
            "overlap_suppression": "disabled in copied study config",
            "stage4_rendering": False,
            "tuning_performed": False,
            "bootstrap": (
                "paired complete four-cell scene blocks resampled within density"
            ),
            "bootstrap_seed": ST.BOOTSTRAP_SEED,
            "bootstrap_replicates": ST.N_BOOTSTRAP,
        },
        "paths": {
            "samples_root": str(samples),
            "hough_root": str(hough),
            "output_dir": str(output),
            "config_source": str(config_file),
        },
        "source_hashes": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                _sha256(path)
            for path in source_files
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cwd": str(Path.cwd()),
        },
        "per_scene": per_scene,
        "aggregate": _summaries(per_scene),
        "interpretation_guard": (
            "Development-only grouping-kernel evidence. It neither changes "
            "the locked evaluation nor supports a new locked-set claim."
        ),
    }
    report_path = output / "results.json"
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_hash = _sha256(report_path)
    (output / "results.sha256").write_text(
        f"{report_hash}  results.json\n", encoding="ascii"
    )
    print(f"wrote {report_path}", flush=True)
    print(f"sha256 {report_hash}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Development-only 2x2 fragment-source/grouping study"
    )
    parser.add_argument(
        "--samples",
        default=str(ROOT / "input" / "synthetic_thick"),
    )
    parser.add_argument(
        "--hough-root",
        default=str(ROOT / "output" / "final_development_w2"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "output" / "development_hybrid_grouping_factorial"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "3.reconnect" / "reconnect_config.yaml"),
    )
    parser.add_argument("--expected-scenes", type=int, default=20)
    args = parser.parse_args(argv)
    run_study(
        args.samples,
        args.hough_root,
        args.output,
        args.config,
        expected_scenes=args.expected_scenes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
