"""Reproducible, scene-paired evaluator for the synthetic FilaSeg experiments.

The runner deliberately keeps the locked set boring: it runs fixed baselines,
or the normal pipeline CLI without forwarding any tuning flags, records every
failure as a row, and scores all successful rows in the same process.  A
baseline grid is available only for development data and is ranked by actual
common-fragment F1, never by an unlabelled proxy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import tifffile
from skimage import io as skio
from skimage.measure import label as cc_label

# Work both as ``python eval/experiment_runner.py`` and as an imported module.
HERE = Path(__file__).resolve().parents[1]  # eval/
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import _evalpath  # noqa: F401,E402  (restores the flat eval/ import namespace)
import baselines  # noqa: E402
import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402
import metrics as M  # noqa: E402
import stats_util  # noqa: E402

METHODS = ("baseline_cc", "baseline_skeleton")
GT_UNASSIGNED_POLICY = "singleton"
ROW_REQUIRED = {"geometry_id", "density", "seed", "mask_variant", "method", "status", "output_path"}


def _sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _jsonable(value: Any) -> Any:
    """Make NumPy values JSON-safe without rounding finite measurements."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _density_from_path(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        m = re.fullmatch(r"(?:cov|density)[_-]?(\d+)", part, re.I)
        if m:
            return int(m.group(1))
    return None


def _variant(mask_name: str, explicit: Optional[str]) -> str:
    if explicit:
        return str(explicit)
    stem = Path(mask_name).stem.lower()
    if "clean" in stem:
        return "clean"
    m = re.search(r"(?:mask[_-]?)?w(?:idth)?[_-]?(\d+)", stem)
    return m.group(1) if m else stem


def discover_scenes(samples_root: Path | str, mask_name: str = "mask.png",
                    mask_variant: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover scene directories deterministically, with identity from metadata.

    A scene must have the named mask. A missing ``gt_labels.tif`` is retained
    and becomes a failed evaluation row rather than silently disappearing.
    The geometry ID is its POSIX relative path, so ``cov20/sample_000`` cannot
    collide with a same-named scene in another density. ``gt_meta.json``
    supplies the exact seed when present; missing metadata is seed ``None``.
    """
    root = Path(samples_root).resolve()
    found: List[Dict[str, Any]] = []
    if not root.is_dir():
        return found
    for candidate in sorted(root.rglob(mask_name), key=lambda p: p.as_posix()):
        scene = candidate.parent
        gt = scene / "gt_labels.tif"
        meta_path = scene / "gt_meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                # Discovery itself remains complete; evaluation will expose this
                # as a failed row rather than silently hiding the scene.
                meta = {"_metadata_error": f"{type(exc).__name__}: {exc}"}
        density = meta.get("density", meta.get("coverage", _density_from_path(scene.relative_to(root))))
        if isinstance(density, float) and 0 < density <= 1:
            density = int(round(density * 100))
        found.append({
            "scene_dir": scene,
            "mask_path": candidate,
            "gt_path": gt,
            "sem_path": scene / "sem.png",
            "geometry_id": scene.relative_to(root).as_posix(),
            "density": int(density) if density is not None else None,
            "seed": meta.get("seed"),
            "mask_variant": _variant(mask_name, mask_variant),
            "meta": meta,
        })
    return found


def _read_mask(path: Path) -> np.ndarray:
    arr = skio.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return np.asarray(arr) > 0


def _native_fragments(mask: np.ndarray) -> List[np.ndarray]:
    labels = cc_label(mask, connectivity=2)
    return [labels == i for i in range(1, int(labels.max()) + 1)]


def _score(mask: np.ndarray, gt_path: Path, pred: Any,
           native_fragments: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """Score labels or an overlap-aware :class:`InstanceSet` without flattening.

    ``load_pred_instances`` deliberately prefers multilabel NPZ files.  Passing
    that InstanceSet all the way through is important: flattening it before
    common_metric would discard the overlap ownership the pipeline preserved.
    """
    common = CM.score_method(
        mask, gt_path, pred, gt_unassigned_policy=GT_UNASSIGNED_POLICY
    )
    gt, meta = IIO.load_gt(gt_path, shape=mask.shape)
    if isinstance(pred, np.ndarray):
        pred = IIO.instances_from_label_image(pred.astype(np.int32))
    if tuple(pred.shape) != tuple(mask.shape):
        raise ValueError(f"prediction shape {pred.shape} != mask shape {mask.shape}")
    frags = native_fragments if native_fragments is not None else _native_fragments(mask)
    native = M.evaluate(pred, gt, frags, gt_kind=meta["kind"]).to_dict()
    # score_method already computed both recovery blocks from this exact
    # common-fragment evaluation. Do not independently recompute the raw pixel
    # diagnostic: that risks drift and obscures which representation it uses.
    shared_fragment_recovery = {
        key.removeprefix("fragment_recovery_"): value
        for key, value in common.items() if key.startswith("fragment_recovery_")
    }
    representation_dependent_pixel_recovery = {
        key.removeprefix("pixel_recovery_"): value
        for key, value in common.items() if key.startswith("pixel_recovery_")
    }
    return _jsonable({"common_metric": common,
                      "shared_fragment_recovery": shared_fragment_recovery,
                      "representation_dependent_pixel_recovery": representation_dependent_pixel_recovery,
                      "native_diagnostics": native})


def _row_base(scene: Dict[str, Any], method: str, output_path: Path,
              config_hashes: Dict[str, Optional[str]], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "geometry_id": scene["geometry_id"], "density": scene["density"],
        "seed": scene["seed"], "mask_variant": scene["mask_variant"],
        "method": method, "output_path": str(output_path),
        "config_hashes": config_hashes, "parameters": params or {},
    }


def _run_baseline(scene: Dict[str, Any], method: str, outputs: Path,
                  config_hashes: Dict[str, Optional[str]], params: Dict[str, Any]) -> Dict[str, Any]:
    # Parameterised names prevent a development grid from overwriting the
    # artifact that makes a previously completed row resumable.
    param_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out = outputs / scene["geometry_id"] / scene["mask_variant"] / f"{method}__{param_id}.tif"
    row = _row_base(scene, method, out, config_hashes, params)
    try:
        if scene["meta"].get("_metadata_error"):
            raise ValueError(scene["meta"]["_metadata_error"])
        mask = _read_mask(scene["mask_path"])
        if method == "baseline_cc":
            labels = baselines.baseline_connected_components(mask, min_area=int(params.get("min_area", 15)))
        elif method == "baseline_skeleton":
            labels = baselines.baseline_skeleton_minturn(mask, **params)
        else:
            raise ValueError(f"unknown baseline method {method!r}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out), np.asarray(labels, dtype=np.int32))
        row.update(status="ok", scores=_score(mask, scene["gt_path"], labels.astype(np.int32)))
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=8))
    return _jsonable(row)


def _run_filaseg(scene: Dict[str, Any], output_root: Path, mode: str,
                 config_hashes: Dict[str, Optional[str]], reconnect_config: Path) -> Dict[str, Any]:
    target = output_root / scene["geometry_id"] / scene["mask_variant"]
    row = _row_base(scene, "filaseg", target, config_hashes, {"filaseg_mode": mode})
    try:
        if mode == "run":
            if not scene["sem_path"].is_file():
                raise FileNotFoundError(f"pipeline requires background image: {scene['sem_path']}")
            target.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "Tools" / "run_full_sem_pipeline.py"),
                   "--mask", str(scene["mask_path"]), "--background", str(scene["sem_path"]),
                   "--base", "filaseg", "--output-root", str(target),
                   "--reconnect-config", str(reconnect_config)]
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
            stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
            (target / "pipeline_stdout.log").write_text(stdout, encoding="utf-8")
            if proc.returncode:
                raise RuntimeError(f"pipeline return code {proc.returncode}; see {target / 'pipeline_stdout.log'}")
            final_lines = [line.split("final:", 1)[1].strip() for line in stdout.splitlines() if "final:" in line]
            if len(final_lines) != 1:
                raise RuntimeError("pipeline did not report exactly one final output directory")
            run_dir = Path(final_lines[0]).resolve().parent
            artifact = IIO.find_pred_artifact(run_dir)
        else:
            run_dir = None
        # ``run`` has one prescribed location.  ``reuse`` also accepts the
        # two layouts emitted by older batch jobs, without guessing an output.
        candidates = [] if mode == "run" else [target]
        if mode == "reuse":
            candidates += [output_root / scene["geometry_id"], output_root / scene["scene_dir"].name]
        if mode != "run":
            artifact = next((IIO.find_pred_artifact(p) for p in candidates if _has_pred_artifact(p)), None)
            run_dir = _artifact_run_dir(artifact) if artifact is not None else None
        if artifact is None or run_dir is None:
            raise FileNotFoundError(f"no reusable FilaSeg output for {scene['geometry_id']} under {output_root}")
        mask = _read_mask(scene["mask_path"])
        pred = IIO.load_pred_instances(artifact)
        if pred.shape != mask.shape:
            raise ValueError(f"pipeline output shape {pred.shape} != mask shape {mask.shape}")
        row["output_path"] = str(artifact)
        native = IIO.load_fragments(run_dir, min_area=15)
        row.update(status="ok", scores=_score(mask, scene["gt_path"], pred, native or None))
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=8))
    return _jsonable(row)


def _has_pred_artifact(path: Path) -> bool:
    try:
        IIO.find_pred_artifact(path)
        return True
    except FileNotFoundError:
        return False


def _artifact_run_dir(artifact: Path) -> Path:
    """Return a conventional run root for a selected final/stage artifact."""
    artifact = Path(artifact)
    if artifact.parent.name == "final":
        return artifact.parent.parent
    for parent in artifact.parents:
        if parent.name in {"3.reconnect", "4.postprocess"}:
            return parent.parent
    return artifact.parent


def row_complete(row: Dict[str, Any], config_hashes: Optional[Dict[str, Optional[str]]] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> bool:
    """A resumable success must match current code/config/parameters exactly."""
    return (ROW_REQUIRED <= set(row) and row.get("status") == "ok" and
            isinstance(row.get("scores"), dict) and
            isinstance(row["scores"].get("common_metric"), dict) and
            (config_hashes is None or row.get("config_hashes") == config_hashes) and
            (parameters is None or row.get("parameters", {}) == parameters) and
            Path(row["output_path"]).is_file())


def _prior_rows(path: Path) -> Dict[tuple, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("per_scene", [])
    except Exception:
        return {}
    return {(r.get("geometry_id"), r.get("mask_variant"), r.get("method"),
             json.dumps(r.get("parameters", {}), sort_keys=True)): r for r in rows}


def _method_key(row: Dict[str, Any]) -> str:
    params = row.get("parameters", {})
    return str(row.get("method")) if not params else f"{row.get('method')}:{json.dumps(params, sort_keys=True)}"


def _summary_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Density-stratified scene summaries with parameter grids kept distinct."""
    methods = sorted({_method_key(r) for r in rows})
    by_method: Dict[str, Any] = {}
    good: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        rs = [r for r in rows if _method_key(r) == method]
        successes = [r for r in rs if r.get("status") == "ok" and isinstance(r.get("scores"), dict)]
        f1 = {f"{r['geometry_id']}|{r['mask_variant']}": float(r["scores"]["common_metric"]["f1"])
              for r in successes if r["scores"].get("common_metric", {}).get("f1") is not None}
        recovery = {f"{r['geometry_id']}|{r['mask_variant']}": float(
            r["scores"]["common_metric"]["fragment_recovery_recovery_rate"])
            for r in successes if r["scores"].get("common_metric", {}).get("fragment_recovery_recovery_rate") is not None}
        good[method] = {"common_metric.f1": f1,
                        "common_metric.fragment_recovery_recovery_rate": recovery}
        strata = {
            f"{r['geometry_id']}|{r['mask_variant']}": str(r.get("density"))
            for r in successes
        }
        f1_by_density: Dict[str, List[float]] = {}
        recovery_by_density: Dict[str, List[float]] = {}
        for scene_id, value in f1.items():
            f1_by_density.setdefault(strata[scene_id], []).append(value)
        for scene_id, value in recovery.items():
            recovery_by_density.setdefault(strata[scene_id], []).append(value)
        by_method[method] = {"n_rows": len(rs), "n_success": len(successes),
                             "n_failed": sum(r.get("status") == "failed" for r in rs),
                             "common_f1": _jsonable(
                                 stats_util.stratified_summarize(f1_by_density).as_dict()),
                             "shared_fragment_recovery_rate": _jsonable(
                                 stats_util.stratified_summarize(
                                     recovery_by_density).as_dict())}
    paired: List[Dict[str, Any]] = []
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            for metric in ("common_metric.f1", "common_metric.fragment_recovery_recovery_rate"):
                left_values, right_values = good[left][metric], good[right][metric]
                ids = sorted(set(left_values) & set(right_values))
                paired_strata = {
                    scene_id: str(next(
                        r.get("density") for r in rows
                        if f"{r['geometry_id']}|{r['mask_variant']}" == scene_id
                    ))
                    for scene_id in ids
                }
                paired.append({"method_a": left, "method_b": right, "metric": metric,
                               "n_paired_scenes": len(ids), "n_missing_a": len(set(right_values) - set(left_values)),
                               "n_missing_b": len(set(left_values) - set(right_values)),
                               "a_minus_b": _jsonable(
                                   stats_util.paired_stratified_summarize_by_scene(
                                       {x: left_values[x] for x in ids},
                                       {x: right_values[x] for x in ids},
                                       paired_strata,
                                   ).as_dict())})
    return {"by_method": by_method, "paired": paired,
            "failure_count": sum(r.get("status") == "failed" for r in rows)}


def _summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall plus per-density (including exact-geometry paired) summaries."""
    out = _summary_block(rows)
    densities = sorted({r.get("density") for r in rows}, key=lambda x: (x is None, x))
    out["by_density"] = {
        str(density): _summary_block([r for r in rows if r.get("density") == density])
        for density in densities
    }
    return out


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    # CSV is intentionally a narrow, lossless numeric projection; complete
    # diagnostics remain nested in JSON rather than being string-rounded.
    keys = ["geometry_id", "density", "seed", "mask_variant", "method", "status", "output_path", "error",
            "parameters", "common_f1", "common_precision", "common_recall", "fragment_recovery_rate",
            "pixel_recovery_rate", "representation_dependent_pixel_recovery_rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cm = row.get("scores", {}).get("common_metric", {})
            shared = row.get("scores", {}).get("shared_fragment_recovery", {})
            pixel = row.get("scores", {}).get("representation_dependent_pixel_recovery", {})
            writer.writerow({**row, "parameters": json.dumps(row.get("parameters", {}), sort_keys=True),
                             "common_f1": cm.get("f1"), "common_precision": cm.get("precision"),
                             "common_recall": cm.get("recall"),
                             "fragment_recovery_rate": shared.get("recovery_rate"),
                             # This compatibility-named column is explicitly
                             # representation-dependent; it is never used in
                             # a comparative aggregate.
                             "pixel_recovery_rate": pixel.get("recovery_rate"),
                             "representation_dependent_pixel_recovery_rate": pixel.get("recovery_rate")})


def _locked(path: Path) -> bool:
    return "input/synthetic_locked_v1" in path.resolve().as_posix().lower()


def run_experiment(samples_root: Path | str, out_json: Path | str, *, mask_name: str = "mask.png",
                   mask_variant: Optional[str] = None, methods: Sequence[str] = METHODS,
                   filaseg: str = "off", filaseg_root: Optional[Path | str] = None,
                   reconnect_config: Optional[Path | str] = None,
                   resume: bool = False, cc_min_area: int = 15,
                   max_gap_px: float = 15.0, max_turn_deg: float = 90.0,
                   sweep_gaps: Optional[Sequence[float]] = None,
                   sweep_turns: Optional[Sequence[float]] = None,
                   configuration_locked: bool = False) -> Dict[str, Any]:
    root = Path(samples_root)
    out_json = Path(out_json)
    methods = tuple(methods)
    if filaseg not in {"off", "reuse", "run"}:
        raise ValueError(f"unknown FilaSeg mode {filaseg!r}")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown baseline method(s): {', '.join(unknown)}")
    if not methods and filaseg == "off":
        raise ValueError("select at least one baseline method or enable FilaSeg")
    if _locked(root) and (sweep_gaps or sweep_turns):
        raise ValueError("baseline tuning/sweeps are forbidden for input/synthetic_locked_v1")
    if _locked(root) and not configuration_locked and (
            cc_min_area != 15 or max_gap_px != 15.0 or max_turn_deg != 90.0):
        raise ValueError(
            "non-default fixed parameters on input/synthetic_locked_v1 require "
            "--configuration-locked; tuning/sweeps remain forbidden"
        )
    scenes = discover_scenes(root, mask_name, mask_variant)
    if not scenes:
        raise ValueError(f"no scenes found under {root} with mask {mask_name!r}")
    reconnect_source = Path(reconnect_config or (ROOT / "3.reconnect" / "reconnect_config.yaml")).resolve()
    if not reconnect_source.is_file():
        raise ValueError(f"reconnect config does not exist: {reconnect_source}")
    hashes = {p.name: _sha256(p) for p in (HERE / "experiment_runner.py", HERE / "baselines.py", HERE / "common_metric.py",
              HERE / "instance_io.py", HERE / "metrics.py", HERE / "stats_util.py", HERE / "run_batch.py",
              ROOT / "Tools" / "run_full_sem_pipeline.py")}
    hashes["reconnect_config_source"] = _sha256(reconnect_source)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                  "platform": platform.platform(), "numpy": np.__version__, "cwd": str(Path.cwd())}
    previous = _prior_rows(out_json) if resume else {}
    base_outputs = out_json.parent / "outputs"
    rows: List[Dict[str, Any]] = []
    grid = bool(sweep_gaps or sweep_turns)
    if grid:
        gaps = sorted({float(v) for v in (sweep_gaps or [max_gap_px])})
        turns = sorted({float(v) for v in (sweep_turns or [max_turn_deg])})
        jobs = [("baseline_skeleton", {"max_gap_px": g, "max_turn_deg": t}) for g in gaps for t in turns]
    else:
        jobs = [(m, {"min_area": cc_min_area} if m == "baseline_cc" else
                      {"max_gap_px": max_gap_px, "max_turn_deg": max_turn_deg}) for m in methods]
    for scene in scenes:
        for method, params in jobs:
            key = (scene["geometry_id"], scene["mask_variant"], method, json.dumps(params, sort_keys=True))
            old = previous.get(key)
            if old is not None and row_complete(old, hashes, params):
                rows.append(old)
            else:
                rows.append(_run_baseline(scene, method, base_outputs, hashes, params))
        if not grid and filaseg != "off":
            froot = Path(filaseg_root) if filaseg_root else base_outputs / "filaseg"
            filaseg_params = {"filaseg_mode": filaseg}
            key = (scene["geometry_id"], scene["mask_variant"], "filaseg", json.dumps(filaseg_params, sort_keys=True))
            old = previous.get(key)
            rows.append(old if old is not None and row_complete(old, hashes, filaseg_params) else
                        _run_filaseg(scene, froot, filaseg, hashes, reconnect_source))
    aggregate = _summaries(rows)
    document: Dict[str, Any] = {"schema_version": 2, "samples_root": str(root.resolve()),
        "configuration": {"mask_name": mask_name, "mask_variant": mask_variant, "methods": list(methods),
            "filaseg": filaseg, "cc_min_area": cc_min_area, "max_gap_px": max_gap_px,
            "max_turn_deg": max_turn_deg, "sweep_gaps": list(sweep_gaps or []), "sweep_turns": list(sweep_turns or []),
            "configuration_locked": configuration_locked,
            "gt_unassigned_policy": GT_UNASSIGNED_POLICY,
            "overall_bootstrap": "within_density_then_scene_weighted_pooling",
            "reconnect_config_source": str(reconnect_source)},
        "config_hashes": hashes, "provenance": provenance, "per_scene": rows, "aggregate": aggregate}
    if grid:
        ranked = []
        for params in sorted({json.dumps(r.get("parameters", {}), sort_keys=True) for r in rows}):
            all_group = [r for r in rows if json.dumps(r.get("parameters", {}), sort_keys=True) == params]
            group = [r for r in all_group if r.get("status") == "ok"]
            # A one-fragment scene has no pair evidence, so common_metric
            # correctly returns a null F1. It remains an explicit successful
            # row, but cannot contribute to a pairwise-F1 ranking.
            values = [float(r["scores"]["common_metric"]["f1"]) for r in group
                      if r["scores"]["common_metric"].get("f1") is not None]
            p = json.loads(params)
            ranked.append({"parameters": p, "n_total": len(all_group), "n_success": len(group),
                "n_failed": sum(r.get("status") == "failed" for r in all_group),
                "n_pair_evidence": len(values),
                "mean_common_f1": float(np.mean(values)) if values else None})
        ranked.sort(key=lambda r: (r["n_failed"] > 0, r["mean_common_f1"] is None, -(r["mean_common_f1"] or -np.inf),
                                   r["parameters"]["max_gap_px"], r["parameters"]["max_turn_deg"]))
        document["baseline_grid"] = {"ranking_rule": "complete parameter sets rank before any set with failures; then descending mean common_metric.f1 over scenes with pair evidence; ties: lower max_gap_px, then lower max_turn_deg",
                                     "ranked": ranked}
    document = _jsonable(document)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(rows, out_json.with_suffix(".csv"))
    return document


def _floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Reproducible paired synthetic-evaluation runner")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True, help="JSON report path (CSV is written beside it)")
    ap.add_argument("--mask-name", default="mask.png")
    ap.add_argument("--mask-variant")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--filaseg", choices=("off", "reuse", "run"), default="off")
    ap.add_argument("--filaseg-root")
    ap.add_argument("--reconnect-config", help="read-only reconnect YAML passed to FilaSeg run mode")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cc-min-area", type=int, default=15)
    ap.add_argument("--max-gap-px", type=float, default=15.0)
    ap.add_argument("--max-turn-deg", type=float, default=90.0)
    ap.add_argument("--sweep-gaps", help="development-only comma-separated baseline grid axis")
    ap.add_argument("--sweep-turns", help="development-only comma-separated baseline grid axis")
    ap.add_argument("--configuration-locked", action="store_true",
                    help="attest that fixed parameters were selected before evaluating synthetic_locked_v1")
    args = ap.parse_args(argv)
    try:
        doc = run_experiment(args.samples, args.out, mask_name=args.mask_name, mask_variant=args.mask_variant,
            methods=[x for x in args.methods.split(",") if x], filaseg=args.filaseg, filaseg_root=args.filaseg_root,
            reconnect_config=args.reconnect_config,
            resume=args.resume, cc_min_area=args.cc_min_area, max_gap_px=args.max_gap_px,
            max_turn_deg=args.max_turn_deg, sweep_gaps=_floats(args.sweep_gaps) if args.sweep_gaps else None,
            sweep_turns=_floats(args.sweep_turns) if args.sweep_turns else None,
            configuration_locked=args.configuration_locked)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"wrote {args.out}: {len(doc['per_scene'])} rows, {doc['aggregate']['failure_count']} failures")
    return 1 if doc["aggregate"]["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
