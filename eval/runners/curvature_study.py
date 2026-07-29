"""Development-only orchestration for the reconnect curvature/q-fit study.

This runner is deliberately scoped to ``input/synthetic_thick/cov*/synth_*``.
It refuses the locked synthetic set, writes one immutable-derived YAML per
variant, and keeps failures as first-class result rows so a study is auditable
and safely resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]  # eval/
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import _evalpath  # noqa: F401,E402  (restores the flat eval/ import namespace)
import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402
import metrics as M  # noqa: E402
import stats_util  # noqa: E402


SYNTHETIC_ROOT = ROOT / "input" / "synthetic_thick"
PRODUCTION_CONFIG = ROOT / "3.reconnect" / "reconnect_config.yaml"
PIPELINE = ROOT / "Tools" / "run_full_sem_pipeline.py"
SHARED_REJECTION_LOG = ROOT / "3.reconnect" / "output" / "rejection_log_straight.csv"
STAGES = ("stage_clear", "stage_strict", "stage_relaxed")
Q_FIT_THRESHOLDS = (0.005, 0.010, 0.020, 0.040, 0.080)


def sha256(path: Path) -> Optional[str]:
    """Return a file digest, or ``None`` for a missing path."""
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _contains_locked(path: Path) -> bool:
    return any(part.lower() == "synthetic_locked_v1" for part in Path(path).resolve().parts)


def reject_locked(path: Path) -> None:
    """Reject the locked synthetic holdout before any study artifacts exist."""
    if _contains_locked(path):
        raise ValueError("synthetic_locked_v1 is forbidden for curvature development studies")


def _density(scene: Path) -> int:
    match = re.fullmatch(r"cov(\d+)", scene.parent.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"scene is not directly under cov*/: {scene}")
    return int(match.group(1))


def discover_synthetic_scenes(root: Path | str = SYNTHETIC_ROOT) -> list[Dict[str, Any]]:
    """Discover only direct ``cov*/synth_*`` development scenes, in order."""
    root = Path(root).resolve()
    reject_locked(root)
    if not root.is_dir():
        return []
    scenes: list[Dict[str, Any]] = []
    # The literal glob is intentional: it excludes every other synthetic
    # family, including all locked or experimental layouts.
    locked = next((path for path in root.rglob("*") if _contains_locked(path)), None)
    if locked is not None:
        reject_locked(locked)
    for scene in sorted(root.rglob("synth_*"), key=lambda p: p.as_posix()):
        # The traversal is recursive, but eligibility is exactly the required
        # development layout: <root>/cov*/synth_* and nothing else.
        try:
            relative = scene.relative_to(root)
        except ValueError:
            continue
        if not scene.is_dir() or len(relative.parts) != 2 or not re.fullmatch(r"cov\d+", scene.parent.name, re.I):
            continue
        reject_locked(scene)
        scenes.append({
            "case_type": "synthetic",
            "scene_id": scene.relative_to(root).as_posix(),
            "density": _density(scene),
            "scene_dir": scene,
            "mask_path": scene / "mask_w1.png",
            "background_path": scene / "sem.png",
            "gt_path": scene / "gt_labels.tif",
        })
    return scenes


def default_variants() -> list[Dict[str, Any]]:
    """The fixed development grid; zero disables the q-fit gate."""
    variants = [
        {"name": "legacy_mixed", "curvature_measure": "legacy_mixed", "max_tip_fit_quality": 0.0},
        {"name": "geometric_disabled", "curvature_measure": "geometric", "max_tip_fit_quality": 0.0},
    ]
    variants.extend({
        "name": f"geometric_qfit_{threshold:.3f}",
        "curvature_measure": "geometric",
        "max_tip_fit_quality": threshold,
    } for threshold in Q_FIT_THRESHOLDS)
    return variants


def write_variant_yaml(source_config: Path | str, destination: Path | str,
                       variant: Dict[str, Any]) -> Path:
    """Derive a variant YAML without ever modifying the source config."""
    source = Path(source_config).resolve()
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"reconnect source config is not a file: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("variant YAML destination must differ from the source config")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"reconnect source config must be a YAML mapping: {source}")
    for stage in STAGES:
        thresholds = cfg.setdefault(stage, {}).setdefault("thresholds", {})
        thresholds["curvature_measure"] = str(variant["curvature_measure"])
        thresholds["max_tip_fit_quality"] = float(variant["max_tip_fit_quality"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return destination


def write_variant_configs(source_config: Path | str, config_dir: Path | str,
                          variants: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Path]:
    """Write deterministic study configs and return them keyed by variant name."""
    configs: Dict[str, Path] = {}
    for variant in variants or default_variants():
        configs[str(variant["name"])] = write_variant_yaml(
            source_config, Path(config_dir) / f"{variant['name']}.yaml", variant,
        )
    return configs


def code_hashes() -> Dict[str, Optional[str]]:
    return {
        "curvature_study.py": sha256(HERE / "curvature_study.py"),
        "run_full_sem_pipeline.py": sha256(PIPELINE),
        "common_metric.py": sha256(HERE / "common_metric.py"),
        "instance_io.py": sha256(HERE / "instance_io.py"),
        "metrics.py": sha256(HERE / "metrics.py"),
    }


def make_resume_key(scene: Dict[str, Any], variant: Dict[str, Any], config_path: Path,
                    hashes: Dict[str, Optional[str]], prediction_artifact: Optional[Path] = None) -> str:
    """Hash every code/config input that makes a run result reusable."""
    payload = {
        "scene_id": scene["scene_id"], "case_type": scene["case_type"],
        "variant": variant, "config_sha256": sha256(config_path), "code_hashes": hashes,
        "mask_sha256": sha256(scene["mask_path"]), "background_sha256": sha256(scene["background_path"]),
        "gt_sha256": sha256(scene["gt_path"]),
        "prediction_artifact_exists": prediction_artifact is not None and Path(prediction_artifact).is_file(),
        "prediction_artifact_sha256": sha256(prediction_artifact) if prediction_artifact is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resume_matches(row: Dict[str, Any], resume_key: str) -> bool:
    """Reuse only a complete successful row with its prediction still intact."""
    if row.get("resume_key") != resume_key or row.get("status") != "ok":
        return False
    run_dir = Path(str(row.get("run_dir", "")))
    if not run_dir.is_dir():
        return False
    try:
        artifact = IIO.find_pred_artifact(run_dir)
    except FileNotFoundError:
        return False
    return (row.get("prediction_artifact") == str(artifact) and
            row.get("prediction_artifact_sha256") == sha256(artifact))


def _parse_run_dir(output: str, fallback: Path) -> Path:
    for line in output.splitlines():
        if "[OK] final:" in line:
            return Path(line.split("[OK] final:", 1)[1].strip()).parent
    return fallback


def shared_rejection_log_state() -> Dict[str, Any]:
    """Identity of the shared log, sufficient to reject stale snapshots."""
    if not SHARED_REJECTION_LOG.is_file():
        return {"exists": False, "mtime_ns": None, "size": None, "sha256": None}
    stat = SHARED_REJECTION_LOG.stat()
    return {"exists": True, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
            "sha256": sha256(SHARED_REJECTION_LOG)}


def snapshot_rejection_log(run_dir: Path, before: Dict[str, Any],
                           after: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Copy only a shared log that changed during this pipeline attempt."""
    after = after or shared_rejection_log_state()
    if not after["exists"] or before == after:
        return None
    destination = run_dir / "3.reconnect" / "rejection_log.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHARED_REJECTION_LOG, destination)
    return destination


def summarize_rejection_log(path: Optional[Path]) -> Dict[str, Any]:
    """Count accepts and rejection reasons, retaining the q-fit gate explicitly."""
    if path is None or not Path(path).is_file():
        return {"available": False, "accepted": 0, "rejected": 0, "reasons": {}, "tip_fit_quality": 0}
    reasons: Counter[str] = Counter()
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            reasons[str(row.get("reason") or "missing_reason")] += 1
    accepted = int(reasons.pop("accepted", 0))
    rejected = int(sum(reasons.values()))
    return {"available": True, "accepted": accepted, "rejected": rejected,
            "reasons": dict(sorted(reasons.items())), "tip_fit_quality": int(reasons.get("tip_fit_quality", 0))}


def score_success(scene: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Score common fragments and the existing native reconnect diagnostics."""
    pred = IIO.load_pred_instances(run_dir)
    common = CM.score_method(scene["mask_path"], scene["gt_path"], pred)
    gt, meta = IIO.load_gt(scene["gt_path"], shape=pred.shape)
    fragments = IIO.load_fragments(run_dir, min_area=15)
    native = M.evaluate(pred, gt, fragments, gt_kind=meta["kind"]).to_dict()
    return _jsonable({"common_metric": common, "native_diagnostics": native})


def _run_one(scene: Dict[str, Any], variant: Dict[str, Any], config_path: Path,
             output_root: Path, hashes: Dict[str, Optional[str]], python: str) -> Dict[str, Any]:
    job_root = output_root / "runs" / str(variant["name"]) / scene["scene_id"]
    expected_run = job_root / "run"
    job_root.mkdir(parents=True, exist_ok=True)
    command = [python, str(PIPELINE), "--mask", str(scene["mask_path"]),
               "--background", str(scene["background_path"]), "--output-root", str(job_root),
               "--base", "run", "--reconnect-config", str(config_path)]
    rejection_before = shared_rejection_log_state()
    try:
        proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
        stdout = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
        returncode = int(proc.returncode)
    except OSError as exc:
        stdout = f"pipeline launch failed: {type(exc).__name__}: {exc}\n"
        returncode = -1
    stdout_path = job_root / "pipeline_stdout.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    run_dir = _parse_run_dir(stdout, expected_run)
    rejection_after = shared_rejection_log_state()
    rejection_log = snapshot_rejection_log(run_dir, rejection_before, rejection_after)
    row: Dict[str, Any] = {
        "case_type": scene["case_type"], "scene_id": scene["scene_id"], "density": scene.get("density"),
        "variant": variant["name"], "parameters": dict(variant), "config_path": str(config_path),
        "config_sha256": sha256(config_path), "run_dir": str(run_dir), "stdout_path": str(stdout_path),
        "returncode": returncode, "rejection_log": str(rejection_log) if rejection_log else None,
        "rejections": summarize_rejection_log(rejection_log),
        "shared_rejection_log_before": rejection_before, "shared_rejection_log_after": rejection_after,
        "resume_key": make_resume_key(scene, variant, config_path, hashes),
    }
    if returncode:
        row.update(status="failed", error=f"pipeline return code {returncode}")
        return row
    try:
        row.update(status="ok", scores=score_success(scene, run_dir))
        artifact = IIO.find_pred_artifact(run_dir)
        row["prediction_artifact"] = str(artifact)
        row["prediction_artifact_sha256"] = sha256(artifact)
        row["resume_key"] = make_resume_key(scene, variant, config_path, hashes, artifact)
    except Exception as exc:
        row.update(status="failed", error=f"scoring {type(exc).__name__}: {exc}")
    return _jsonable(row)


def rank_variants(rows: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Rank only successful synthetic development scenes, never real cases."""
    grouped: Dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in rows:
        name = str(row.get("variant"))
        if row.get("case_type") != "synthetic":
            continue
        totals[name] += 1
        if row.get("status") != "ok":
            failures[name] += 1
            continue
        f1 = row.get("scores", {}).get("common_metric", {}).get("f1")
        if f1 is not None:
            grouped[name].append(float(f1))
    names = sorted(set(grouped) | set(failures) | set(totals))
    ranking = []
    for name in names:
        summary = _jsonable(stats_util.summarize(grouped[name]).as_dict())
        ranking.append({"variant": name, "n_total": int(totals[name]),
                        "n_pair_evidence": len(grouped[name]), "n_success": len(grouped[name]),
                        "failure_count": int(failures[name]), "n_failed": int(failures[name]),
                        "common_f1": summary, "mean_common_f1": summary["mean"]})
    # Complete variants are always ranked ahead of variants with any failed
    # scene, even when an incomplete variant's mean looks better.
    ranking.sort(key=lambda r: (r["failure_count"] > 0, r["mean_common_f1"] is None,
                                -(r["mean_common_f1"] or 0.0), r["variant"]))
    return ranking


def density_summaries(rows: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, int], list[float]] = defaultdict(list)
    totals: Counter[tuple[str, int]] = Counter()
    failures: Counter[tuple[str, int]] = Counter()
    for row in rows:
        if row.get("case_type") != "synthetic" or row.get("density") is None:
            continue
        key = (str(row["variant"]), int(row["density"]))
        totals[key] += 1
        if row.get("status") != "ok":
            failures[key] += 1
            continue
        f1 = row.get("scores", {}).get("common_metric", {}).get("f1")
        if f1 is not None:
            groups[key].append(float(f1))
    summaries = []
    for (variant, density) in sorted(set(groups) | set(totals)):
        summary = _jsonable(stats_util.summarize(groups[(variant, density)]).as_dict())
        summaries.append({"variant": variant, "density": density, "n_total": int(totals[(variant, density)]),
                          "n_pair_evidence": len(groups[(variant, density)]), "n_success": len(groups[(variant, density)]),
                          "failure_count": int(failures[(variant, density)]), "common_f1": summary,
                          "mean_common_f1": summary["mean"], "median_common_f1": float(np.median(groups[(variant, density)])) if groups[(variant, density)] else None})
    return summaries


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = ["case_type", "scene_id", "density", "variant", "status", "returncode", "config_sha256",
              "run_dir", "stdout_path", "rejection_log", "error", "common_f1", "common_precision",
              "common_recall", "rejection_accepted", "rejection_rejected", "tip_fit_quality_rejections"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metric = row.get("scores", {}).get("common_metric", {})
            rejection = row.get("rejections", {})
            writer.writerow({**row, "common_f1": metric.get("f1"), "common_precision": metric.get("precision"),
                             "common_recall": metric.get("recall"), "rejection_accepted": rejection.get("accepted"),
                             "rejection_rejected": rejection.get("rejected"),
                             "tip_fit_quality_rejections": rejection.get("tip_fit_quality")})


def _real_scene(mask: Path, background: Path, gt: Path, scene_id: str) -> Dict[str, Any]:
    return {"case_type": "real", "scene_id": scene_id, "density": None, "scene_dir": mask.parent,
            "mask_path": mask.resolve(), "background_path": background.resolve(), "gt_path": gt.resolve()}


def run_study(*, output_root: Path | str, source_config: Path | str = PRODUCTION_CONFIG,
              synthetic_root: Path | str = SYNTHETIC_ROOT, resume: bool = False,
              variants: Optional[Sequence[Dict[str, Any]]] = None, python: str = sys.executable,
              real_case: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the fixed curvature study sequentially and write JSON plus CSV."""
    synthetic_root = Path(synthetic_root)
    reject_locked(synthetic_root)
    source_config = Path(source_config).resolve()
    output_root = Path(output_root).resolve()
    selected_variants = list(variants or default_variants())
    scenes = discover_synthetic_scenes(synthetic_root)
    if real_case is not None:
        scenes.append(_real_scene(Path(real_case["mask_path"]), Path(real_case["background_path"]),
                                  Path(real_case["gt_path"]), str(real_case.get("scene_id", "real_development"))))
    configs = write_variant_configs(source_config, output_root / "configs", selected_variants)
    hashes = code_hashes()
    report_path = output_root / "curvature_study.json"
    previous: Dict[tuple[str, str], Dict[str, Any]] = {}
    if resume and report_path.is_file():
        try:
            previous = {(str(r["variant"]), str(r["scene_id"])): r
                        for r in json.loads(report_path.read_text(encoding="utf-8")).get("runs", [])}
        except (OSError, ValueError, TypeError, KeyError):
            previous = {}
    rows: list[Dict[str, Any]] = []
    for variant in selected_variants:
        config_path = configs[str(variant["name"])]
        for scene in scenes:  # Sequential by design: reconnect has one shared rejection log.
            old = previous.get((str(variant["name"]), str(scene["scene_id"])))
            old_artifact = Path(str(old["prediction_artifact"])) if old and old.get("prediction_artifact") else None
            key = make_resume_key(scene, variant, config_path, hashes, old_artifact)
            if old is not None and resume_matches(old, key):
                rows.append(old)
            else:
                rows.append(_run_one(scene, variant, config_path, output_root, hashes, python))
    overall = rank_variants(rows)
    document = _jsonable({
        "schema_version": 1,
        "study": "development_only_curvature_qfit",
        "synthetic_root": str(synthetic_root.resolve()),
        "source_config": str(source_config), "source_config_sha256": sha256(source_config),
        "variants": [{**variant, "config_path": str(configs[str(variant["name"])]),
                      "config_sha256": sha256(configs[str(variant["name"])])} for variant in selected_variants],
        "ranking_rule": "descending mean common_metric.f1 over successful synthetic development scenes only; real development cases are excluded",
        "ranking": overall, "overall_variant_summaries": overall,
        "per_density": density_summaries(rows), "runs": rows,
        "provenance": {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                       "platform": platform.platform(), "numpy": np.__version__, "code_hashes": hashes},
    })
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(rows, output_root / "curvature_study.csv")
    return document


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Development-only curvature/q-fit reconnect study")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--source-config", type=Path, default=PRODUCTION_CONFIG)
    ap.add_argument("--synthetic-root", type=Path, default=SYNTHETIC_ROOT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--real-mask", type=Path)
    ap.add_argument("--real-background", type=Path)
    ap.add_argument("--real-gt", type=Path)
    ap.add_argument("--real-id", default="real_development")
    args = ap.parse_args(argv)
    real_parts = (args.real_mask, args.real_background, args.real_gt)
    if any(real_parts) and not all(real_parts):
        ap.error("--real-mask, --real-background, and --real-gt must be supplied together")
    try:
        doc = run_study(output_root=args.output_root, source_config=args.source_config,
                        synthetic_root=args.synthetic_root, resume=args.resume,
                        real_case=({"mask_path": args.real_mask, "background_path": args.real_background,
                                    "gt_path": args.real_gt, "scene_id": args.real_id} if all(real_parts) else None))
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    failures = sum(row.get("status") == "failed" for row in doc["runs"])
    print(f"wrote {Path(args.output_root) / 'curvature_study.json'}: {len(doc['runs'])} rows, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
