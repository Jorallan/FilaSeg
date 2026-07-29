"""Re-run the real development-case ablations with one frozen reconnect config.

The real crop is development data. These runs diagnose components; they are
never used to select the locked configuration or to estimate generalisation.
Both the method-independent common-fragment score and the FilaSeg-native
diagnostic are retained for every successful row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]  # repository root
HERE = Path(__file__).resolve().parents[1]  # eval/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)

import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402
import metrics as M  # noqa: E402
from curvature_study import _jsonable, _parse_run_dir  # noqa: E402

PIPELINE = ROOT / "Tools" / "run_full_sem_pipeline.py"

VARIANTS = (
    {"name": "locked", "arguments": []},
    {"name": "content_adaptation_disabled", "arguments": ["--stringart-no-auto-scale"]},
    # One offset and a vote threshold of one is a genuine single-grid run.
    # Merely setting vote-min=1 would still combine all four default grids.
    {"name": "single_tile_grid", "arguments": ["--tile-grid-offsets", "1", "--tile-grid-vote-min", "1"]},
    {"name": "fixed_render_width", "arguments": ["--no-smart-width"]},
    {"name": "orientation_gate_disabled",
     "arguments": ["--reconnect-max-orientation-mismatch-deg", "180"]},
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score(mask: Path, gt_path: Path, prediction_path: Path, run_dir: Path) -> dict[str, Any]:
    pred = IIO.load_pred_instances(prediction_path)
    common = CM.score_method(mask, gt_path, pred)
    gt, meta = IIO.load_gt(gt_path, shape=pred.shape)
    fragments = IIO.load_fragments(run_dir, min_area=15)
    native = M.evaluate(pred, gt, fragments, gt_kind=meta["kind"]).to_dict()
    return _jsonable({"common_metric": common, "native_diagnostics": native})


def run(*, mask: Path, background: Path, gt: Path, reconnect_config: Path,
        output_root: Path, python: str = sys.executable) -> dict[str, Any]:
    mask, background, gt, reconnect_config = (
        mask.resolve(), background.resolve(), gt.resolve(), reconnect_config.resolve()
    )
    for path in (mask, background, gt, reconnect_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    locked_run: Optional[Path] = None
    for variant in VARIANTS:
        job_root = output_root.resolve() / "runs" / str(variant["name"])
        job_root.mkdir(parents=True, exist_ok=True)
        command = [
            python, str(PIPELINE), "--mask", str(mask), "--background", str(background),
            "--output-root", str(job_root), "--base", "run",
            "--reconnect-config", str(reconnect_config), *variant["arguments"],
        ]
        proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
        stdout = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
        stdout_path = job_root / "pipeline_stdout.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        run_dir = _parse_run_dir(stdout, job_root / "run")
        row: dict[str, Any] = {
            "case_type": "real_development", "variant": variant["name"],
            "arguments": variant["arguments"], "command": command,
            "returncode": int(proc.returncode), "run_dir": str(run_dir),
            "stdout_path": str(stdout_path), "config_sha256": sha256(reconnect_config),
        }
        if proc.returncode:
            row.update(status="failed", error=f"pipeline return code {proc.returncode}")
        else:
            try:
                row.update(status="ok", scores=score(mask, gt, run_dir, run_dir))
                row["prediction_artifact"] = str(IIO.find_pred_artifact(run_dir))
                if variant["name"] == "locked":
                    locked_run = run_dir
            except Exception as exc:  # retained in the report, never omitted
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(_jsonable(row))

    # "Before Stage 4" is derived from the identical locked run. It changes
    # only the evaluated prediction, not the Stage-1 fragment set.
    if locked_run is not None:
        stage3 = locked_run / "3.reconnect"
        row = {
            "case_type": "real_development", "variant": "before_stage4",
            "arguments": [], "run_dir": str(locked_run), "prediction_path": str(stage3),
            "config_sha256": sha256(reconnect_config),
        }
        try:
            row.update(status="ok", scores=score(mask, gt, stage3, locked_run))
            row["prediction_artifact"] = str(IIO.find_pred_artifact(stage3))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(_jsonable(row))

    document = _jsonable({
        "schema_version": 1,
        "study": "real_development_case_component_ablation",
        "interpretation": "development diagnostic only; not independent validation and not used for tuning",
        "inputs": {"mask": str(mask), "background": str(background), "manual_reference": str(gt)},
        "reconnect_config": str(reconnect_config), "reconnect_config_sha256": sha256(reconnect_config),
        "variants": list(VARIANTS), "runs": rows,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(),
            "code_hashes": {
                "real_ablation_runner.py": sha256(Path(__file__)),
                "run_full_sem_pipeline.py": sha256(PIPELINE),
                "common_metric.py": sha256(HERE / "common_metric.py"),
                "metrics.py": sha256(HERE / "metrics.py"),
                "instance_io.py": sha256(HERE / "instance_io.py"),
            },
        },
    })
    output_root.mkdir(parents=True, exist_ok=True)
    report = output_root / "real_ablation_results.json"
    report.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return document


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--background", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--reconnect-config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        doc = run(mask=args.mask, background=args.background, gt=args.gt,
                  reconnect_config=args.reconnect_config, output_root=args.output_root)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    failures = sum(row.get("status") == "failed" for row in doc["runs"])
    print(f"wrote {args.output_root / 'real_ablation_results.json'}: "
          f"{len(doc['runs'])} rows, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
