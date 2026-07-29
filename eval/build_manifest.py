"""Build a complete, non-selective evaluation manifest from runner reports.

The manifest is a direct projection of ``experiment_runner`` JSON reports. It
does not discover scenes from the filesystem and never filters successful rows
or ranks methods: failed rows are evidence and remain in the archive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Retained for legacy reports that predate an explicit generator command in
# their JSON provenance. Do not infer or replace a development command.
#
# The string below is a HISTORICAL RECORD of the command as actually run, and is
# deliberately left byte-for-byte unchanged. The generator itself was promoted
# from sandbox/ to eval/synth_thick.py on 2026-07-29 (sandbox/ is no longer
# tracked); to re-run the locked generation today, substitute
# `eval/synth_thick.py` for `sandbox/synth_thick.py`. The arguments are
# unchanged and the module is byte-identical to the promoted copy.
LOCKED_GENERATOR_COMMAND = (
    "python sandbox/synth_thick.py --out input/synthetic_locked_v1 "
    "--coverages 0.20,0.30,0.40,0.50,0.60 --n 25 --size 512 --seed0 20260801"
)
SEED_FORMULA = "seed = seed0 + round(coverage*100)*100 + sample_index"

FIELDS = [
    "seed", "density", "geometry_id", "dataset_role", "mask_width",
    "degradation", "input_mask", "ground_truth", "method", "output_path",
    "config_hash", "status", "failure_notes",
]


def _generator_command(report: Mapping[str, Any], role: str) -> str:
    """Return only explicitly recorded generator provenance when available."""
    provenance = report.get("provenance", {})
    configuration = report.get("configuration", {})
    for container in (report, provenance, configuration):
        if isinstance(container, Mapping) and container.get("generator_command"):
            return str(container["generator_command"])
    if role == "locked":
        return LOCKED_GENERATOR_COMMAND
    return "not recorded in experiment report"


def _config_hash(row: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    if row.get("config_hash") is not None:
        return str(row["config_hash"])
    for hashes in (row.get("config_hashes"), report.get("config_hashes")):
        if isinstance(hashes, Mapping):
            value = hashes.get("reconnect_config_source")
            if value is not None:
                return str(value)
    return ""


def _scene_paths(row: Mapping[str, Any], report: Mapping[str, Any]) -> Tuple[str, str]:
    """Use row paths when present, otherwise deterministically reconstruct them."""
    input_mask = row.get("input_mask", row.get("mask_path"))
    ground_truth = row.get("ground_truth", row.get("gt_path"))
    if input_mask is not None and ground_truth is not None:
        return str(input_mask), str(ground_truth)

    root = report.get("samples_root")
    if not root:
        return str(input_mask or ""), str(ground_truth or "")
    config = report.get("configuration", {})
    mask_name = config.get("mask_name", "mask.png") if isinstance(config, Mapping) else "mask.png"
    scene = Path(str(root)) / str(row["geometry_id"])
    return (str(input_mask) if input_mask is not None else (scene / str(mask_name)).as_posix(),
            str(ground_truth) if ground_truth is not None else (scene / "gt_labels.tif").as_posix())


def _manifest_rows(report: Mapping[str, Any], role: str) -> List[Dict[str, Any]]:
    if role not in {"development", "locked"}:
        raise ValueError(f"unknown dataset role {role!r}")
    source_rows = report.get("per_scene")
    if not isinstance(source_rows, list):
        raise ValueError(f"{role} report has no per_scene list")

    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(source_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{role} per_scene[{index}] is not an object")
        required = ("seed", "density", "geometry_id", "mask_variant", "method", "status", "output_path")
        missing = [key for key in required if raw.get(key) is None]
        if missing:
            raise ValueError(f"{role} per_scene[{index}] missing required field(s): {', '.join(missing)}")
        try:
            seed, density = int(raw["seed"]), int(raw["density"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} per_scene[{index}] has invalid seed/density") from exc
        mask_width = str(raw.get("mask_width", raw["mask_variant"]))
        input_mask, ground_truth = _scene_paths(raw, report)
        rows.append({
            "seed": seed,
            "density": density,
            "geometry_id": str(raw["geometry_id"]),
            "dataset_role": role,
            "mask_width": mask_width,
            "degradation": str(raw.get("degradation", "none" if mask_width == "clean" else "degraded")),
            "input_mask": input_mask,
            "ground_truth": ground_truth,
            "method": str(raw["method"]),
            "output_path": str(raw["output_path"]),
            "config_hash": _config_hash(raw, report),
            "status": str(raw["status"]),
            "failure_notes": str(raw.get("failure_notes", raw.get("error", "")) or ""),
        })
    return rows


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate scene/method runs and inconsistent scene provenance."""
    seen: set[Tuple[str, str, str, str]] = set()
    scene_provenance: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for row in rows:
        key = (str(row["dataset_role"]), str(row["geometry_id"]),
               str(row["mask_width"]), str(row["method"]))
        if key in seen:
            raise ValueError(f"duplicate scene/method manifest key: {key}")
        seen.add(key)
        scene_key = (str(row["dataset_role"]), str(row["geometry_id"]))
        provenance = (int(row["seed"]), int(row["density"]))
        prior = scene_provenance.setdefault(scene_key, provenance)
        if prior != provenance:
            raise ValueError(
                f"inconsistent seed/density for scene {scene_key}: {prior} vs {provenance}"
            )


def build_manifest(locked_results: Optional[Path | str] = None,
                   development_results: Optional[Path | str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load reports without selecting by performance; return rows and headers."""
    sources = [("development", development_results), ("locked", locked_results)]
    reports: List[Tuple[str, Path, Mapping[str, Any]]] = []
    rows: List[Dict[str, Any]] = []
    for role, source in sources:
        if not source:
            continue
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{role} results report does not exist: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError(f"{role} results report must contain a JSON object")
        reports.append((role, path, report))
        rows.extend(_manifest_rows(report, role))
    if not reports:
        raise ValueError("provide --locked-results/--results and/or --development-results")
    _validate_rows(rows)

    headers = [
        "# FilaSeg development + locked evaluation manifest",
        "# rows are copied from experiment_runner reports without performance filtering; failed rows are retained",
        f"# seed_formula: {SEED_FORMULA}",
    ]
    for role, path, report in reports:
        scene_seed = {
            geometry: {"seed": seed, "density": density}
            for (dataset_role, geometry), (seed, density) in sorted(
                {(r["dataset_role"], r["geometry_id"]): (r["seed"], r["density"])
                 for r in rows if r["dataset_role"] == role}.items()
            ) if dataset_role == role
        }
        headers.extend([
            f"# {role}_results: {path}",
            f"# {role}_generator: {_generator_command(report, role)}",
            f"# {role}_seed_provenance: {json.dumps(scene_seed, sort_keys=True, separators=(',', ':'))}",
        ])
    return sorted(rows, key=lambda r: (r["dataset_role"], r["density"], r["geometry_id"],
                                       r["mask_width"], r["method"])), headers


def write_manifest(rows: Iterable[Mapping[str, Any]], headers: Iterable[str], out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        for header in headers:
            fh.write(f"{header}\n")
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a development + locked evaluation manifest")
    ap.add_argument("--locked-results", default="", help="locked experiment_runner JSON report")
    ap.add_argument("--results", default="", help="deprecated alias for --locked-results")
    ap.add_argument("--development-results", default="", help="development experiment_runner JSON report")
    ap.add_argument("--out", default=str(ROOT / "paper" / "evaluation_manifest.csv"))
    args = ap.parse_args()
    if args.locked_results and args.results and Path(args.locked_results) != Path(args.results):
        ap.error("--locked-results and --results disagree; supply only one")
    locked = args.locked_results or args.results
    try:
        rows, headers = build_manifest(locked, args.development_results)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    write_manifest(rows, headers, args.out)
    print(f"wrote {args.out} ({len(rows)} rows; no performance-based selection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
