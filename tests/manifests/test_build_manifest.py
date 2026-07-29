"""Isolated fixtures for development + locked manifest construction."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)
import build_manifest as BM  # noqa: E402


def _row(geometry: str, method: str, *, seed: int = 101, density: int = 20,
         status: str = "ok", error: str = "") -> dict:
    return {
        "geometry_id": geometry, "density": density, "seed": seed,
        "mask_variant": "3", "method": method, "status": status,
        "output_path": f"/outputs/{geometry}/{method}.tif", "error": error,
        "config_hashes": {"reconnect_config_source": "abc123"},
    }


def _report(rows: list[dict], root: str, generator: str) -> dict:
    return {"samples_root": root, "generator_command": generator,
            "configuration": {"mask_name": "mask.png"}, "per_scene": rows}


class BuildManifestTests(unittest.TestCase):
    def test_combines_roles_retains_failure_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dev = tmp / "development.json"; locked = tmp / "locked.json"; out = tmp / "manifest.csv"
            dev.write_text(json.dumps(_report([_row("cov20/dev_a", "baseline_cc", status="failed", error="boom")],
                                              "/data/dev", "python make_dev --seed0 7")), encoding="utf-8")
            locked.write_text(json.dumps(_report([_row("cov20/lock_a", "baseline_cc", seed=801)],
                                                 "/data/locked", "python make_locked --seed0 801")), encoding="utf-8")
            rows, headers = BM.build_manifest(locked, dev)
            BM.write_manifest(rows, headers, out)
            self.assertEqual(len(rows), 2)
            failed = next(r for r in rows if r["dataset_role"] == "development")
            self.assertEqual((failed["status"], failed["failure_notes"]), ("failed", "boom"))
            self.assertEqual(failed["input_mask"], "/data/dev/cov20/dev_a/mask.png")
            self.assertEqual(failed["output_hash_status"], "not_applicable_failed")
            self.assertEqual(failed["output_sha256"], "")
            self.assertEqual(failed["input_mask_hash_status"], "missing")
            self.assertIn("python make_dev --seed0 7", "\n".join(headers))
            self.assertIn('"cov20/lock_a":{"density":20,"seed":801}', "\n".join(headers))
            with out.open(newline="", encoding="utf-8") as fh:
                content = fh.read().splitlines()
            header_index = next(i for i, line in enumerate(content) if line.startswith("seed,"))
            parsed = list(csv.DictReader(content[header_index:]))
            self.assertEqual(set(parsed[0]), set(BM.FIELDS))
            self.assertEqual(len(parsed), 2)

    def test_hashes_artifacts_and_records_source_report_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = tmp / "scene"
            scene.mkdir()
            input_mask = scene / "mask.png"; input_mask.write_bytes(b"input-mask")
            ground_truth = scene / "gt_labels.tif"; ground_truth.write_bytes(b"ground-truth")
            output = scene / "result.tif"; output.write_bytes(b"method-output")
            row = _row("scene", "baseline_cc")
            row.update({"input_mask": str(input_mask), "ground_truth": str(ground_truth),
                        "output_path": str(output)})
            report_path = tmp / "locked.json"
            report_path.write_text(json.dumps(_report([row], str(tmp), "gen")), encoding="utf-8")

            rows, _ = BM.build_manifest(report_path)
            result = rows[0]
            self.assertEqual(result["dataset_role"], "locked")
            self.assertEqual(result["source_report"], str(report_path))
            self.assertEqual(result["source_report_sha256"],
                             hashlib.sha256(report_path.read_bytes()).hexdigest())
            for field, artifact in (("input_mask", input_mask), ("ground_truth", ground_truth),
                                    ("output", output)):
                self.assertEqual(result[f"{field}_sha256"],
                                 hashlib.sha256(artifact.read_bytes()).hexdigest())
                self.assertEqual(result[f"{field}_hash_status"], "present")
            self.assertEqual(result["ground_truth_selected"], str(ground_truth))
            self.assertEqual(result["ground_truth_selected_sha256"],
                             hashlib.sha256(ground_truth.read_bytes()).hexdigest())
            self.assertEqual(result["ground_truth_selected_hash_status"], "present")

    def test_records_overlap_aware_ground_truth_actually_selected_by_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = tmp / "scene"
            scene.mkdir()
            flat = scene / "gt_labels.tif"; flat.write_bytes(b"flat")
            overlap = scene / "gt_multilabel.npz"; overlap.write_bytes(b"overlap")
            row = _row("scene", "baseline_cc")
            row.update({"ground_truth": str(flat)})
            report_path = tmp / "locked.json"
            report_path.write_text(json.dumps(_report([row], str(tmp), "gen")),
                                   encoding="utf-8")

            result = BM.build_manifest(report_path)[0][0]
            self.assertEqual(result["ground_truth"], str(flat))
            self.assertEqual(result["ground_truth_selected"], str(overlap))
            self.assertEqual(result["ground_truth_selected_sha256"],
                             hashlib.sha256(overlap.read_bytes()).hexdigest())
            self.assertEqual(result["ground_truth_selected_hash_status"], "present")

    def test_successful_missing_output_has_explicit_hash_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = tmp / "locked.json"
            path.write_text(json.dumps(_report([_row("cov20/a", "baseline_cc")], str(tmp), "gen")),
                            encoding="utf-8")
            rows, _ = BM.build_manifest(path)
            self.assertEqual(rows[0]["output_hash_status"], "missing")
            self.assertEqual(rows[0]["output_sha256"], "")

    def test_field_schema_is_stable(self):
        self.assertEqual(BM.FIELDS, [
            "seed", "density", "geometry_id", "dataset_role", "mask_width",
            "degradation", "input_mask", "ground_truth", "method", "output_path",
            "config_hash", "status", "failure_notes", "source_report",
            "source_report_sha256", "input_mask_sha256", "input_mask_hash_status",
            "ground_truth_sha256", "ground_truth_hash_status", "output_sha256",
            "output_hash_status", "ground_truth_selected",
            "ground_truth_selected_sha256", "ground_truth_selected_hash_status",
        ])

    def test_duplicate_scene_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locked.json"
            path.write_text(json.dumps(_report([_row("cov20/a", "baseline_cc"),
                                                _row("cov20/a", "baseline_cc")], "/data", "gen")),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate scene/method"):
                BM.build_manifest(path)

    def test_inconsistent_seed_for_one_scene_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locked.json"
            path.write_text(json.dumps(_report([_row("cov20/a", "baseline_cc", seed=1),
                                                _row("cov20/a", "baseline_skeleton", seed=2)], "/data", "gen")),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent seed/density"):
                BM.build_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
