"""Toy tests for the reproducible experiment runner (no pipeline invocation)."""
from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import experiment_runner as ER  # noqa: E402


def _scene(root: Path, density: int, name: str, seed: int, *, broken: bool = False) -> Path:
    d = root / f"cov{density}" / name
    d.mkdir(parents=True)
    mask = np.zeros((24, 24), np.uint8)
    mask[10:14, 3:21] = 255
    gt = np.zeros_like(mask, dtype=np.int32)
    gt[10:14, 3:21] = 1
    tifffile.imwrite(str(d / "mask.png"), mask)
    tifffile.imwrite(str(d / "gt_labels.tif"), gt)
    (d / "gt_meta.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    if broken:
        (d / "gt_labels.tif").unlink()
    return d


class ExperimentRunnerTests(unittest.TestCase):
    def test_discovery_has_deterministic_exact_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _scene(root, 40, "z", 9); _scene(root, 20, "a", 3)
            got = ER.discover_scenes(root)
            self.assertEqual([x["geometry_id"] for x in got], ["cov20/a", "cov40/z"])
            self.assertEqual([(x["density"], x["seed"], x["mask_variant"]) for x in got], [(20, 3, "mask"), (40, 9, "mask")])

    def test_failure_is_retained_and_csv_written(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"; scene = _scene(root, 20, "a", 1)
            # A missing GT is still a discovered scene and must become a row.
            (scene / "gt_labels.tif").unlink()
            out = Path(td) / "report.json"
            doc = ER.run_experiment(root, out, methods=["baseline_cc"])
            self.assertEqual(len(doc["per_scene"]), 1)
            self.assertEqual(doc["per_scene"][0]["status"], "failed")
            self.assertIn("error", doc["per_scene"][0])
            self.assertTrue(out.with_suffix(".csv").is_file())

    def test_locked_sweep_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            locked = Path(td) / "input" / "synthetic_locked_v1"
            _scene(locked, 20, "a", 1)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                ER.run_experiment(locked, Path(td) / "x.json", sweep_gaps=[1, 2])

    def test_locked_fixed_parameters_require_explicit_attestation(self):
        with tempfile.TemporaryDirectory() as td:
            locked = Path(td) / "input" / "synthetic_locked_v1"
            _scene(locked, 20, "a", 1)
            with self.assertRaisesRegex(ValueError, "configuration-locked"):
                ER.run_experiment(locked, Path(td) / "x.json", methods=["baseline_cc"],
                                  cc_min_area=1, max_gap_px=28, max_turn_deg=20)
            doc = ER.run_experiment(locked, Path(td) / "ok.json", methods=["baseline_cc"],
                                    cc_min_area=1, max_gap_px=28, max_turn_deg=20,
                                    configuration_locked=True)
            self.assertTrue(doc["configuration"]["configuration_locked"])

    def test_resume_requires_complete_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"; _scene(root, 20, "a", 1)
            out = Path(td) / "report.json"
            first = ER.run_experiment(root, out, methods=["baseline_cc"])
            self.assertTrue(ER.row_complete(first["per_scene"][0]))
            second = ER.run_experiment(root, out, methods=["baseline_cc"], resume=True)
            self.assertEqual(second["per_scene"][0]["output_path"], first["per_scene"][0]["output_path"])
            Path(first["per_scene"][0]["output_path"]).unlink()
            third = ER.run_experiment(root, out, methods=["baseline_cc"], resume=True)
            self.assertTrue(Path(third["per_scene"][0]["output_path"]).is_file())

    def test_resume_is_invalidated_when_selected_config_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"; _scene(root, 20, "a", 1)
            out = Path(td) / "report.json"; config = Path(td) / "reconnect.yaml"
            config.write_text("threshold: 1\n", encoding="utf-8")
            first = ER.run_experiment(root, out, methods=["baseline_cc"], reconnect_config=config)
            old_hash = first["per_scene"][0]["config_hashes"]["reconnect_config_source"]
            saved = json.loads(out.read_text(encoding="utf-8"))
            saved["per_scene"][0]["scores"]["resume_marker"] = "must not survive"
            out.write_text(json.dumps(saved), encoding="utf-8")
            config.write_text("threshold: 2\n", encoding="utf-8")
            second = ER.run_experiment(root, out, methods=["baseline_cc"], reconnect_config=config, resume=True)
            self.assertNotEqual(second["per_scene"][0]["config_hashes"]["reconnect_config_source"], old_hash)
            self.assertNotIn("resume_marker", second["per_scene"][0]["scores"])

    def test_filaseg_reuse_preserves_overlap_aware_npz(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"; scene = _scene(root, 20, "a", 1)
            pred_dir = Path(td) / "filaseg" / "cov20" / "a" / "mask" / "final"
            pred_dir.mkdir(parents=True)
            flat = np.flatnonzero(tifffile.imread(str(scene / "gt_labels.tif"))).astype(np.int64)
            # Two fully overlapping layers: loading this as a TIFF would lose
            # one ID, whereas load_pred_instances retains both masks.
            np.savez(pred_dir / "toy_reconnect_multilabel.npz", shape=np.array([24, 24]),
                     ids=np.array([1, 2]), indptr=np.array([0, len(flat), 2 * len(flat)]),
                     indices=np.concatenate([flat, flat]))
            doc = ER.run_experiment(root, Path(td) / "report.json", methods=[],
                                    filaseg="reuse", filaseg_root=Path(td) / "filaseg")
            row = doc["per_scene"][0]
            self.assertEqual(row["method"], "filaseg")
            self.assertEqual(row["status"], "ok")
            self.assertTrue(row["output_path"].endswith(".npz"))
            self.assertEqual(row["scores"]["shared_fragment_recovery"]["n_gt_instances"], 1)
            self.assertEqual(row["scores"]["representation_dependent_pixel_recovery"]["n_gt_instances"], 1)
            self.assertNotIn("pixel_instance", row["scores"])
            with (Path(td) / "report.csv").open(newline="", encoding="utf-8") as fh:
                headers = csv.DictReader(fh).fieldnames
            self.assertIn("fragment_recovery_rate", headers)
            self.assertIn("pixel_recovery_rate", headers)

    def test_pairing_is_by_scene_id_and_grid_ranks(self):
        rows = [
            {"geometry_id": "cov20/a", "density": 20, "mask_variant": "mask", "method": "a", "status": "ok", "scores": {"common_metric": {"f1": .2}}},
            {"geometry_id": "cov20/b", "density": 20, "mask_variant": "mask", "method": "a", "status": "ok", "scores": {"common_metric": {"f1": .4}}},
            {"geometry_id": "cov20/b", "density": 20, "mask_variant": "mask", "method": "b", "status": "ok", "scores": {"common_metric": {"f1": .1}}},
        ]
        paired = ER._summaries(rows)["paired"][0]
        self.assertEqual(paired["n_paired_scenes"], 1)
        self.assertEqual(paired["n_missing_b"], 1)
        self.assertEqual(ER._summaries(rows)["by_density"]["20"]["paired"][0]["n_paired_scenes"], 1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dev"; _scene(root, 20, "a", 1)
            doc = ER.run_experiment(root, Path(td) / "grid.json", sweep_gaps=[1, 2], sweep_turns=[45, 90])
            self.assertEqual(len(doc["baseline_grid"]["ranked"]), 4)
            self.assertIn("mean common_metric.f1", doc["baseline_grid"]["ranking_rule"])


if __name__ == "__main__":
    unittest.main()
