"""Unit coverage for development-only curvature-study orchestration."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import curvature_study as CS  # noqa: E402


def _source_config(path: Path) -> None:
    stages = {
        stage: {"thresholds": {"curvature_measure": "geometric", "max_tip_fit_quality": 0.0}}
        for stage in CS.STAGES
    }
    stages["unrelated"] = {"keep": True}
    path.write_text(yaml.safe_dump(stages, sort_keys=False), encoding="utf-8")


class CurvatureStudyTests(unittest.TestCase):
    def test_locked_synthetic_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            locked = Path(td) / "input" / "synthetic_locked_v1"
            locked.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                CS.discover_synthetic_scenes(locked)

    def test_variant_yamls_set_all_stages_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.yaml"
            _source_config(source)
            source_bytes = source.read_bytes()

            variants = CS.default_variants()
            configs = CS.write_variant_configs(source, root / "variants", variants)

            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(len(configs), 7)
            legacy = yaml.safe_load(configs["legacy_mixed"].read_text(encoding="utf-8"))
            qfit = yaml.safe_load(configs["geometric_qfit_0.020"].read_text(encoding="utf-8"))
            for stage in CS.STAGES:
                self.assertEqual(legacy[stage]["thresholds"]["curvature_measure"], "legacy_mixed")
                self.assertEqual(legacy[stage]["thresholds"]["max_tip_fit_quality"], 0.0)
                self.assertEqual(qfit[stage]["thresholds"]["curvature_measure"], "geometric")
                self.assertEqual(qfit[stage]["thresholds"]["max_tip_fit_quality"], 0.020)
            self.assertTrue(legacy["unrelated"]["keep"])

    def test_ranking_excludes_real_development_case(self):
        rows = [
            {"case_type": "synthetic", "variant": "legacy_mixed", "status": "ok",
             "scores": {"common_metric": {"f1": 0.50}}},
            {"case_type": "synthetic", "variant": "geometric_qfit_0.010", "status": "ok",
             "scores": {"common_metric": {"f1": 0.40}}},
            {"case_type": "synthetic", "variant": "geometric_qfit_0.010", "status": "failed"},
            {"case_type": "real", "variant": "geometric_qfit_0.010", "status": "ok",
             "scores": {"common_metric": {"f1": 1.00}}},
        ]
        ranking = CS.rank_variants(rows)

        self.assertEqual([entry["variant"] for entry in ranking], ["legacy_mixed", "geometric_qfit_0.010"])
        self.assertEqual(ranking[1]["mean_common_f1"], 0.40)
        self.assertEqual(ranking[1]["n_total"], 2)
        self.assertEqual(ranking[1]["n_pair_evidence"], 1)
        self.assertEqual(ranking[1]["failure_count"], 1)

    def test_resume_key_invalidates_when_config_hash_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("mask.png", "sem.png", "gt_labels.tif"):
                (root / name).write_bytes(name.encode("ascii"))
            config = root / "variant.yaml"
            config.write_text("x: 1\n", encoding="utf-8")
            run_dir = root / "run"
            artifact = run_dir / "final" / "run_reconnect_labels.tif"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"prediction")
            scene = {"scene_id": "cov20/synth_1", "case_type": "synthetic",
                     "mask_path": root / "mask.png", "background_path": root / "sem.png",
                     "gt_path": root / "gt_labels.tif"}
            variant = {"name": "geometric_disabled", "curvature_measure": "geometric", "max_tip_fit_quality": 0.0}
            hashes = {"curvature_study.py": "code-a"}
            first = CS.make_resume_key(scene, variant, config, hashes, artifact)
            row = {"status": "ok", "resume_key": first, "run_dir": str(run_dir),
                   "prediction_artifact": str(artifact), "prediction_artifact_sha256": CS.sha256(artifact)}

            config.write_text("x: 2\n", encoding="utf-8")
            second = CS.make_resume_key(scene, variant, config, hashes, artifact)

            self.assertNotEqual(first, second)
            self.assertTrue(CS.resume_matches(row, first))
            self.assertFalse(CS.resume_matches(row, second))
            artifact.unlink()
            self.assertFalse(CS.resume_matches(row, first))

    def test_failed_rows_are_retried_even_when_their_hash_matches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            artifact = run_dir / "final" / "result_reconnect_labels.tif"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"prediction")
            key = "same-key"
            failed = {"status": "failed", "resume_key": key, "run_dir": str(run_dir),
                      "prediction_artifact": str(artifact), "prediction_artifact_sha256": CS.sha256(artifact)}

            self.assertFalse(CS.resume_matches(failed, key))

    def test_stale_shared_rejection_log_is_not_snapshotted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared_rejection.csv"
            shared.write_text("reason\naccepted\n", encoding="utf-8")
            with patch.object(CS, "SHARED_REJECTION_LOG", shared):
                before = CS.shared_rejection_log_state()
                run_dir = root / "run"
                self.assertIsNone(CS.snapshot_rejection_log(run_dir, before))
                self.assertFalse((run_dir / "3.reconnect" / "rejection_log.csv").exists())

                shared.write_text("reason\ntip_fit_quality\n", encoding="utf-8")
                after = CS.shared_rejection_log_state()
                copied = CS.snapshot_rejection_log(run_dir, before, after)

            self.assertIsNotNone(copied)
            self.assertEqual(copied.read_text(encoding="utf-8"), "reason\ntip_fit_quality\n")


if __name__ == "__main__":
    unittest.main()
