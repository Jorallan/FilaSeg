"""Unit tests for the full-runner reconnect-config artifact handling.

These tests only exercise path resolution and config copying; they never start
pipeline stages.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Tools import run_full_sem_pipeline as runner


class TestReconnectConfigSelection(unittest.TestCase):
    def test_default_source_is_production_config(self):
        with patch.object(sys, "argv", ["run_full_sem_pipeline.py"]):
            args = runner.parse_args()

        self.assertEqual(args.reconnect_config, runner.DEFAULT_RECONNECT_CONFIG)
        self.assertEqual(
            runner.resolve_reconnect_config(args.reconnect_config),
            (runner.ROOT / "3.reconnect" / "reconnect_config.yaml").resolve(),
        )

    def test_custom_source_is_used_for_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "study_config.yaml"
            source.write_text("stage_clear:\n  search_size_px: 16\n", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()

            self.assertEqual(runner.resolve_reconnect_config(source), source.resolve())
            scaled = runner.prepare_reconnect_config(source, run_dir, scale_factor=2.0)
            self.assertEqual(scaled, run_dir / "reconnect_config_scaled.yaml")
            self.assertIn("search_size_px: 8", scaled.read_text(encoding="utf-8"))

    def test_preparing_config_never_mutates_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "study_config.yaml"
            source.write_text("overlap:\n  mode: kill\n", encoding="utf-8")
            before = source.read_bytes()
            before_hash = runner.file_sha256(source)
            run_dir = root / "run"
            run_dir.mkdir()

            active = runner.prepare_reconnect_config(source, run_dir, scale_factor=1.0)

            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(runner.file_sha256(source), before_hash)
            self.assertEqual(active, run_dir / "reconnect_config_active.yaml")

    def test_active_copy_has_matching_content_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "study_config.yaml"
            source.write_text("stage_clear:\n  search_size_px: 16\n", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()

            active = runner.prepare_reconnect_config(source, run_dir, scale_factor=1.0)

            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(active.read_bytes(), source.read_bytes())
            self.assertEqual(runner.file_sha256(active), expected_hash)

    def test_orientation_ablation_override_changes_all_stages_only_in_active_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "study_config.yaml"
            source.write_text(
                "\n".join([
                    "overlap: {mode: kill, kill_thr: 0.8, trim_dilate_px: 1}",
                    "stage_clear: {thresholds: {max_orientation_mismatch_deg: 40}}",
                    "stage_ambiguous: {thresholds: {max_orientation_mismatch_deg: 40}}",
                    "stage_relaxed: {thresholds: {max_orientation_mismatch_deg: 40}}",
                ]) + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"; run_dir.mkdir()
            active = runner.prepare_reconnect_config(source, run_dir, scale_factor=1.0)
            with patch.object(
                sys, "argv",
                ["run_full_sem_pipeline.py",
                 "--reconnect-max-orientation-mismatch-deg", "180"],
            ):
                args = runner.parse_args()
            runner.apply_reconnect_config_overrides(active, args, branch_count=18)
            selected = yaml.safe_load(active.read_text(encoding="utf-8"))
            for stage in ("stage_clear", "stage_ambiguous", "stage_relaxed"):
                self.assertEqual(
                    selected[stage]["thresholds"]["max_orientation_mismatch_deg"], 180.0
                )
            original = yaml.safe_load(source.read_text(encoding="utf-8"))
            self.assertEqual(
                original["stage_clear"]["thresholds"]["max_orientation_mismatch_deg"], 40
            )


if __name__ == "__main__":
    unittest.main()
