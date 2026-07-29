"""Focused tests for the development-only 2x2 grouping study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.studies import hybrid_grouping_factorial as HGF


class HybridGroupingFactorialTests(unittest.TestCase):
    def test_locked_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked evaluation"):
            HGF._reject_locked_path(
                Path("input") / "synthetic_locked_v1" / "cov20"
            )
        with self.assertRaisesRegex(ValueError, "locked evaluation"):
            HGF._reject_locked_path(
                Path("output") / "locked_evaluation" / "results.json"
            )

    def test_shared_orientation_bins_ignore_axis_direction(self):
        a = np.zeros((31, 31), dtype=bool)
        b = np.zeros_like(a)
        a[15, 3:28] = True
        b[3:28, 15] = True
        self.assertEqual(HGF.fragment_orientation_layer(a), 0)
        self.assertEqual(HGF.fragment_orientation_layer(b), 6)

    def test_grouping_config_is_copied_and_disables_overlap_suppression(self):
        source = {
            "overlap": {"kill_thr": 0.3, "trim_dilate_px": 10},
            "runtime": {"verbose": True},
            "debug": {"rejection_log_path": "somewhere.csv"},
        }
        got = HGF.grouping_kernel_config(source)
        self.assertEqual(source["overlap"]["kill_thr"], 0.3)
        self.assertEqual(got["overlap"]["kill_thr"], 1.01)
        self.assertEqual(got["overlap"]["trim_dilate_px"], 0)
        self.assertFalse(got["runtime"]["verbose"])
        self.assertIsNone(got["debug"]["rejection_log_path"])

    def test_output_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already-there"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                HGF.run_study(
                    Path(temporary) / "development",
                    Path(temporary) / "hough",
                    output,
                    Path(temporary) / "config.yaml",
                    expected_scenes=0,
                )


if __name__ == "__main__":
    unittest.main()
