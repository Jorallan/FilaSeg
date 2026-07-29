"""Focused regression tests for scene-level statistical summaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import _evalpath  # noqa: F401,E402  (restores the flat eval/ import namespace)
import stats_util as SU  # noqa: E402


class TestStatsUtil(unittest.TestCase):

    def test_scene_keyed_pairing_is_deterministic_and_order_independent(self):
        a = {"scene-2": 0.8, "scene-1": 0.4, "scene-3": 0.9}
        b = {"scene-3": 0.3, "scene-1": 0.2, "scene-2": 0.5}
        first = SU.paired_summarize_by_scene(a, b, seed=9, n_boot=500)
        second = SU.paired_summarize_by_scene(dict(reversed(list(a.items()))), b,
                                               seed=9, n_boot=500)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertAlmostEqual(first.mean, (0.2 + 0.3 + 0.6) / 3.0)

    def test_scene_key_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scene IDs must match exactly"):
            SU.paired_summarize_by_scene({"a": 1.0}, {"b": 1.0}, n_boot=10)

    def test_stopping_rule(self):
        narrow = SU.Summary(3, 0.5, 0.1, 0.48, 0.52, 0.02)
        wide = SU.Summary(3, 0.5, 0.1, 0.44, 0.56, 0.06)
        self.assertFalse(SU.needs_more_scenes(narrow, target_halfwidth=0.05))
        self.assertTrue(SU.needs_more_scenes(wide, target_halfwidth=0.05))

    def test_degenerate_pearson_is_explicit_nan(self):
        result = SU.pearson_r([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])
        self.assertEqual(result["n"], 3)
        self.assertTrue(np.isnan(result["r"]))
        self.assertTrue(np.isnan(result["r_ci_lo"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
