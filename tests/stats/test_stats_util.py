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

    def test_stratified_summary_is_deterministic_and_preserves_pooled_mean(self):
        values = {"dense": [0.1, 0.4, 0.9], "sparse": [0.2, 0.8]}
        first = SU.stratified_summarize(values, seed=9, n_boot=500)
        second = SU.stratified_summarize(dict(reversed(list(values.items()))),
                                         seed=9, n_boot=500)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertAlmostEqual(first.mean, np.mean([0.1, 0.4, 0.9, 0.2, 0.8]))

    def test_stratified_ci_differs_from_unstratified_on_imbalanced_fixture(self):
        values = {
            "large-low-variance": [0.0] * 100,
            "small-high-variance": [-20.0, 20.0],
        }
        pooled = [v for group in values.values() for v in group]
        stratified = SU.stratified_summarize(values, seed=17, n_boot=2000)
        unstratified = SU.summarize(pooled, seed=17, n_boot=2000)
        self.assertAlmostEqual(stratified.mean, unstratified.mean)
        self.assertNotEqual((stratified.ci_lo, stratified.ci_hi),
                            (unstratified.ci_lo, unstratified.ci_hi))

    def test_paired_stratified_validates_alignment_and_stratum_keys(self):
        with self.assertRaisesRegex(ValueError, "stratum keys must match exactly"):
            SU.paired_stratified_summarize({"one": [1.0]}, {"two": [0.0]},
                                           n_boot=10)
        with self.assertRaisesRegex(ValueError, "must align within stratum"):
            SU.paired_stratified_summarize({"one": [1.0, 2.0]},
                                           {"one": [0.0]}, n_boot=10)

    def test_scene_stratified_pairing_validates_strata_and_is_order_independent(self):
        a = {"scene-2": 0.8, "scene-1": 0.4, "scene-3": 0.9}
        b = {"scene-3": 0.3, "scene-1": 0.2, "scene-2": 0.5}
        strata = {"scene-1": "low", "scene-2": "high", "scene-3": "high"}
        first = SU.paired_stratified_summarize_by_scene(a, b, strata,
                                                        seed=9, n_boot=500)
        second = SU.paired_stratified_summarize_by_scene(
            dict(reversed(list(a.items()))), b, dict(reversed(list(strata.items()))),
            seed=9, n_boot=500)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertAlmostEqual(first.mean, (0.2 + 0.3 + 0.6) / 3.0)
        with self.assertRaisesRegex(ValueError, "stratum scene IDs must match"):
            SU.paired_stratified_summarize_by_scene(a, b, {"scene-1": "low"},
                                                    n_boot=10)

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
