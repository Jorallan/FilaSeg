"""Static guards for the reproducible real development-case ablation set."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import real_ablation_runner as R  # noqa: E402


class RealAblationRunnerTests(unittest.TestCase):
    def test_named_ablations_are_unique_and_include_true_single_grid(self):
        variants = {row["name"]: row["arguments"] for row in R.VARIANTS}
        self.assertEqual(len(variants), len(R.VARIANTS))
        self.assertEqual(
            variants["single_tile_grid"],
            ["--tile-grid-offsets", "1", "--tile-grid-vote-min", "1"],
        )

    def test_orientation_gate_ablation_uses_explicit_disable_value(self):
        variants = {row["name"]: row["arguments"] for row in R.VARIANTS}
        self.assertEqual(
            variants["orientation_gate_disabled"],
            ["--reconnect-max-orientation-mismatch-deg", "180"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
