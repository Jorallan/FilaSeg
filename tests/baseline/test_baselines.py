"""Tests for eval/baselines.py -- pure mask-only classical baselines.

Run with:
    C:\\Repos\\venv_cnt\\Scripts\\python.exe -m unittest tests.test_baselines -v
from the repository root. Builds small synthetic masks with numpy/skimage
only; never runs the FilaSeg pipeline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from skimage.draw import line as skline
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.baseline.baselines import (
    baseline_connected_components,
    baseline_skeleton_minturn,
    minimum_turn_group_fragments,
)


def _n_instances(lab: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(lab)))


def _thick_line(shape, r0, c0, r1, c1, width=3):
    m = np.zeros(shape, dtype=bool)
    rr, cc = skline(r0, c0, r1, c1)
    m[rr, cc] = True
    if width > 1:
        m = binary_dilation(m, iterations=width // 2)
    return m


class TestBaselineConnectedComponents(unittest.TestCase):
    def test_two_disjoint_filaments(self):
        shape = (100, 100)
        m1 = _thick_line(shape, 10, 10, 10, 80, width=3)
        m2 = _thick_line(shape, 60, 10, 60, 80, width=3)
        mask = m1 | m2
        lab = baseline_connected_components(mask, min_area=15)
        n = _n_instances(lab)
        print(f"[A/disjoint] instances = {n}")
        self.assertEqual(n, 2)

    def test_broken_filament_not_bridged(self):
        shape = (100, 100)
        m1 = _thick_line(shape, 50, 10, 50, 45, width=3)
        m2 = _thick_line(shape, 50, 51, 50, 90, width=3)  # 6 px gap (45..51)
        mask = m1 | m2
        lab = baseline_connected_components(mask, min_area=15)
        n = _n_instances(lab)
        print(f"[A/broken] instances = {n}")
        self.assertEqual(n, 2)

    def test_crossing_fuses_into_one(self):
        shape = (100, 100)
        m1 = _thick_line(shape, 10, 50, 90, 50, width=3)  # vertical
        m2 = _thick_line(shape, 50, 10, 50, 90, width=3)  # horizontal -> X-ish (a +)
        # Rotate to an actual X (diagonal) crossing instead of a +.
        m1 = _thick_line(shape, 10, 10, 90, 90, width=3)
        m2 = _thick_line(shape, 10, 90, 90, 10, width=3)
        mask = m1 | m2
        lab = baseline_connected_components(mask, min_area=15)
        n = _n_instances(lab)
        print(f"[A/crossing] instances = {n}")
        self.assertEqual(n, 1)

    def test_speck_rejected(self):
        shape = (100, 100)
        long_fil = _thick_line(shape, 20, 20, 20, 90, width=3)
        speck = np.zeros(shape, dtype=bool)
        speck[70, 70] = True
        speck[70, 71] = True
        speck[71, 70] = True  # exactly 3 px blob, well below min_area=15
        speck = speck & ~long_fil
        mask = long_fil | speck
        lab = baseline_connected_components(mask, min_area=15)
        n = _n_instances(lab)
        print(f"[A/speck] instances = {n}")
        self.assertEqual(n, 1)

    def test_determinism(self):
        shape = (80, 80)
        m1 = _thick_line(shape, 10, 10, 70, 70, width=3)
        lab1 = baseline_connected_components(m1, min_area=15)
        lab2 = baseline_connected_components(m1, min_area=15)
        self.assertTrue(np.array_equal(lab1, lab2))


class TestExplicitFragmentMinimumTurn(unittest.TestCase):
    def test_preserves_support_and_groups_gap(self):
        shape = (80, 100)
        left = _thick_line(shape, 40, 5, 40, 42, width=1)
        right = _thick_line(shape, 40, 49, 40, 94, width=1)
        groups = minimum_turn_group_fragments(
            [left, right], max_gap_px=15.0, max_turn_deg=45.0
        )
        self.assertEqual(len(groups), 1)
        only = next(iter(groups.values()))
        self.assertTrue(np.array_equal(only, left | right))
        self.assertFalse(only[40, 45], "grouping must not render a bridge")

    def test_keeps_overlapping_inputs_distinct(self):
        shape = (80, 80)
        horizontal = _thick_line(shape, 40, 5, 40, 74, width=1)
        vertical = _thick_line(shape, 5, 40, 74, 40, width=1)
        groups = minimum_turn_group_fragments(
            [horizontal, vertical], max_gap_px=2.0, max_turn_deg=30.0
        )
        self.assertEqual(len(groups), 2)
        supports = list(groups.values())
        self.assertTrue(any(np.array_equal(mask, horizontal) for mask in supports))
        self.assertTrue(any(np.array_equal(mask, vertical) for mask in supports))


class TestBaselineSkeletonMinturn(unittest.TestCase):
    def test_two_disjoint_filaments(self):
        shape = (100, 100)
        m1 = _thick_line(shape, 10, 10, 10, 80, width=3)
        m2 = _thick_line(shape, 60, 10, 60, 80, width=3)
        mask = m1 | m2
        lab = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        n = _n_instances(lab)
        print(f"[B/disjoint] instances = {n}")
        self.assertEqual(n, 2)

    def test_broken_filament_bridged(self):
        shape = (100, 100)
        m1 = _thick_line(shape, 50, 10, 50, 45, width=3)
        m2 = _thick_line(shape, 50, 51, 50, 90, width=3)  # 6 px gap
        mask = m1 | m2

        lab_cc = baseline_connected_components(mask, min_area=15)
        n_cc = _n_instances(lab_cc)
        print(f"[A/broken-for-B-comparison] instances = {n_cc}")
        self.assertEqual(n_cc, 2)

        lab = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        n = _n_instances(lab)
        print(f"[B/broken] instances = {n}")
        self.assertEqual(n, 1)

    def test_crossing_separated(self):
        shape = (120, 120)
        m1 = _thick_line(shape, 10, 10, 110, 110, width=3)
        m2 = _thick_line(shape, 10, 110, 110, 10, width=3)
        mask = m1 | m2

        lab_cc = baseline_connected_components(mask, min_area=15)
        n_cc = _n_instances(lab_cc)
        print(f"[A/crossing-for-B-comparison] instances = {n_cc}")
        self.assertEqual(n_cc, 1)

        lab = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        n = _n_instances(lab)
        print(f"[B/crossing] instances = {n}")
        self.assertEqual(
            n, 2,
            "baseline B should separate the X crossing via minimum-turn "
            "pairing (opposite arms joined, adjacent arms rejected as a "
            "90-degree turn); investigate junction handling if this fails.",
        )

    def test_full_pixel_coverage(self):
        shape = (150, 150)
        rng = np.random.RandomState(0)
        mask = np.zeros(shape, dtype=bool)
        mask |= _thick_line(shape, 5, 5, 140, 140, width=3)
        mask |= _thick_line(shape, 5, 140, 140, 5, width=3)
        mask |= _thick_line(shape, 75, 5, 75, 140, width=3)
        mask |= _thick_line(shape, 5, 75, 140, 75, width=3)
        mask |= _thick_line(shape, 20, 100, 100, 20, width=2)
        lab = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        n = _n_instances(lab)
        print(f"[B/coverage] instances = {n}, "
              f"fg_px = {int(mask.sum())}, labelled_px = {int((lab > 0).sum())}")
        self.assertEqual(int((lab > 0).sum()), int(mask.sum()))

    def test_determinism(self):
        shape = (120, 120)
        m1 = _thick_line(shape, 10, 10, 110, 110, width=3)
        m2 = _thick_line(shape, 10, 110, 110, 10, width=3)
        mask = m1 | m2
        lab1 = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        lab2 = baseline_skeleton_minturn(mask, max_gap_px=15.0)
        print(f"[B/determinism] instances run1 = {_n_instances(lab1)}, "
              f"run2 = {_n_instances(lab2)}")
        self.assertTrue(np.array_equal(lab1, lab2))


if __name__ == "__main__":
    unittest.main()
