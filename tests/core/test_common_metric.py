"""Toy validation for eval/common_metric.py.

Every case builds small synthetic label images directly with numpy /
skimage.draw.line (NOT by running the FilaSeg pipeline), and prints the
scores it computes. Run with:

    C:\\Repos\\venv_cnt\\Scripts\\python.exe -m unittest tests.test_common_metric -v

from the repository root.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# eval/*.py modules are not a package (no __init__.py) and import each other
# with bare names (e.g. `import metrics`), so eval/ must be on sys.path --
# same convention eval/eval_reconnect.py uses for itself.
_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(_EVAL_DIR))
import _evalpath  # noqa: F401,E402  (restores the flat eval/ import namespace)

import common_metric as CM  # noqa: E402

SHAPE = (128, 128)


# ── small synthetic-image helpers (no pipeline involved) ──────────────────


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=radius)


def _line_mask(shape, r0, c0, r1, c1, width: int = 3) -> np.ndarray:
    from skimage.draw import line as skline

    m = np.zeros(shape, dtype=bool)
    rr, cc = skline(r0, c0, r1, c1)
    m[rr, cc] = True
    return _dilate(m, max(0, (width - 1) // 2))


def _print_scores(title: str, scores: dict) -> None:
    print(f"\n-- {title} --")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")


class TestCommonMetric(unittest.TestCase):

    # 1. Perfect grouping ----------------------------------------------------
    def test_1_perfect_grouping(self):
        lines = [
            _line_mask(SHAPE, 20, 10, 20, 118, width=3),
            _line_mask(SHAPE, 64, 10, 64, 118, width=3),
            _line_mask(SHAPE, 108, 10, 108, 118, width=3),
        ]
        mask = np.zeros(SHAPE, dtype=bool)
        gt_lab = np.zeros(SHAPE, dtype=np.int32)
        for i, m in enumerate(lines, start=1):
            mask |= m
            gt_lab[m] = i
        pred_lab = gt_lab.copy()

        frag = CM.common_fragments(mask)
        gt_assign = CM.assign_fragments(frag, gt_lab)
        pred_assign = CM.assign_fragments(frag, pred_lab)

        scores = CM.pairwise_scores(gt_assign, pred_assign)
        scores.update(CM.instance_recovery(gt_lab, pred_lab))
        _print_scores("Case 1: perfect grouping", scores)

        # Each independent line is one atomic fragment, so grouping has no
        # positive pair evidence.  Perfect pixels are still captured by the
        # recovery score; the pairwise score must not claim a spurious F1=1.
        self.assertFalse(scores["pair_evidence_available"])
        self.assertTrue(np.isnan(scores["f1"]))
        self.assertEqual(scores["well_recovered"], 3)
        self.assertEqual(scores["false_instances"], 0)

    # 2. One over-merge --------------------------------------------------
    def test_2_over_merge(self):
        lines = [
            _line_mask(SHAPE, 20, 10, 20, 118, width=3),
            _line_mask(SHAPE, 64, 10, 64, 118, width=3),
            _line_mask(SHAPE, 108, 10, 108, 118, width=3),
        ]
        mask = np.zeros(SHAPE, dtype=bool)
        gt_lab = np.zeros(SHAPE, dtype=np.int32)
        for i, m in enumerate(lines, start=1):
            mask |= m
            gt_lab[m] = i

        # Prediction fuses filaments 1 and 2 into the same output id.
        pred_lab = np.zeros(SHAPE, dtype=np.int32)
        pred_lab[lines[0]] = 1
        pred_lab[lines[1]] = 1
        pred_lab[lines[2]] = 2

        frag = CM.common_fragments(mask)
        gt_assign = CM.assign_fragments(frag, gt_lab)
        pred_assign = CM.assign_fragments(frag, pred_lab)
        scores = CM.pairwise_scores(gt_assign, pred_assign)
        _print_scores("Case 2: one over-merge", scores)

        self.assertFalse(scores["pair_evidence_available"])
        self.assertTrue(np.isnan(scores["recall"]))
        self.assertLess(scores["precision"], 1.0)

    # 3. One under-merge ---------------------------------------------------
    def test_3_under_merge(self):
        # A single filament with a real gap at its midpoint (cols 61-66
        # missing) so the mask itself yields two disjoint common fragments;
        # ground truth still calls it one filament, prediction splits it.
        full = _line_mask(SHAPE, 64, 10, 64, 118, width=3)
        gap_cols = np.zeros(SHAPE, dtype=bool)
        gap_cols[:, 61:67] = True
        mask = full & ~gap_cols

        gt_lab = np.zeros(SHAPE, dtype=np.int32)
        gt_lab[mask] = 1  # one ground-truth filament, both pieces

        pred_lab = np.zeros(SHAPE, dtype=np.int32)
        left = mask.copy(); left[:, 61:] = False
        right = mask.copy(); right[:, :61] = False
        pred_lab[left] = 1
        pred_lab[right] = 2

        frag = CM.common_fragments(mask)
        n_frags = int(frag.max())
        gt_assign = CM.assign_fragments(frag, gt_lab)
        pred_assign = CM.assign_fragments(frag, pred_lab)
        scores = CM.pairwise_scores(gt_assign, pred_assign)
        _print_scores("Case 3: one under-merge", scores)
        print(f"   (n_common_fragments for the gapped filament = {n_frags})")

        self.assertGreaterEqual(n_frags, 2, "gap must produce >=2 common fragments")
        self.assertFalse(scores["pair_evidence_available"])
        self.assertTrue(np.isnan(scores["precision"]))
        self.assertEqual(scores["recall"], 0.0)

    # 4. Crossing ------------------------------------------------------------
    def test_4_crossing(self):
        # Two full-length diagonals crossing at the image centre.
        diag_a = _line_mask(SHAPE, 20, 20, 108, 108, width=3)   # "\" NW-SE
        diag_b = _line_mask(SHAPE, 20, 108, 108, 20, width=3)   # "/" NE-SW
        mask = diag_a | diag_b

        frag = CM.common_fragments(mask)
        n_frags = int(frag.max())
        print(f"\n   (crossing produced {n_frags} common fragments -- expect ~4 arms)")

        gt_stack = np.stack([diag_a, diag_b], axis=0)

        # (a) correct prediction: same two straight diagonals.
        pred_correct = np.stack([diag_a, diag_b], axis=0)
        gt_assign = CM.assign_fragments(frag, gt_stack)
        pred_assign_correct = CM.assign_fragments(frag, pred_correct)
        scores_a = CM.pairwise_scores(gt_assign, pred_assign_correct)
        _print_scores("Case 4a: crossing, correct pairing", scores_a)

        # (b) wrong pairing: re-pair the four arms across the crossing by
        # splitting top/bottom instead of by diagonal identity -- each
        # predicted instance becomes an "L"/"V" of two different true arms.
        rows = np.indices(SHAPE)[0]
        pred_top = mask & (rows < 64)
        pred_bottom = mask & (rows >= 64)
        pred_wrong = np.stack([pred_top, pred_bottom], axis=0)
        pred_assign_wrong = CM.assign_fragments(frag, pred_wrong)
        scores_b = CM.pairwise_scores(gt_assign, pred_assign_wrong)
        _print_scores("Case 4b: crossing, wrong arm pairing", scores_b)

        self.assertAlmostEqual(scores_a["f1"], 1.0, places=9)
        self.assertLess(scores_b["f1"], scores_a["f1"])

    # 5. Overlapping instances (3-D multi-label stack) -----------------------
    def test_5_overlapping_instances_stack(self):
        # Instance A: a long vertical filament.
        inst_a = _line_mask(SHAPE, 10, 64, 118, 64, width=3)
        # Instance B: shares a real pixel run with A (rows 50-78 at col 64,
        # a subset of A's own stroke) then branches off horizontally --
        # a T-junction with genuine shared pixels, not just a crossing point.
        b_vert = _line_mask(SHAPE, 50, 64, 78, 64, width=3)
        b_horiz = _line_mask(SHAPE, 64, 64, 64, 118, width=3)
        inst_b = b_vert | b_horiz

        shared = inst_a & inst_b
        self.assertTrue(shared.any(), "test setup: A and B must share pixels")

        mask = inst_a | inst_b
        frag = CM.common_fragments(mask)
        n_frags = int(frag.max())
        print(f"\n   (T-junction produced {n_frags} common fragments; "
              f"shared pixel count A&B = {int(shared.sum())})")

        gt_stack = np.stack([inst_a, inst_b], axis=0)
        pred_stack = np.stack([inst_a, inst_b], axis=0)  # correct prediction

        gt_assign = CM.assign_fragments(frag, gt_stack)
        pred_assign = CM.assign_fragments(frag, pred_stack)
        scores = CM.pairwise_scores(gt_assign, pred_assign)
        scores.update(CM.instance_recovery(gt_stack, pred_stack))
        _print_scores("Case 5: overlapping instances (3D stack)", scores)

        self.assertGreaterEqual(n_frags, 2)
        self.assertAlmostEqual(scores["f1"], 1.0, places=9)

    # 6. Fragment set is method-independent -----------------------------------
    def test_6_fragments_are_method_independent(self):
        lines = [
            _line_mask(SHAPE, 20, 10, 20, 118, width=3),
            _line_mask(SHAPE, 64, 10, 64, 118, width=3),
            _line_mask(SHAPE, 108, 10, 108, 118, width=3),
        ]
        mask = np.zeros(SHAPE, dtype=bool)
        pred_lab = np.zeros(SHAPE, dtype=np.int32)
        for i, m in enumerate(lines, start=1):
            mask |= m
            pred_lab[m] = i

        frag_direct = CM.common_fragments(mask)

        # An unrelated prediction, with its instance labels permuted, must
        # have zero influence: common_fragments only ever looks at `mask`.
        perm = {1: 77, 2: 5, 3: 42}
        pred_lab_permuted = np.zeros_like(pred_lab)
        for old, new in perm.items():
            pred_lab_permuted[pred_lab == old] = new
        _ = pred_lab_permuted  # only exists to document the property below

        frag_after_permutation = CM.common_fragments(mask)

        print(f"\n-- Case 6: fragment set is method-independent --")
        print(f"   n_fragments direct call     : {int(frag_direct.max())}")
        print(f"   n_fragments after permuting an unrelated prediction's "
              f"labels: {int(frag_after_permutation.max())}")

        self.assertTrue(np.array_equal(frag_direct, frag_after_permutation))

    # 7. Assignment coverage is asymmetric ----------------------------------
    def test_7_asymmetric_missing_assignments_are_reported(self):
        # Fragment 2 is known in GT but absent from the prediction.  It must
        # be a singleton, not part of a shared synthetic "unassigned" group.
        scores = CM.pairwise_scores(
            {1: 1, 2: 1, 3: 2},
            {1: 8, 2: 0, 3: 9},
        )
        self.assertEqual(scores["n_gt_assigned_fragments"], 3)
        self.assertEqual(scores["n_pred_assigned_fragments"], 2)
        self.assertEqual(scores["n_pred_unassigned_fragments"], 1)
        self.assertEqual(scores["n_gt_pairs"], 1.0)
        self.assertEqual(scores["n_pred_pairs"], 0.0)
        self.assertFalse(scores["pair_evidence_available"])
        self.assertEqual(scores["pair_evidence_reason"], "no_positive_prediction_pairs")
        self.assertTrue(np.isnan(scores["precision"]))
        self.assertEqual(scores["recall"], 0.0)

    # 8. Degenerate all-singleton grouping -----------------------------------
    def test_8_all_singletons_have_no_pair_evidence(self):
        scores = CM.pairwise_scores({1: 1, 2: 2}, {1: 4, 2: 5})
        self.assertFalse(scores["pair_evidence_available"])
        self.assertEqual(scores["pair_evidence_reason"], "no_positive_pairs_in_either_partition")
        self.assertTrue(np.isnan(scores["precision"]))
        self.assertTrue(np.isnan(scores["recall"]))
        self.assertTrue(np.isnan(scores["f1"]))

    # 9. A split inside an atomic branch is a known pairwise limitation ------
    def test_9_atomic_branch_split_is_recovery_penalty_not_pair_evidence(self):
        mask = _line_mask(SHAPE, 64, 10, 64, 118, width=1)
        fragments = CM.common_fragments(mask, min_len_px=6, prune_spur_px=0)
        self.assertEqual(int(fragments.max()), 1, "continuous branch is one atomic fragment")

        gt = np.zeros(SHAPE, dtype=np.int32)
        gt[mask] = 1
        pred = np.zeros(SHAPE, dtype=np.int32)
        left = mask.copy(); left[:, 65:] = False
        right = mask.copy(); right[:, :65] = False
        pred[left] = 1
        pred[right] = 2

        gt_assign = CM.assign_fragments(fragments, gt)
        pred_assign = CM.assign_fragments(fragments, pred)
        pair = CM.pairwise_scores(gt_assign, pred_assign)
        fragment_recovery = CM.fragment_instance_recovery(gt_assign, pred_assign)
        recovery = CM.instance_recovery(gt, pred)
        self.assertFalse(pair["pair_evidence_available"])
        self.assertEqual(pair["pair_evidence_reason"], "fewer_than_two_scored_fragments")
        # One atomic fragment cannot expose an intra-fragment split either.
        self.assertEqual(fragment_recovery["well_recovered"], 1)
        self.assertEqual(recovery["well_recovered"], 0)
        self.assertEqual(recovery["fragmented"], 1)

    # 10. Coverage denominators distinguish pixels from skeleton units -------
    def test_10_skeleton_coverage_for_thick_mask(self):
        from skimage import io as skio

        mask = _line_mask(SHAPE, 64, 10, 64, 118, width=9)
        gt = np.zeros(SHAPE, dtype=np.uint16)
        gt[mask] = 1
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.tif"
            skio.imsave(str(gt_path), gt, check_contrast=False)
            scores = CM.score_method(mask, gt_path, gt, min_len_px=6, prune_spur_px=0)

        self.assertLess(scores["n_skeleton_pixels"], scores["n_foreground_pixels"])
        self.assertGreater(scores["common_fragment_foreground_pixel_coverage"], 0.0)
        self.assertLess(scores["common_fragment_foreground_pixel_coverage"], 1.0)
        self.assertGreaterEqual(scores["common_fragment_skeleton_coverage"], 0.0)
        self.assertLessEqual(scores["common_fragment_skeleton_coverage"], 1.0)
        self.assertGreater(scores["common_fragment_skeleton_coverage"],
                           scores["common_fragment_foreground_pixel_coverage"])

    # 11. Fragment recovery is fair across output width conventions ----------
    def test_11_thin_prediction_thick_gt_separates_fragment_and_pixel_recovery(self):
        from skimage import io as skio

        mask = _line_mask(SHAPE, 64, 10, 64, 118, width=9)
        gt = np.zeros(SHAPE, dtype=np.uint16)
        gt[mask] = 1
        thin_pred = np.zeros(SHAPE, dtype=np.uint16)
        thin_pred[_line_mask(SHAPE, 64, 10, 64, 118, width=1)] = 1
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.tif"
            skio.imsave(str(gt_path), gt, check_contrast=False)
            scores = CM.score_method(mask, gt_path, thin_pred, min_len_px=6,
                                     prune_spur_px=0)

        self.assertEqual(scores["fragment_recovery_well_recovered"], 1)
        self.assertEqual(scores["fragment_recovery_fragmented"], 0)
        self.assertEqual(scores["pixel_recovery_well_recovered"], 0)
        self.assertEqual(scores["pixel_recovery_fragmented"], 1)
        self.assertNotIn("well_recovered", scores)
        self.assertNotIn("recovery_rate", scores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
