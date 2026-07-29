"""Focused tests for the non-overwriting representation audit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import instance_io as IIO  # noqa: E402
import representation_audit as RA  # noqa: E402


class RepresentationAuditTests(unittest.TestCase):
    def test_centerline_transform_preserves_ids_and_independent_layers(self):
        shape = (31, 31)
        horizontal = np.zeros(shape, dtype=bool)
        vertical = np.zeros(shape, dtype=bool)
        horizontal[13:18, 3:28] = True
        vertical[3:28, 13:18] = True
        source = IIO.InstanceSet(
            shape, {7: horizontal, 9: vertical}, source="thick-overlap"
        )
        got = RA.centerline_instances(source)
        self.assertEqual(set(got.masks), {7, 9})
        self.assertTrue(got.masks[7][15, 15])
        self.assertTrue(got.masks[9][15, 15])
        self.assertLess(got.masks[7].sum(), horizontal.sum())
        self.assertLess(got.masks[9].sum(), vertical.sum())

    def test_stage3_resolution_is_explicit_and_requires_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            final = run / "final"
            stage3 = run / "3.reconnect"
            final.mkdir(parents=True)
            stage3.mkdir()
            final_path = final / "mask_reconnect_multilabel.npz"
            final_path.write_bytes(b"final")
            expected = stage3 / final_path.name
            with self.assertRaises(FileNotFoundError):
                RA.resolve_stage3_artifact(final_path)
            expected.write_bytes(b"stage3")
            self.assertEqual(RA.resolve_stage3_artifact(final_path), expected)
            with self.assertRaisesRegex(ValueError, "final/"):
                RA.resolve_stage3_artifact(expected)

    def test_qualitative_selection_uses_primary_paired_difference(self):
        rows = []
        differences = [-0.9, -0.6, -0.4, -0.02, 0.01, 0.03, 0.2, 0.5, 0.8]
        for index, difference in enumerate(differences):
            rows.append({
                "geometry_id": f"cov20/s{index}",
                "density": 20,
                "scores": {
                    RA.PRIMARY_FILASEG: {"f1": 0.5 + difference / 2},
                    RA.PRIMARY_BASELINE: {"f1": 0.5 - difference / 2},
                },
            })
        selected = RA.select_qualitative_scenes(rows)
        self.assertEqual(
            [row["geometry_id"] for row in selected["three_largest_filaseg_wins"]],
            ["cov20/s8", "cov20/s7", "cov20/s6"],
        )
        self.assertEqual(
            [row["geometry_id"] for row in selected["three_largest_minimum_turn_wins"]],
            ["cov20/s0", "cov20/s1", "cov20/s2"],
        )
        self.assertEqual(
            [row["geometry_id"] for row in selected["two_near_ties"]],
            ["cov20/s4", "cov20/s3"],
        )

    def test_source_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locked.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                RA.build_report(path, path)


if __name__ == "__main__":
    unittest.main()
