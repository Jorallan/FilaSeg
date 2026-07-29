from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "eval", ROOT / "eval" / "generators"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import synth_factorial as SF  # noqa: E402


class FactorialGeneratorTests(unittest.TestCase):
    def test_locked_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            SF.reject_locked(ROOT / "input" / "synthetic_locked_v1")

    def test_small_grid_is_complete_and_width_invariant(self):
        params = SF.T.thick_params()
        params.update(len_frac=(0.20, 0.45), false_pos=0.0, preview=False)
        with tempfile.TemporaryDirectory() as td:
            records = SF.generate_factorial(
                Path(td) / "development",
                length_densities=(0.012, 0.024),
                width_centres_px=(4.0, 8.0),
                seeds=(701,),
                shape=(96, 96),
                width_half_range_px=0.25,
                params=params,
            )
            self.assertEqual(len(records), 4)
            for density in (0.012, 0.024):
                group = [
                    record
                    for record in records
                    if record["target_length_density_px_per_px2"] == density
                ]
                self.assertEqual(len(group), 2)
                for name in (
                    "mask.png",
                    "mask_clean.png",
                    "gt_centerlines_multilabel.npz",
                ):
                    self.assertEqual(
                        len({record["artifact_sha256"][name] for record in group}), 1
                    )
                self.assertEqual(
                    len({record["pairwise_crossing_events"] for record in group}), 1
                )
                self.assertEqual(len({record["gap_count"] for record in group}), 1)
                self.assertLess(
                    group[0]["achieved_thick_mask_coverage"],
                    group[1]["achieved_thick_mask_coverage"],
                )
            manifest = json.loads(
                (Path(td) / "development" / "factorial_manifest.json").read_text()
            )
            self.assertEqual(manifest["n_cells"], 4)

    def test_gap_statistics_have_consistent_counts(self):
        filament = SF.G.Filament(
            1,
            np.asarray([[2.0, 5.0], [17.0, 5.0]], dtype=np.float32),
            1.0,
            1.0,
            np.ones(2, dtype=np.float32),
        )
        mask = np.zeros((20, 20), dtype=bool)
        mask[5, 2:8] = True
        mask[5, 13:18] = True
        result = SF.gap_statistics(mask, [filament])
        self.assertGreater(result["gap_count"], 0)
        self.assertEqual(
            result["missing_centreline_samples"],
            round(
                result["missing_centreline_sample_fraction"]
                * result["total_centreline_samples"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
