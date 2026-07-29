from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "eval" / "studies") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval" / "studies"))

import factorial_density_width as FDW  # noqa: E402


def _record(seed: int, density: float, width: float, suffix: str):
    same = {
        "mask.png": f"mask-{seed}-{density}",
        "mask_clean.png": f"clean-{seed}-{density}",
        "gt_centerlines_multilabel.npz": f"gt-{seed}-{density}",
    }
    return {
        "seed": seed,
        "shape": [10, 10],
        "target_length_density_px_per_px2": density,
        "achieved_length_density_px_per_px2": density + 0.001,
        "achieved_length_density_um_per_um2": 10.0,
        "total_visible_centreline_length_px": 10.0,
        "last_filament_length_px": 1.0,
        "clean_1px_union_coverage": density,
        "degraded_input_coverage": density / 2,
        "bundle_width_level_px": width,
        "achieved_thick_mask_coverage": 0.1 + width / 100,
        "n_filaments": int(density * 1000),
        "pairwise_crossing_events": int(density * 100),
        "unique_crossing_regions": int(density * 90),
        "crossing_density_per_megapixel": density * 1000,
        "gap_count": int(density * 100),
        "missing_centreline_sample_fraction": 0.1,
        "mean_gap_length_px": 2.0,
        "median_gap_length_px": 2.0,
        "p90_gap_length_px": 3.0,
        "max_gap_length_px": 4.0,
        "filaments_with_no_detected_gap": 1,
        "centreline_gt_content_sha256": f"content-{seed}-{density}",
        "artifact_sha256": {
            **same,
            "sem.png": f"sem-{suffix}",
            "gt_thick_multilabel.npz": f"thick-{suffix}",
        },
        "scene_dir": suffix,
    }


class FactorialStudyTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            _record(seed, density, width, f"{seed}-{density}-{width}")
            for seed in (1, 2)
            for density in (0.02, 0.04)
            for width in (6.0, 11.0)
        ]

    def test_validate_grid_passes_width_negative_control(self):
        result = FDW.validate_grid(self.records)
        self.assertEqual(result["n_cells"], 8)
        self.assertEqual(result["stage3_width_negative_control"]["status"], "pass")

    def test_validate_grid_catches_width_leakage(self):
        self.records[1]["artifact_sha256"]["mask.png"] = "changed"
        with self.assertRaisesRegex(ValueError, "mask.png varies"):
            FDW.validate_grid(self.records)

    def test_empirical_stage3_negative_control(self):
        rows = [
            {
                "seed": record["seed"],
                "target_length_density_px_per_px2": record[
                    "target_length_density_px_per_px2"
                ],
                "bundle_width_level_px": record["bundle_width_level_px"],
                "common_f1": record["seed"] / 10
                + record["target_length_density_px_per_px2"],
            }
            for record in self.records
        ]
        result = FDW.validate_stage3_results(self.records, rows)
        self.assertEqual(result["maximum_within_geometry_score_range"], 0.0)
        rows[1]["common_f1"] += 0.01
        with self.assertRaisesRegex(ValueError, "negative control failed"):
            FDW.validate_stage3_results(self.records, rows)


if __name__ == "__main__":
    unittest.main()
