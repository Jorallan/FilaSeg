"""Fixture coverage for paper/scripts/results/make_result_macros.py."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The manuscript lives in a separate repository and paper/ is only a local
# working copy, untracked here. A fresh clone of the coding repository has no
# paper/ directory, so skip this fixture rather than failing the whole suite.
_RESULT_SCRIPTS = ROOT / "paper" / "scripts" / "results"
if (_RESULT_SCRIPTS / "make_result_macros.py").is_file():
    sys.path.insert(0, str(_RESULT_SCRIPTS))
    import make_result_macros as M  # noqa: E402
else:
    M = None


def _row(method: str, scene: str, density: int, f1: float, *, status: str = "ok"):
    row = {"method": method, "geometry_id": scene, "mask_variant": "axis", "density": density, "status": status}
    if status == "ok":
        row["scores"] = {"common_metric": {"precision": f1 + .1, "recall": f1 - .1, "f1": f1,
                                               "fragment_recovery_recovery_rate": f1 + .05},
                         "native_diagnostics": {"clustering": {"pairwise_precision": f1 + .1,
                                                                    "pairwise_recall": f1 - .1, "pairwise_f1": f1},
                                                "instance": {"recovery_rate": f1}, "panoptic": {"pq": f1 - .2}}}
    return row


def _stat(value: float, n: int = 2):
    return {
        "n": n,
        "mean": value,
        "sd": .01,
        "ci_lo": value - .02,
        "ci_hi": value + .02,
        "ci_halfwidth": .02,
    }


def _corrected_method(f1: float):
    return {
        "precision": _stat(f1 + .1),
        "recall": _stat(f1 - .1),
        "f1": _stat(f1),
        "fragment_recovery_recovery_rate": _stat(f1 + .05),
    }


def _corrected_pair(delta: float):
    return {
        "precision": _stat(delta),
        "recall": _stat(delta),
        "f1": _stat(delta),
        "fragment_recovery_recovery_rate": _stat(delta),
    }


@unittest.skipIf(M is None, "paper/figscripts/make_result_macros.py not present "
                            "(manuscript repository not checked out alongside)")
class ResultMacrosTests(unittest.TestCase):
    def test_builds_deterministic_macros_summary_and_baseline_table(self):
        locked = {"per_scene": [_row("filaseg", "a", 20, .8), _row("baseline_cc", "a", 20, .5),
                                  _row("filaseg", "b", 20, .6), _row("baseline_cc", "b", 20, .4),
                                  _row("baseline_cc", "c", 20, .1, status="failed")]}
        curvature = {"runs": [{**_row("unused", "d", 20, .7), "variant": "geometric", "case_type": "synthetic"}]}
        real = {"runs": [{**_row("unused", "real", 0, .75), "variant": "locked", "case_type": "real_development"}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for name, data in (("locked.json", locked), ("curve.json", curvature), ("real.json", real)):
                path = root / name; path.write_text(json.dumps(data), encoding="utf-8"); paths.append(path)
            result = M.build_results(*paths, root / "results")
            self.assertEqual(result["locked"]["locked_scene_count"], 3)
            paired = result["locked"]["paired_filaseg_minus_baseline"]["baseline_cc"]["f1"]
            self.assertEqual(paired["n_paired_scenes"], 2)
            self.assertAlmostEqual(paired["summary"]["mean"], .25)
            density_paired = result["locked"]["per_density"]["20"]["paired_filaseg_minus_baseline"]["baseline_cc"]["f1"]
            self.assertEqual(density_paired["n_paired_scenes"], 2)
            self.assertEqual(result["locked"]["methods"]["baseline_cc"]["failure_count"], 1)
            tex = (root / "results" / "filaseg_results.tex").read_text(encoding="utf-8")
            table = (root / "results" / "filaseg_baseline_table.tex").read_text(encoding="utf-8")
            self.assertIn(r"\newcommand{\LockedFilaSegCommonFOne}", tex)
            self.assertIn(r"\newcommand{\LockedDensityTwoZeroFilaSegCommonFOneMean}", tex)
            commands = re.findall(r"\\newcommand\{\\([^}]+)\}", tex)
            self.assertTrue(commands)
            for command in commands:
                self.assertRegex(command, r"^[A-Za-z]+$")
            self.assertNotIn("Connected components", table)
            self.assertNotIn("Failures", table)
            sanity = (root / "results" / "filaseg_sanity_table.tex").read_text(encoding="utf-8")
            self.assertIn("Connected components", sanity)
            self.assertTrue((root / "results" / "filaseg_results_summary.json").is_file())

    def test_missing_filaseg_is_explicit_not_invented(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, data in (("locked", {"per_scene": [_row("baseline_cc", "a", 20, .5)]}),
                               ("curve", {"runs": []}), ("real", {"runs": []})):
                (root / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
            result = M.build_results(root / "locked.json", root / "curve.json", root / "real.json", root / "out")
            self.assertNotIn("LockedFilaSegCommonFOne", result["macros"])
            self.assertNotIn("RealLockedCommonFOne", result["macros"])

    def test_corrected_report_uses_stored_stratified_summaries_and_stage3_headline(self):
        locked = {"per_scene": [
            _row("filaseg", "a", 20, .4),
            _row("baseline_skeleton", "a", 20, .5),
            _row("baseline_cc", "a", 20, .1),
        ]}
        by_method = {
            "connected_components_centerline": _corrected_method(.1),
            "minimum_turn_centerline": _corrected_method(.5),
            "filaseg_stage3_centerline": _corrected_method(.8),
            "filaseg_stage4_centerline": _corrected_method(.7),
            "filaseg_stage4_rendered": _corrected_method(.6),
        }
        paired = {
            "primary_filaseg_minus_minimum_turn": _corrected_pair(.3),
            "primary_filaseg_minus_connected_components": _corrected_pair(.7),
            "stage4_rendered_minus_stage3_centerline": _corrected_pair(-.2),
            "stage4_centerline_minus_stage3_centerline": _corrected_pair(-.1),
            "stage4_rendered_minus_stage4_centerline": _corrected_pair(-.1),
        }
        corrected = {
            "per_scene": [{"geometry_id": "a"}, {"geometry_id": "b"}],
            "aggregate": {
                "bootstrap": {"overall_resampling": "within_density_then_scene_weighted_pooling"},
                "by_method": by_method,
                "paired": paired,
                "by_density": {},
            },
        }
        curvature = {"runs": []}
        real = {"runs": []}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, data in (
                ("locked.json", locked),
                ("corrected.json", corrected),
                ("curve.json", curvature),
                ("real.json", real),
            ):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            result = M.build_results(
                root / "locked.json",
                root / "curve.json",
                root / "real.json",
                root / "results",
                representation_json=root / "corrected.json",
            )
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["locked"]["endpoint_source"], "representation_audit_corrected")
            self.assertEqual(result["macros"]["LockedFilaSegCommonFOneMean"], "0.800")
            self.assertEqual(
                result["macros"]["LockedFilaSegStageFourRenderedCommonFOneMean"],
                "0.600",
            )
            self.assertEqual(
                result["macros"]["FilaSegMinusBaselineSkeletonCommonFOneMean"],
                "0.300",
            )
            self.assertEqual(
                result["macros"][
                    "FilaSegStageFourRenderedMinusStageThreeCommonFOneMean"
                ],
                "-0.200",
            )
            table = (root / "results" / "filaseg_baseline_table.tex").read_text(encoding="utf-8")
            self.assertIn("FilaSeg (Stage 3 centreline)", table)
            self.assertIn("Paired difference", table)
            self.assertNotIn("Connected components", table)


if __name__ == "__main__":
    unittest.main()
