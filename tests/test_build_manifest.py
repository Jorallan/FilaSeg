"""Isolated fixtures for development + locked manifest construction."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import build_manifest as BM  # noqa: E402


def _row(geometry: str, method: str, *, seed: int = 101, density: int = 20,
         status: str = "ok", error: str = "") -> dict:
    return {
        "geometry_id": geometry, "density": density, "seed": seed,
        "mask_variant": "3", "method": method, "status": status,
        "output_path": f"/outputs/{geometry}/{method}.tif", "error": error,
        "config_hashes": {"reconnect_config_source": "abc123"},
    }


def _report(rows: list[dict], root: str, generator: str) -> dict:
    return {"samples_root": root, "generator_command": generator,
            "configuration": {"mask_name": "mask.png"}, "per_scene": rows}


class BuildManifestTests(unittest.TestCase):
    def test_combines_roles_retains_failure_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dev = tmp / "development.json"; locked = tmp / "locked.json"; out = tmp / "manifest.csv"
            dev.write_text(json.dumps(_report([_row("cov20/dev_a", "baseline_cc", status="failed", error="boom")],
                                              "/data/dev", "python make_dev --seed0 7")), encoding="utf-8")
            locked.write_text(json.dumps(_report([_row("cov20/lock_a", "baseline_cc", seed=801)],
                                                 "/data/locked", "python make_locked --seed0 801")), encoding="utf-8")
            rows, headers = BM.build_manifest(locked, dev)
            BM.write_manifest(rows, headers, out)
            self.assertEqual(len(rows), 2)
            failed = next(r for r in rows if r["dataset_role"] == "development")
            self.assertEqual((failed["status"], failed["failure_notes"]), ("failed", "boom"))
            self.assertEqual(failed["input_mask"], "/data/dev/cov20/dev_a/mask.png")
            self.assertIn("python make_dev --seed0 7", "\n".join(headers))
            self.assertIn('"cov20/lock_a":{"density":20,"seed":801}', "\n".join(headers))
            with out.open(newline="", encoding="utf-8") as fh:
                content = fh.read().splitlines()
            header_index = next(i for i, line in enumerate(content) if line.startswith("seed,"))
            parsed = list(csv.DictReader(content[header_index:]))
            self.assertEqual(set(parsed[0]), set(BM.FIELDS))
            self.assertEqual(len(parsed), 2)

    def test_duplicate_scene_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locked.json"
            path.write_text(json.dumps(_report([_row("cov20/a", "baseline_cc"),
                                                _row("cov20/a", "baseline_cc")], "/data", "gen")),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate scene/method"):
                BM.build_manifest(path)

    def test_inconsistent_seed_for_one_scene_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locked.json"
            path.write_text(json.dumps(_report([_row("cov20/a", "baseline_cc", seed=1),
                                                _row("cov20/a", "baseline_skeleton", seed=2)], "/data", "gen")),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent seed/density"):
                BM.build_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
