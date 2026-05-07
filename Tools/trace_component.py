"""Trace a label ID from any pipeline step back to preprocess/stringart branch origins.

Usage examples
--------------
# Trace final IDs 25, 38, 42 in the most recent run:
python Tools/trace_component.py --run output/full_pipeline/<run_folder> --ids 25 38 42

# Trace from reconnect step:
python Tools/trace_component.py --run output/full_pipeline/<run_folder> --step reconnect --ids 7 24

# Trace from postprocess step:
python Tools/trace_component.py --run output/full_pipeline/<run_folder> --step postprocess --ids 5

Outputs
-------
For each ID, prints:
  - bounding box in the source label image
  - preprocess branch files containing its pixels, sorted by overlap
  - whether those pixels were already present in raw stringart branches
  - the dominant branch for a quick orientation/source check
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from skimage import io as skio

ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults to skip CLI arguments when running directly.
DEFAULT_RUN = ROOT / "output" / "full_pipeline" / "sem_full_00008_mask255_crop_20260507_105149"
DEFAULT_STEP = "final"   # "final" | "reconnect" | "postprocess"
DEFAULT_IDS = [25, 38, 39, 42, 68]


def _branch_index(path: Path) -> int:
    m = re.search(r"_branch_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def find_label_tif(step_dir: Path) -> Path:
    for pat in ["*_post_labels.tif", "*_reconnect_labels.tif", "*_labels.tif"]:
        hits = [p for p in step_dir.glob(pat) if "dilated" not in p.stem]
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"No label TIFF found in {step_dir}")


def step_dir(run: Path, step: str) -> Path:
    mapping = {
        "final": run / "final",
        "reconnect": run / "3.reconnect",
        "postprocess": run / "4.postprocess",
        "preprocess": run / "2.preprocess" / "branches",
    }
    d = mapping.get(step.lower())
    if d is None:
        raise ValueError(f"Unknown step '{step}'. Choose: final, reconnect, postprocess, preprocess.")
    return d


def branch_files(run: Path) -> list[Path]:
    d = run / "2.preprocess" / "branches"
    return sorted(d.glob("*_branch_*.png"), key=_branch_index)


def raw_branch_files(run: Path) -> list[Path]:
    d = run / "2.preprocess" / "raw_branches"
    if not d.exists():
        return []
    return sorted(d.glob("*_branch_*.png"), key=_branch_index)


def load_bin(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (img > 0) if img is not None else np.zeros((1, 1), dtype=bool)


def trace_id(lbl: np.ndarray, label_id: int, branches: list[Path], raw_branches: list[Path]) -> dict:
    mask = lbl == label_id
    if not mask.any():
        return {"id": label_id, "found": False}

    pts = np.argwhere(mask)
    r0, c0 = pts.min(0).tolist()
    r1, c1 = pts.max(0).tolist()

    branch_hits = []
    raw_map = {p.name: p for p in raw_branches}
    for bp in branches:
        b_bin = load_bin(bp)
        if b_bin.shape != mask.shape:
            continue
        ov = int(np.count_nonzero(mask & b_bin))
        if ov == 0:
            continue
        raw_path = raw_map.get(bp.name)
        raw_ov = 0
        if raw_path:
            r_bin = load_bin(raw_path)
            if r_bin.shape == mask.shape:
                raw_ov = int(np.count_nonzero(mask & r_bin))
        branch_hits.append({
            "branch": _branch_index(bp),
            "file": bp.name,
            "preprocess_px": ov,
            "stringart_px": raw_ov,
        })

    branch_hits.sort(key=lambda x: -x["preprocess_px"])
    dominant = branch_hits[0]["branch"] if branch_hits else None
    return {
        "id": label_id,
        "found": True,
        "area_px": int(mask.sum()),
        "bbox_rc": [r0, c0, r1, c1],
        "dominant_branch": dominant,
        "branches": branch_hits,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trace pipeline label IDs back to stringart/preprocess branches.")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="Pipeline run root folder.")
    ap.add_argument("--step", default=DEFAULT_STEP,
                    choices=["final", "reconnect", "postprocess", "preprocess"],
                    help="Which step's label TIFF to start from.")
    ap.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS, help="Label IDs to trace.")
    ap.add_argument("--quiet", action="store_true", help="Only print dominant branch per ID.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    sd = step_dir(run, args.step)
    lbl_path = find_label_tif(sd)
    lbl = skio.imread(str(lbl_path)).astype(np.int32)

    branches = branch_files(run)
    raw_branches = raw_branch_files(run)

    print(f"Run     : {run.name}")
    print(f"Step    : {args.step} ({lbl_path.name})")
    print(f"Branches: {len(branches)} preprocess | {len(raw_branches)} raw stringart")
    print()

    for label_id in args.ids:
        info = trace_id(lbl, label_id, branches, raw_branches)
        if not info["found"]:
            print(f"  ID {label_id}: NOT FOUND in {lbl_path.name}")
            continue
        r0, c0, r1, c1 = info["bbox_rc"]
        print(f"ID {label_id:4d}  area={info['area_px']:5d}px  bbox=r[{r0}:{r1}] c[{c0}:{c1}]"
              f"  dominant_branch={info['dominant_branch']}")
        if not args.quiet:
            for h in info["branches"]:
                bar = "#" * max(1, h["preprocess_px"] // 5)
                raw_note = f"  (stringart: {h['stringart_px']}px)" if h["stringart_px"] else "  (not in raw)"
                print(f"    branch_{h['branch']:02d}  pre={h['preprocess_px']:4d}px {bar}{raw_note}")
        print()


if __name__ == "__main__":
    main()
