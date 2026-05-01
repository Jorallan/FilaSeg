from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# ── User configuration ────────────────────────────────────────────────────
# Only relevant when running this script directly (not via run_full_sem_pipeline.py).
DEFAULT_INPUT  = ROOT / "1.stringart" / "output" / "sem_full_00000_1p66_mask255_crop" / "branches"
DEFAULT_OUTPUT = ROOT / "2.preprocess" / "output" / "sem_full_00000_1p66_mask255_crop" / "branches"

DEFAULT_BIN_THRESHOLD    = 127  # grayscale threshold for binarising each branch mask
DEFAULT_LINE_CLOSE_LEN   = 9    # length of the oriented closing kernel (px); bridges small raster gaps
DEFAULT_LINE_CLOSE_ITERS = 1    # how many times to apply the closing
DEFAULT_MIN_COMPONENT_AREA = 8  # remove connected components smaller than this (px)
# ─────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean stringart branch masks before reconnect.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--run-config", type=Path, default=None, help="Defaults to ../run_config.json from input.")
    ap.add_argument("--bin-threshold", type=int, default=DEFAULT_BIN_THRESHOLD)
    ap.add_argument("--line-close-len", type=int, default=DEFAULT_LINE_CLOSE_LEN)
    ap.add_argument("--line-close-iters", type=int, default=DEFAULT_LINE_CLOSE_ITERS)
    ap.add_argument("--min-component-area", type=int, default=DEFAULT_MIN_COMPONENT_AREA)
    ap.add_argument("--copy-source", action="store_true", help="Copy raw branch inputs beside output.")
    return ap.parse_args()


def bool255(img: np.ndarray) -> np.ndarray:
    return (img > 0).astype(np.uint8) * 255


def read_bin(path: Path, thr: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return bool255(cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)[1])


def branch_index(path: Path) -> int | None:
    m = re.search(r"_branch_(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def branch_paths(folder: Path) -> list[Path]:
    paths = []
    for p in folder.glob("*_branch_*.png"):
        if branch_index(p) is not None:
            paths.append(p)
    return sorted(paths, key=lambda p: branch_index(p) or 0)


def load_angle_centers(input_dir: Path, run_config: Path | None, n: int) -> list[float]:
    cfg_path = run_config or (input_dir.parent / "run_config.json")
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        bins = data.get("angle_bins", [])
        if len(bins) >= n:
            return [0.5 * (float(lo) + float(hi)) for lo, hi in bins[:n]]
        step = float(data.get("CONFIG", {}).get("ANGLE_STEP_DEG", 180.0 / max(1, n)))
    else:
        step = 180.0 / max(1, n)
    return [(i + 0.5) * step for i in range(n)]


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(float(angle_deg))
    dx = np.cos(rad) * c
    dy = np.sin(rad) * c
    p0 = (int(round(c - dx)), int(round(c - dy)))
    p1 = (int(round(c + dx)), int(round(c + dy)))
    cv2.line(k, p0, p1, 1, 1, cv2.LINE_8)
    return k


def remove_small_components(mask255: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 1:
        return mask255
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask255 > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask255)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 255
    return out


def process_branch(mask255: np.ndarray, angle_deg: float, args: argparse.Namespace) -> np.ndarray:
    out = remove_small_components(mask255, args.min_component_area)
    if args.line_close_len > 1 and args.line_close_iters > 0:
        ker = line_kernel(args.line_close_len, angle_deg)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=int(args.line_close_iters))
    out = remove_small_components(out, args.min_component_area)
    return bool255(out)


def base_stem_from_paths(paths: list[Path]) -> str:
    stem = paths[0].stem
    return re.sub(r"_branch_\d+$", "", stem)


def main() -> None:
    args = parse_args()
    paths = branch_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No *_branch_*.png files found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.copy_source:
        raw = args.output.parent / "raw_branches"
        if raw.exists():
            shutil.rmtree(raw)
        shutil.copytree(args.input, raw)

    centers = load_angle_centers(args.input, args.run_config, len(paths))
    merge = None
    per_branch = []
    for p, angle in zip(paths, centers):
        cleaned = process_branch(read_bin(p, args.bin_threshold), angle, args)
        out_path = args.output / p.name
        cv2.imwrite(str(out_path), cleaned)
        merge = cleaned if merge is None else cv2.bitwise_or(merge, cleaned)
        per_branch.append({
            "file": p.name,
            "angle_center_deg": angle,
            "input_px": int(cv2.countNonZero(read_bin(p, args.bin_threshold))),
            "output_px": int(cv2.countNonZero(cleaned)),
        })

    base = base_stem_from_paths(paths)
    cv2.imwrite(str(args.output / f"{base}_branches_merge.png"), merge)
    for name in ("filament_width.json",):
        src = args.input / name
        if src.exists():
            shutil.copy2(src, args.output / name)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "branches": len(paths),
        "line_close_len": int(args.line_close_len),
        "line_close_iters": int(args.line_close_iters),
        "min_component_area": int(args.min_component_area),
        "input_merge_px": int(sum(x["input_px"] for x in per_branch)),
        "output_merge_px": int(cv2.countNonZero(merge)),
        "per_branch": per_branch,
    }
    (args.output.parent / "pre_process_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_branch"}, indent=2))


if __name__ == "__main__":
    main()
