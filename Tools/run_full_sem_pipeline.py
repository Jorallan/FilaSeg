from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
from skimage import io as skio
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]

# ══════════════════════════════════════════════════════════════════════════
# User configuration
# Priority: CLI flag > these defaults > each stage script's own config block
# ══════════════════════════════════════════════════════════════════════════

# ── Inputs / outputs ──────────────────────────────────────────────────────
DEFAULT_MASK = ROOT / "input" / "SEM08" / "crops" / "sem_full_00008_mask255_crop.png"
DEFAULT_BG   = ROOT / "input" / "SEM08" / "crops" / "sem_full_00008_overlay_crop.png"
DEFAULT_OUT  = ROOT / "output" / "full_pipeline"

# ── Reconnect ─────────────────────────────────────────────────────────────
# "straight" = standard evaluator  |  "curvy" = arc-aware (recommended for CNT)
DEFAULT_RECONNECT_VERSION = "straight"

# ── Preprocess (stage 2) ──────────────────────────────────────────────────
DEFAULT_PRE_BIN_THRESHOLD    = 127  # grayscale threshold for binarising branch masks
DEFAULT_PRE_LINE_CLOSE_LEN   = 4    # oriented closing kernel length (px)
DEFAULT_PRE_LINE_CLOSE_ITERS = 2    # closing iterations
DEFAULT_PRE_MIN_COMPONENT    = 4    # drop connected components smaller than this (px)

# ── Postprocess (stage 4) ─────────────────────────────────────────────────
DEFAULT_THICKEN_PX         = 5    # px to thicken each final bundle centerline
DEFAULT_SMOOTH_WINDOW      = 8    # moving-window size for centerline smoothing
DEFAULT_MIN_KEEP_LEN       = 12   # drop bundles whose skeleton is shorter than this (px)
DEFAULT_ABSORB_LEN         = 30   # absorb bundles shorter than this into a longer neighbour
DEFAULT_ABSORB_RADIUS      = 6    # halo radius (px) for neighbour detection during absorb
DEFAULT_BRIDGE_RADIUS      = 4    # re-join same-source split pieces within this radius
DEFAULT_OVERLAY_ALPHA      = 0.72 # overlay blend (0 = background only, 1 = labels only)

# ── DEM JSON ──────────────────────────────────────────────────────────────
DEFAULT_POLYLINE_STEP = 1   # keep every Nth centerline point in DEM JSON (1 = all)

# ══════════════════════════════════════════════════════════════════════════

STAGE_STRINGART  = "1.stringart"
STAGE_PREPROCESS = "2.preprocess"
STAGE_RECONNECT  = "3.reconnect"
STAGE_POSTPROCESS = "4.postprocess"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SEM mask -> stringart -> preprocess -> reconnect -> postprocess.")
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--background", type=Path, default=DEFAULT_BG)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base", default=None, help="Output base name. Defaults to mask stem.")
    ap.add_argument("--python", default=sys.executable)
    # reconnect
    ap.add_argument("--reconnect-version", default=DEFAULT_RECONNECT_VERSION, choices=["straight", "curvy"])
    # preprocess
    ap.add_argument("--pre-bin-threshold",    type=int,   default=DEFAULT_PRE_BIN_THRESHOLD)
    ap.add_argument("--pre-line-close-len",   type=int,   default=DEFAULT_PRE_LINE_CLOSE_LEN)
    ap.add_argument("--pre-line-close-iters", type=int,   default=DEFAULT_PRE_LINE_CLOSE_ITERS)
    ap.add_argument("--pre-min-component",    type=int,   default=DEFAULT_PRE_MIN_COMPONENT)
    # postprocess
    ap.add_argument("--thicken-px",      type=int,   default=DEFAULT_THICKEN_PX)
    ap.add_argument("--smooth-window",   type=int,   default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--min-keep-len",    type=int,   default=DEFAULT_MIN_KEEP_LEN)
    ap.add_argument("--absorb-len",      type=int,   default=DEFAULT_ABSORB_LEN)
    ap.add_argument("--absorb-radius",   type=int,   default=DEFAULT_ABSORB_RADIUS)
    ap.add_argument("--bridge-radius",   type=int,   default=DEFAULT_BRIDGE_RADIUS)
    ap.add_argument("--overlay-alpha",   type=float, default=DEFAULT_OVERLAY_ALPHA)
    # DEM JSON
    ap.add_argument("--polyline-step",   type=int,   default=DEFAULT_POLYLINE_STEP)
    return ap.parse_args()


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("\n[run]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def unique_run_dir(root: Path, base: str) -> Path:
    run_dir = root / base
    if not run_dir.exists():
        return run_dir
    return root / f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"


def find_one(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No {pattern!r} found in {folder}")
    return hits[0]


def resolve_from_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    nb = convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    return [tuple(int(v) for v in p) for p in np.argwhere((skel > 0) & (nb == 1))]


def neighbors8(p: tuple[int, int], skel: np.ndarray):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                yield rr, cc


def shortest_path(skel: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    q = deque([start])
    parent = {start: None}
    while q:
        p = q.popleft()
        if p == goal:
            break
        for n in neighbors8(p, skel):
            if n not in parent:
                parent[n] = p
                q.append(n)
    if goal not in parent:
        return np.argwhere(skel > 0).astype(np.float32)
    out = []
    p = goal
    while p is not None:
        out.append(p)
        p = parent[p]
    return np.asarray(out[::-1], dtype=np.float32)


def centerline(mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(mask)
    pts = np.argwhere(sk > 0)
    if len(pts) < 2:
        return pts.astype(np.float32)
    eps = endpoints(sk)
    if len(eps) >= 2:
        a, b = max(
            ((a, b) for i, a in enumerate(eps) for b in eps[i + 1 :]),
            key=lambda ab: (ab[0][0] - ab[1][0]) ** 2 + (ab[0][1] - ab[1][1]) ** 2,
        )
        return shortest_path(sk, a, b)
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    xyc = xy - xy.mean(axis=0, keepdims=True)
    axis = np.linalg.eigh(np.cov(xyc.T))[1][:, 1]
    return pts[np.argsort(xyc @ axis)].astype(np.float32)


def path_length(path_rc: np.ndarray) -> float:
    if len(path_rc) < 2:
        return 0.0
    d = np.diff(path_rc.astype(np.float32), axis=0)
    return float(np.sqrt(np.sum(d * d, axis=1)).sum())


def bundle_json(labels_path: Path, out_json: Path, *, args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    lbl = skio.imread(str(labels_path)).astype(np.int32)
    bundles = []
    step = max(1, int(args.polyline_step))
    for cid in [int(v) for v in np.unique(lbl) if v]:
        mask = lbl == cid
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        ctr = pts.mean(axis=0)
        line = centerline(mask)
        if len(line) > 2 and step > 1:
            line = np.vstack([line[::step], line[-1]])
        length = path_length(line)
        endpoints_rc = [line[0].tolist(), line[-1].tolist()] if len(line) else []
        bundles.append({
            "id": cid,
            "area_px": int(mask.sum()),
            "bbox_rc": [int(r0), int(r1), int(c0), int(c1)],
            "bbox_xyxy": [int(c0), int(r0), int(c1), int(r1)],
            "centroid_rc": [float(ctr[0]), float(ctr[1])],
            "centroid_xy": [float(ctr[1]), float(ctr[0])],
            "length_px": length,
            "mean_width_px": float(mask.sum() / max(1.0, length)),
            "endpoints_rc": endpoints_rc,
            "endpoints_xy": [[float(p[1]), float(p[0])] for p in endpoints_rc],
            "centerline_rc": [[float(r), float(c)] for r, c in line.tolist()],
            "centerline_xy": [[float(c), float(r)] for r, c in line.tolist()],
        })
    data = {
        "schema": "filaseg.dem_bundles.v1",
        "units": "pixels",
        "coordinate_system": {
            "origin": "top_left",
            "rc": "[row, col]",
            "xy": "[x=col, y=row]",
        },
        "image_shape_rc": [int(lbl.shape[0]), int(lbl.shape[1])],
        "bundle_count": len(bundles),
        "source": {k: str(v) for k, v in paths.items()},
        "parameters": {
            "reconnect_version": args.reconnect_version,
            "thicken_px": int(args.thicken_px),
            "smooth_window": int(args.smooth_window),
            "polyline_step": step,
        },
        "bundles": bundles,
    }
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main() -> None:
    args = parse_args()
    args.mask = resolve_from_root(args.mask)
    args.background = resolve_from_root(args.background)
    args.output_root = resolve_from_root(args.output_root)
    base = args.base or args.mask.stem
    run_dir = unique_run_dir(args.output_root, base)
    stringart_run = run_dir / STAGE_STRINGART
    pre_branches = run_dir / STAGE_PREPROCESS / "branches"
    reconnect_out = run_dir / STAGE_RECONNECT
    post_out = run_dir / STAGE_POSTPROCESS
    final_out = run_dir / "final"
    final_out.mkdir(parents=True, exist_ok=True)

    run([
        args.python, ROOT / "1.stringart" / "stringart_tiles.py",
        "--input", args.mask,
        "--output-root", run_dir,
        "--output-folder-name", STAGE_STRINGART,
    ])
    run([
        args.python, ROOT / "2.preprocess" / "preprocess_stringart_branches.py",
        "--input", stringart_run / "branches",
        "--output", pre_branches,
        "--run-config", stringart_run / "run_config.json",
        "--copy-source",
        "--bin-threshold",    str(args.pre_bin_threshold),
        "--line-close-len",   str(args.pre_line_close_len),
        "--line-close-iters", str(args.pre_line_close_iters),
        "--min-component-area", str(args.pre_min_component),
    ])
    run([
        args.python, "reconnect_run.py",
        "--version", args.reconnect_version,
        "--config", ROOT / "3.reconnect" / "reconnect_config.yaml",
        "--input", pre_branches,
        "--output", reconnect_out,
        "--background", args.background,
        "--no_show",
    ], cwd=ROOT / "3.reconnect")
    run([
        args.python, ROOT / "4.postprocess" / "post_process_reconnect.py",
        "--input", reconnect_out,
        "--output", post_out,
        "--background", args.background,
        "--thicken-px",                 str(args.thicken_px),
        "--smooth-window",              str(args.smooth_window),
        "--min-keep-len",               str(args.min_keep_len),
        "--absorb-len",                 str(args.absorb_len),
        "--absorb-radius",              str(args.absorb_radius),
        "--same-source-bridge-radius",  str(args.bridge_radius),
        "--overlay-alpha",              str(args.overlay_alpha),
    ])

    post_labels = find_one(post_out, "*_post_labels.tif")
    post_color = find_one(post_out, "*_post_labels_preview.png")
    post_overlay = find_one(post_out, "*_post_overlay.png")
    labels_final = final_out / f"{base}_reconnect_labels.tif"
    color_final = final_out / f"{base}_instances_color.png"
    overlay_final = final_out / f"{base}_reconnect_overlay.png"
    shutil.copy2(post_labels, labels_final)
    shutil.copy2(post_color, color_final)
    shutil.copy2(post_overlay, overlay_final)
    shutil.copy2(args.background, final_out / f"{base}_original{args.background.suffix}")

    paths = {
        "mask": args.mask,
        "background": args.background,
        "stringart": stringart_run,
        "preprocess_branches": pre_branches,
        "reconnect": reconnect_out,
        "postprocess": post_out,
        "final_labels": labels_final,
        "final_overlay": overlay_final,
        "instances_color": color_final,
    }
    dem = bundle_json(labels_final, final_out / f"{base}_bundles_dem.json", args=args, paths=paths)
    manifest = {
        "base": base,
        "run_dir": str(run_dir),
        "final_dir": str(final_out),
        "interactive_command": (
            f"python 3.reconnect\\reconnect_interactive.py --input {final_out} "
            f"--base {base} --label full_pipeline"
        ),
        "bundle_count": int(dem["bundle_count"]),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    (final_out / f"{base}_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n[OK] final:", final_out)
    print("[OK] interactive:", manifest["interactive_command"])
    print("[OK] DEM JSON:", final_out / f"{base}_bundles_dem.json")


if __name__ == "__main__":
    main()
