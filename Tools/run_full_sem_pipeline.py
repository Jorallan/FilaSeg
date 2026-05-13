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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_helper import (  # noqa: E402  (import after sys.path tweak)
    resolve_um_per_px,
    scale_pipeline_args,
    write_scaled_reconnect_yaml,
)

# ══════════════════════════════════════════════════════════════════════════
# User configuration
# Priority: CLI flag > these defaults > each stage script's own config block
# ══════════════════════════════════════════════════════════════════════════

# ── Inputs / outputs ──────────────────────────────────────────────────────
_INPUT_FOLDER = "sem_full_00000_1p66_crop512"                    # ← change this to switch dataset
DEFAULT_MASK  = ROOT / "input" / _INPUT_FOLDER / "mask.png"
DEFAULT_BG    = ROOT / "input" / _INPUT_FOLDER / "sem.png"
DEFAULT_OUT   = ROOT / "output" / "full_pipeline"

# ── Physical scale (auto-scales pixel parameters across magnifications) ───
# Reference: SEM08 = 1.66 µm HFW / 1536 px = 0.001081 µm/px (FEI Helios tag).
# Pixel-distance defaults below AND in reconnect_config.yaml are tuned at this
# scale. When the input image has a different µm/px, every distance/area/
# curvature parameter is multiplied at runtime by sf = um_per_px / _REF_UM_PER_PX.
# Defaults are NEVER modified — a scaled YAML copy is written to the run dir.
# To disable scaling: set _USE_SCALING = False or pass --no-scale on the CLI.
_REF_UM_PER_PX     = 1.66 / 1536            # SEM08 reference; do not change
_USE_SCALING       = True                    # master on/off switch
DEFAULT_UM_PER_PX  = _REF_UM_PER_PX          # change per dataset, or pass --um-per-px

# ── Stringart (stage 1) ───────────────────────────────────────────────────
# Tile-grid voting: None/1 = single grid; integer N = generated offsets in
# stringart_tiles.py; JSON list = explicit [oy,ox] origins.
DEFAULT_TILE_SIZE       = 128
DEFAULT_ANGLE_STEP_DEG  = 15
DEFAULT_TILE_GRID_OFFSETS  = 4
DEFAULT_TILE_GRID_VOTE_MIN = 2

# ── Preprocess (stage 2) ──────────────────────────────────────────────────
DEFAULT_PRE_BIN_THRESHOLD    = 127  # grayscale threshold for binarising branch masks
DEFAULT_PRE_LINE_CLOSE_LEN   = 4    # oriented closing kernel length (px)
DEFAULT_PRE_LINE_CLOSE_ITERS = 2    # closing iterations
DEFAULT_PRE_MIN_COMPONENT    = 10    # drop connected components smaller than this (px)
DEFAULT_PRE_CLEAN_TO_PATH    = True # reduce multi-tip components to dominant 2-tip path
DEFAULT_PRE_CLEAN_SMOOTH_WIN = 4    # smoothing window for the dominant path
DEFAULT_PRE_TARGET_WIDTH_PX  = 0    # 0 = use per-branch median width; >0 = uniform width across all components
DEFAULT_PRE_FIT_DEGREE       = 2    # 0=skip spline fit; 1/2/3=parametric B-spline degree (forces physically meaningful curve)
DEFAULT_PRE_FIT_SMOOTHING    = 1.5  # spline smoothing factor multiplier (higher = stiffer fit)

# ── Reconnect (stage 3) ───────────────────────────────────────────────────
DEFAULT_RECONNECT_VERSION   = "straight"
# Reconnect overlap handling (modifies the generated YAML `overlap` block):
#   mode = "trim"  → keep all viable sub-components after removing overlap.
#          "kill"  → drop the entire smaller component (legacy).
#   kill_thr        triggers above-threshold overlap action (0.3-0.7).
#   trim_dilate_px  optional halo: temporarily dilate the larger component
#                   during the overlap test. 0 disables it.
DEFAULT_RECO_OVERLAP_MODE     = "trim"
DEFAULT_RECO_OVERLAP_KILL_THR = 0.3
DEFAULT_RECO_TRIM_DILATE_PX   = 0


# ── Postprocess (stage 4) ─────────────────────────────────────────────────
DEFAULT_THICKEN_PX         = 8    # px to thicken each final bundle centerline
DEFAULT_SMOOTH_WINDOW      = 8    # moving-window size for centerline smoothing
DEFAULT_MIN_KEEP_LEN       = 20   # drop bundles whose skeleton is shorter than this (px)
DEFAULT_OVERLAY_ALPHA      = 0.72 # overlay blend (0 = background only, 1 = labels only)
DEFAULT_OVERLAP_ABSORB_THR = 0.6  # postprocess overlap: absorb near-duplicate IDs into the larger
DEFAULT_OCCLUSION_TRIM_THR = 0.25 # postprocess overlap: trim lower-priority layers hidden by earlier layers
DEFAULT_OCCLUSION_TRIM_MIN_PX = 50 # hidden rendered pixels required before occlusion trim triggers
DEFAULT_TIP_TRIM_FRAC      = 0.10  # postprocess overlap: trim pixels near an ID's skeleton tip

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
    ap.add_argument(
        "--reconnect-version",
        default=DEFAULT_RECONNECT_VERSION,
        choices=["straight"],
        help="Compatibility option; only the straight reconnect evaluator is supported.",
    )
    ap.add_argument("--reco-overlap-mode", type=str, default=DEFAULT_RECO_OVERLAP_MODE,
                    choices=["kill", "trim"],
                    help='Reconnect overlap-handling mode. "trim" keeps viable sub-components of '
                         'the smaller after removing overlap; "kill" drops the smaller entirely.')
    ap.add_argument("--reco-overlap-kill-thr", type=float, default=DEFAULT_RECO_OVERLAP_KILL_THR,
                    help="Fraction-of-overlap above which the small component is killed/trimmed.")
    ap.add_argument("--reco-trim-dilate-px", type=int, default=DEFAULT_RECO_TRIM_DILATE_PX,
                    help="Temporary dilation (px) of the larger component during the trim test; not persisted.")
    # preprocess
    ap.add_argument("--pre-bin-threshold",    type=int,   default=DEFAULT_PRE_BIN_THRESHOLD)
    ap.add_argument("--pre-line-close-len",   type=int,   default=DEFAULT_PRE_LINE_CLOSE_LEN)
    ap.add_argument("--pre-line-close-iters", type=int,   default=DEFAULT_PRE_LINE_CLOSE_ITERS)
    ap.add_argument("--pre-min-component",    type=int,   default=DEFAULT_PRE_MIN_COMPONENT)
    ap.add_argument("--pre-clean-to-path",   action=argparse.BooleanOptionalAction, default=DEFAULT_PRE_CLEAN_TO_PATH)
    ap.add_argument("--pre-clean-smooth-win",type=int,   default=DEFAULT_PRE_CLEAN_SMOOTH_WIN)
    ap.add_argument("--pre-target-width-px", type=int,   default=DEFAULT_PRE_TARGET_WIDTH_PX,
                    help="Width to re-render every preprocess component at. 0=per-branch median.")
    ap.add_argument("--pre-fit-degree", type=int, default=DEFAULT_PRE_FIT_DEGREE,
                    help="Polynomial/B-spline degree for skeleton smoothing in preprocess (0=skip).")
    ap.add_argument("--pre-fit-smoothing", type=float, default=DEFAULT_PRE_FIT_SMOOTHING,
                    help="Spline smoothing factor multiplier in preprocess.")
    # postprocess
    ap.add_argument("--thicken-px",      type=int,   default=DEFAULT_THICKEN_PX)
    ap.add_argument("--smooth-window",   type=int,   default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--min-keep-len",    type=int,   default=DEFAULT_MIN_KEEP_LEN)
    ap.add_argument("--overlay-alpha",   type=float, default=DEFAULT_OVERLAY_ALPHA)
    ap.add_argument("--overlap-absorb-thr", type=float, default=DEFAULT_OVERLAP_ABSORB_THR,
                    help="Postprocess: absorb near-duplicate IDs whose intersection covers >= this "
                         "fraction of the smaller. 0 disables. Try 0.6-0.7.")
    ap.add_argument("--occlusion-trim-thr", type=float, default=DEFAULT_OCCLUSION_TRIM_THR,
                    help="Postprocess: trim lower-priority layers whose rendered pixels are mostly "
                         "already covered. 0 disables.")
    ap.add_argument("--occlusion-trim-min-px", type=int, default=DEFAULT_OCCLUSION_TRIM_MIN_PX,
                    help="Postprocess: minimum hidden rendered pixels before occlusion trimming triggers.")
    ap.add_argument("--tip-trim-frac", type=float, default=DEFAULT_TIP_TRIM_FRAC,
                    help="Postprocess: erase overlap pixels at an ID's skeleton tip when overlap is "
                         "tip-dominant. 0 disables. Try 0.15.")
    # DEM JSON
    ap.add_argument("--polyline-step",   type=int,   default=DEFAULT_POLYLINE_STEP)
    # physical-scale auto-scaling
    ap.add_argument("--um-per-px", type=float, default=None,
                    help="µm per pixel of the input image. Overrides DEFAULT_UM_PER_PX "
                         "and any filename-based detection.")
    ap.add_argument("--no-scale", action="store_true",
                    help="Disable pixel-parameter scaling; use raw defaults regardless of µm/px.")
    # stringart multi-grid voting (passed through to stage 1)
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    ap.add_argument("--angle-step-deg", type=int, default=DEFAULT_ANGLE_STEP_DEG)
    ap.add_argument("--tile-grid-offsets", type=str, default=DEFAULT_TILE_GRID_OFFSETS,
                    help='Tile-grid offset count (1,2,3,4,...) or JSON list of [oy,ox] origins, '
                         'e.g. "4" or "[[0,0],[64,0],[0,64],[64,64]]". Omit for single-grid.')
    ap.add_argument("--tile-grid-vote-min", type=int, default=DEFAULT_TILE_GRID_VOTE_MIN,
                    help="Min grids a pixel must appear in to be kept (1=OR, 2+=majority).")
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
    base = args.base or args.mask.parent.name
    run_dir = unique_run_dir(args.output_root, base)
    stringart_run = run_dir / STAGE_STRINGART
    pre_branches = run_dir / STAGE_PREPROCESS / "branches"
    reconnect_out = run_dir / STAGE_RECONNECT
    post_out = run_dir / STAGE_POSTPROCESS
    final_out = run_dir / "final"
    final_out.mkdir(parents=True, exist_ok=True)

    # ── Resolve physical scale and (optionally) scale all pixel parameters ──
    if _USE_SCALING and not args.no_scale:
        um_per_px, scale_source = resolve_um_per_px(
            args.um_per_px, args.mask, DEFAULT_UM_PER_PX,
        )
        scale_factor = um_per_px / _REF_UM_PER_PX if _REF_UM_PER_PX else 1.0
    else:
        um_per_px, scale_source, scale_factor = _REF_UM_PER_PX, "scaling-disabled", 1.0

    if abs(scale_factor - 1.0) >= 1e-3:
        print(f"[scale] um_per_px={um_per_px:.6f}  source={scale_source}  sf={scale_factor:.4f}×")
        if not (0.1 <= scale_factor <= 10.0):
            print(f"[scale] WARNING: scale factor {scale_factor:.3f} is outside expected 0.1–10× range")
        scale_pipeline_args(args, scale_factor)
        reconnect_cfg = run_dir / "reconnect_config_scaled.yaml"
        write_scaled_reconnect_yaml(
            ROOT / "3.reconnect" / "reconnect_config.yaml", reconnect_cfg, scale_factor,
        )
    else:
        print(f"[scale] no scaling applied  (source={scale_source}, um_per_px={um_per_px:.6f})")
        reconnect_cfg = ROOT / "3.reconnect" / "reconnect_config.yaml"

    # Apply CLI overrides to the reconnect YAML overlap block. If we had not
    # already produced a scaled copy, copy the original into the run dir first
    # so we never mutate the source YAML.
    import yaml as _yaml
    if reconnect_cfg == ROOT / "3.reconnect" / "reconnect_config.yaml":
        reconnect_cfg = run_dir / "reconnect_config_active.yaml"
        reconnect_cfg.write_text(
            (ROOT / "3.reconnect" / "reconnect_config.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    _cfg = _yaml.safe_load(reconnect_cfg.read_text(encoding="utf-8"))
    _overlap = _cfg.setdefault("overlap", {})
    _overlap["mode"] = args.reco_overlap_mode
    _overlap["kill_thr"] = float(args.reco_overlap_kill_thr)
    _overlap["trim_dilate_px"] = int(args.reco_trim_dilate_px)
    _cm = _cfg.setdefault("stage_clear", {}).setdefault("thresholds", {})
    _cm["clear_merge_backward_max_layer_gap"] = max(1, int(round(4 * DEFAULT_ANGLE_STEP_DEG / max(1, args.angle_step_deg))))
    reconnect_cfg.write_text(_yaml.dump(_cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")

    run([
        args.python, ROOT / "1.stringart" / "stringart_tiles.py",
        "--input", args.mask,
        "--output-root", run_dir,
        "--output-folder-name", STAGE_STRINGART,
        "--tile-size", str(args.tile_size),
        "--angle-step-deg", str(args.angle_step_deg),
        *(["--tile-grid-offsets", args.tile_grid_offsets] if args.tile_grid_offsets else []),
        "--tile-grid-vote-min", str(args.tile_grid_vote_min),
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
        *(["--clean-to-path"] if args.pre_clean_to_path else ["--no-clean-to-path"]),
        "--clean-smooth-win", str(args.pre_clean_smooth_win),
        "--target-width-px", str(args.pre_target_width_px),
        "--fit-degree", str(args.pre_fit_degree),
        "--fit-smoothing", str(args.pre_fit_smoothing),
    ])
    run([
        args.python, "reconnect_run.py",
        "--version", args.reconnect_version,
        "--config", reconnect_cfg,
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
        "--overlay-alpha",              str(args.overlay_alpha),
        "--overlap-absorb-thr",         str(args.overlap_absorb_thr),
        "--occlusion-trim-thr",         str(args.occlusion_trim_thr),
        "--occlusion-trim-min-px",      str(args.occlusion_trim_min_px),
        "--tip-trim-frac",              str(args.tip_trim_frac),
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
    for p in post_out.glob("*_post_multilabel*"):
        shutil.copy2(p, final_out / p.name.replace("_post_", "_reconnect_"))
    for p in post_out.glob("*_post_overlap.png"):
        shutil.copy2(p, final_out / p.name.replace("_post_", "_reconnect_"))
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
        "interactive_command": f"python Tools\\visualize_ids.py --input {final_out}",
        "bundle_count": int(dem["bundle_count"]),
        "scale": {
            "ref_um_per_px": _REF_UM_PER_PX,
            "input_um_per_px": um_per_px,
            "scale_factor": scale_factor,
            "source": scale_source,
            "reconnect_config_used": str(reconnect_cfg),
        },
        "paths": {k: str(v) for k, v in paths.items()},
    }
    (final_out / f"{base}_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n[OK] final:", final_out)
    print("[OK] interactive:", manifest["interactive_command"])
    print("[OK] DEM JSON:", final_out / f"{base}_bundles_dem.json")


if __name__ == "__main__":
    main()
