"""Physical-scale resolution + pixel-parameter scaling for run_full_sem_pipeline.py.

The reference values in run_full_sem_pipeline.py and reconnect_config.yaml are
tuned for SEM08 (1.66 µm HFW / 1536 px = 0.001081 µm/px). When the input image
has a different µm/px, every distance-based parameter is rescaled so the
thresholds gate the same *physical* filament dimensions.

Conventions
-----------
    sf = input_um_per_px / ref_um_per_px     (>1 means lower magnification:
                                              each pixel covers more µm, so
                                              physical features take fewer px)

    linear / step-count parameters  →  scaled = ref / sf
    area parameters (px²)           →  scaled = ref / sf²
    curvature (1/radius_px)         →  scaled = ref × sf

The original config files are never modified — a scaled copy of
reconnect_config.yaml is written to the per-run output directory.
"""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

# Reconnect YAML keys that are linear pixel distances or pixel-step counts.
PX_LINEAR_KEYS = frozenset({
    # thresholds
    "search_size_px", "max_tip_distance_px", "max_line_residual_px",
    "max_smooth_rms_px", "clear_merge_max_dist_px",
    "clear_merge_max_line_resid_px", "min_tip_trace_len_px",
    # morphology
    "dilate_px", "bp_dilate_px", "tip_dir_steps",
    # reconnect geometry (skeleton step counts proportional to pixel distance)
    "trace_steps", "fit_points", "smooth_trace_points", "bridge_samples",
    "bridge_clearance_px",
})
# Pixel-area keys: scale by sf².
PX_AREA_KEYS = frozenset({"clean_min_area", "min_component_area", "min_keep_area"})
# Curvature keys (1/radius_px): scale by 1/sf.
PX_CURV_KEYS = frozenset({"max_curvature_delta"})

# Filename FOV patterns.
#   '_1p66_', '20p7micron', '1p66.tif'                → matching <int>p<int> µm
#   '_1.66_um_', '_3.45um.', '_20.7 um'               → matching <float> µm
# Anchor on start-of-string or a non-digit/non-letter boundary so the leading
# segment does not need a literal underscore (works for '20p7micron' too).
_RE_FOV_P  = re.compile(r"(?:^|[_\s.\-])(\d+)p(\d+)(?:[_.\s\-]|micron|um|$)", re.IGNORECASE)
_RE_FOV_UM = re.compile(r"(?:^|[_\s.\-])(\d+(?:\.\d+)?)\s*_?um", re.IGNORECASE)


def parse_fov_um_from_name(stem: str) -> float | None:
    """Extract field-of-view (µm) from a filename stem. Returns None if not found."""
    for rx, builder in (
        (_RE_FOV_UM, lambda m: float(m.group(1))),
        (_RE_FOV_P,  lambda m: float(f"{m.group(1)}.{m.group(2)}")),
    ):
        m = rx.search(stem)
        if m:
            try:
                v = builder(m)
            except ValueError:
                continue
            if 0.01 <= v <= 10000.0:
                return v
    return None


def read_png_width(path: Path) -> int | None:
    """Return PNG width without loading pixel data. Returns None if unreadable."""
    try:
        with open(path, "rb") as f:
            if f.read(8)[:4] != b"\x89PNG":
                return None
            f.read(4)   # IHDR length
            f.read(4)   # "IHDR"
            return struct.unpack(">I", f.read(4))[0]
    except Exception:
        return None


_CROP_MARKERS = ("crop", "_cropped", "_subset", "_roi", "_tile_")


def _looks_cropped(path: Path) -> bool:
    """Heuristic: filename or folder name suggests this is a derived crop, so
    the embedded FOV is the original's, not this image's. Skip filename FOV
    detection in that case to avoid wrong µm/px."""
    name = (mask_path := path).stem.lower() + " " + mask_path.parent.name.lower()
    return any(marker in name for marker in _CROP_MARKERS)


def resolve_um_per_px(
    cli_um_per_px: float | None,
    mask_path: Path,
    default_um_per_px: float,
) -> tuple[float, str]:
    """Determine µm/px and report which source supplied it.

    Priority:
      1. CLI override --um-per-px
      2. Filename '_NpN_' or '_N.NN_um_' (in mask name OR parent folder name)
         combined with PNG width of the mask -- skipped when the path has a
         crop/subset marker, since for crops the FOV in the name refers to the
         ORIGINAL image, not the cropped pixels.
      3. default_um_per_px
    """
    if cli_um_per_px is not None:
        return float(cli_um_per_px), "cli"
    if _looks_cropped(mask_path):
        return float(default_um_per_px), "default(crop-marker)"
    fov = parse_fov_um_from_name(mask_path.stem)
    if fov is None:
        fov = parse_fov_um_from_name(mask_path.parent.name)
    if fov is not None:
        w = read_png_width(mask_path)
        if w:
            return fov / w, f"filename(fov={fov}µm,w={w}px)"
    return float(default_um_per_px), "default"


def scale_pipeline_args(args: argparse.Namespace, sf: float) -> None:
    """Scale CLI args (preprocess + postprocess pixel parameters) in place.

    sf = input_um_per_px / ref_um_per_px. A *physical* distance D maps to
    pixel count = D / um_per_px, so at higher µm/px (sf > 1) the same physical
    distance occupies fewer pixels: scaled = ref / sf. Areas use 1/sf**2.
    """
    if abs(sf - 1.0) < 1e-3:
        return
    def _lin(v: int) -> int:  return max(1, int(round(v / sf)))
    def _area(v: int) -> int: return max(1, int(round(v / (sf * sf))))
    args.pre_line_close_len   = _lin(args.pre_line_close_len)
    args.pre_clean_smooth_win = _lin(args.pre_clean_smooth_win)
    args.pre_min_component    = _area(args.pre_min_component)
    args.thicken_px           = _lin(args.thicken_px)
    args.smooth_window        = _lin(args.smooth_window)
    args.min_keep_len         = _lin(args.min_keep_len)
    args.smart_width_search_px = _lin(args.smart_width_search_px)
    args.smart_width_min_px    = _lin(args.smart_width_min_px)
    args.smart_width_max_px    = _lin(args.smart_width_max_px)
    args.absorb_len           = _lin(args.absorb_len)
    args.absorb_radius        = _lin(args.absorb_radius)
    args.bridge_radius        = _lin(args.bridge_radius)
    # NOT scaled: pre_bin_threshold (intensity), pre_line_close_iters (count),
    # overlay_alpha (fraction).


def _scale_yaml_dict(d: dict, sf: float) -> dict:
    """Scale reconnect YAML dict in place.

    Linear pixel distances and step counts are physical → divide by sf.
    Areas → divide by sf**2. Curvature (1/radius_px) → multiply by sf.
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _scale_yaml_dict(v, sf)
        elif k in PX_LINEAR_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, int):
                out[k] = max(1, int(round(v / sf)))
            else:
                out[k] = round(v / sf, 4)
        elif k in PX_AREA_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, int):
                out[k] = max(1, int(round(v / (sf * sf))))
            else:
                out[k] = round(v / (sf * sf), 4)
        elif k in PX_CURV_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v) * sf, 4)
        else:
            out[k] = v
    return out


def write_scaled_reconnect_yaml(src: Path, dest: Path, sf: float) -> None:
    """Read reconnect_config.yaml, scale pixel-keyed entries, write to dest."""
    import yaml
    cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    if abs(sf - 1.0) >= 1e-3:
        cfg = _scale_yaml_dict(cfg, sf)
    Path(dest).write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
