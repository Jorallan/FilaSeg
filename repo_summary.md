# filaments_quantification - Repo Summary

## Purpose

This repository is being shaped into a CNT bundle extraction pipeline:

1. Start with a binary SEM/UNet mask and a greyscale or overlay image.
2. Use `1.stringart` to convert the mask into branch-like primitives.
3. Use `2.preprocess` to remove obvious branch artifacts before reconnect.
4. Use `3.reconnect` v7 to merge branch fragments into candidate bundles.
5. Use `4.postprocess` to smooth and slightly thicken bundle labels.
6. Export overlays, colored instance labels, and a DEM-oriented JSON bundle file.

## Current Layout

```text
Tools/
  crop_pair_interactive.py      Matplotlib paired cropper for aligned mask/overlay images.
  run_full_sem_pipeline.py      End-to-end runner for the current SEM bundle workflow.
  visualize_ids.py             Interactive viewer for label IDs in reconnect/final outputs.
  trace_component.py           Label-ID provenance tool for branch origins.
  mask_edit.py                  Legacy OpenCV mask editing utility.

1.stringart/
  stringart_tiles.py            Primary line-based vectorizer.
  stringart_tiles_curve.py      Curve primitive experiment.

2.preprocess/
  preprocess_stringart_branches.py
                                Branch cleanup between stringart and reconnect.

3.reconnect/
  reconnect_run.py              Main reconnect CLI.
  reconnect_debug.py            Inspection and rejection-log helper.
  reconnect_utils_straight.py   Standard straight-line evaluator.
  reconnect_utils_curvy.py      Arc-aware evaluator for curved filaments (CNT default).
  reconnect_config.yaml         Unified config; [curvy] section holds curvy-only keys.

4.postprocess/
  post_process_reconnect.py     Smooths/thickens reconnect labels and writes previews.

output/
  full_pipeline/                Generated runs, ignored by git.
```

## Reconnect Notes

`reconnect_utils_curvy` extends the straight evaluator for CNT work. The important behavior is:

- Curvature-aware candidate rescue so curved fragments are not rejected only because they fail a straight-line residual gate.
- Clear short-gap merge handling for visually obvious reconnections.
- Same-direction absorb handling for tiny fragments that lie on top of a longer trunk.
- A sharper turn cap for clear merges so obvious raster noise does not create severe kinked bundles.
- Relabeling lets longer surviving trunks claim overlaps first.
- The straight evaluator can write candidate rejection/acceptance metrics to `debug.rejection_log_path`.

Curvy-only config keys live in the `[curvy]` section of `3.reconnect/reconnect_config.yaml`, including `clear_merge_max_turn_deg`, `same_dir_absorb_max_dist_px`, `same_dir_absorb_max_line_resid_px`, `same_dir_absorb_max_arc_miss_px`, and `same_dir_absorb_min_parallel`.

## Stringart Acceptance

`1.stringart/stringart_tiles.py` uses conservative Hough defaults plus a mask-support density gate:

```text
MIN_ACCEPT_NEWPIX      6
MIN_ACCEPT_DENSITY     0.45
HOUGH_THRESHOLD        18
HOUGH_MIN_LINE_LENGTH  4
HOUGH_MAX_LINE_GAP     5
RESIDUAL_DILATE_KERNEL 2
```

The goal is to reject fake long chords over black background. A line can still bridge tiny raster gaps, but most of the proposed line must lie on the original mask. `--min-accept-density`, `--residual-dilate-kernel`, and `--residual-dilate-iters` are exposed as CLI overrides.

## End-To-End Runner

Use `Tools/run_full_sem_pipeline.py` from the repo root:

```powershell
python Tools\run_full_sem_pipeline.py `
  --mask 1.stringart\input\SEM05\crops\sem_full_00006_mask255_crop.png `
  --background 1.stringart\input\SEM05\crops\sem_full_00006_overlay_crop.png
```

The runner creates:

```text
output\full_pipeline\<base>\1.stringart\
output\full_pipeline\<base>\2.preprocess\
output\full_pipeline\<base>\3.reconnect\
output\full_pipeline\<base>\4.postprocess\
output\full_pipeline\<base>\final\
```

The `final` folder is the handoff folder. It contains the final overlay, final labels, colored instance preview, pipeline manifest, copied background image, and `<base>_bundles_dem.json`.

## Preprocess

`2.preprocess/preprocess_stringart_branches.py` prepares stringart branch masks for reconnect:

- Reads `*_branch_*.png` files from the stringart `branches` folder.
- Uses stringart `run_config.json` to recover the angle center for each branch.
- Thresholds every branch to a clean binary mask.
- Removes small connected components.
- Applies an oriented morphological close along that branch angle to close small raster gaps.
- Optionally reduces multi-tip components to the dominant smoothed two-tip skeleton path.
- Writes cleaned branch PNGs, a merged branch mask, copied width metadata, and `pre_process_summary.json`.

The intent is conservative cleanup: keep the stringart branch identities, remove obvious specks, and make fragmented line masks less brittle before reconnect scoring.

## Postprocess

`4.postprocess/post_process_reconnect.py` prepares reconnect labels for the final handoff:

- Reads the raw `*_reconnect_labels.tif` instance image.
- Splits disconnected components that accidentally share one label.
- Drops very short pieces below `--min-keep-len`.
- Absorbs short nearby pieces into longer neighboring bundles using an `--absorb-radius` halo.
- Skeletonizes each kept bundle and extracts a dominant centerline.
- Smooths that centerline with `--smooth-window`.
- Redraws it as a slightly thicker label using `--thicken-px`.
- Writes post labels, color preview, overlay, and `post_process_summary.json`.

The intent is to make each bundle look like one cleaner physical filament instance before final overlay and DEM JSON export.

## Interactive Review And Tracing

Final outputs can be opened with the label-ID viewer:

```powershell
python Tools\visualize_ids.py --input output\full_pipeline\<base>\final
```

For direct script use, `Tools/visualize_ids.py` also supports `--ask-input` or editing `SCRIPT_INPUT`.

Label IDs can be traced back to preprocess/stringart branch origins:

```powershell
python Tools\trace_component.py `
  --run output\full_pipeline\<base> `
  --step final `
  --ids 25 38 42
```

Recent runs:

```text
output\full_pipeline\sem_full_00008_mask255_crop\final
output\full_pipeline\sem_full_00006_mask255_crop\final
```

## DEM JSON Contract

The DEM export uses schema `filaseg.dem_bundles.v1`.

Top-level fields:

```text
schema
units
coordinate_system
image_shape_rc
bundle_count
source
parameters
bundles
```

Per-bundle fields:

```text
id
area_px
bbox_rc
bbox_xyxy
centroid_rc
centroid_xy
length_px
mean_width_px
endpoints_rc
endpoints_xy
centerline_rc
centerline_xy
```

Coordinates are pixel coordinates. `rc` means `[row, col]`; `xy` means `[x=col, y=row]`.

## Environment

The current workflow has been run with `C:\Repos\venv_cnt\Scripts\python.exe` and depends on `numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, and `pyyaml`.
