# FilaSeg

Tools for turning filament masks from microscopy images into reconnectable bundle instances.

The current CNT workflow starts from a binary mask and a greyscale/overlay image, vectorizes the mask with `1.stringart`, cleans the branch set, reconnects branch fragments, then exports smoothed/thickened bundle instances plus a DEM-ready JSON file.

## Repository Layout

```text
Tools/
  crop_pair_interactive.py      Interactive paired cropper for mask + overlay images.
  run_full_sem_pipeline.py      End-to-end SEM mask -> bundles runner.
  visualize_ids.py             Interactive label-ID viewer for reconnect/final outputs.
  trace_component.py           Trace label IDs back to preprocess/stringart branches.
  mask_edit.py                  Legacy/manual mask editor utility.

1.stringart/
  stringart_tiles.py            Tile-wise greedy vectorizer.

2.preprocess/
  preprocess_stringart_branches.py
                                Cleans stringart branch images before reconnect.

3.reconnect/
  reconnect_run.py              CLI entry point for staged reconnect.
  reconnect_debug.py            Inspection and rejection-log helper.
  reconnect_utils_straight.py   Standard straight-line evaluator.
  reconnect_config.yaml         Staged reconnect config.

4.postprocess/
  post_process_reconnect.py     Smooths/thickens reconnect labels and exports previews.
```

Generated pipeline artifacts are written under `output/full_pipeline/` and are ignored by git.

## End-To-End Pipeline

Run from the repo root:

```powershell
python Tools\run_full_sem_pipeline.py `
  --mask input\sem_full_00000_1p66_crop512\mask.png `
  --background input\sem_full_00000_1p66_crop512\sem.png
```

For the local CNT virtualenv used during development:

```powershell
C:\Repos\venv_cnt\Scripts\python.exe Tools\run_full_sem_pipeline.py `
  --mask input\sem_full_00000_1p66_crop512\mask.png `
  --background input\sem_full_00000_1p66_crop512\sem.png
```

The final outputs live in:

```text
output\full_pipeline\<base>\final\
```

The intermediate stage folders are:

```text
output\full_pipeline\<base>\1.stringart\
output\full_pipeline\<base>\2.preprocess\
output\full_pipeline\<base>\3.reconnect\
output\full_pipeline\<base>\4.postprocess\
```

Expected final files:

```text
<base>_reconnect_overlay.png       Overlay of final bundles on the source image.
<base>_reconnect_labels.tif        Integer instance labels for each bundle.
<base>_instances_color.png         Colored bundle-instance preview.
<base>_bundles_dem.json            DEM-oriented bundle geometry export.
<base>_pipeline_manifest.json      Paths, parameters, and interactive command.
<base>_original.png                Copied source background image.
```

## Interactive Review

The final folder uses the same naming convention as reconnect outputs, so it can be opened with the label-ID viewer:

```powershell
python Tools\visualize_ids.py --input output\full_pipeline\<base>\final
```

For direct IDE/script use, edit `SCRIPT_INPUT` in `Tools/visualize_ids.py`, or run `python Tools\visualize_ids.py --ask-input` to type the folder path at startup.

To trace final or intermediate label IDs back to their preprocess/stringart branch origins:

```powershell
python Tools\trace_component.py `
  --run output\full_pipeline\<base> `
  --step final `
  --ids 25 38 42
```

## DEM JSON

`<base>_bundles_dem.json` uses schema `filaseg.dem_bundles.v1`. Coordinates are stored in pixels with origin at the top-left. Each bundle includes id, area, bounding box, centroid, estimated length, mean width, endpoints, and centerline in both `rc` (`row, col`) and `xy` (`x=col, y=row`) forms.

## Preprocess And Postprocess

`2.preprocess/preprocess_stringart_branches.py` runs after stringart and before reconnect. It thresholds each per-angle branch image, removes tiny connected components, applies an angle-matched morphological close to bridge small gaps inside each branch, optionally reduces multi-tip components to their dominant smoothed two-tip skeleton path, and writes a cleaned branch set plus a merged preview and JSON summary. It uses `run_config.json` from stringart to know the angle of each branch. The full runner exposes this as `--pre-clean-to-path` / `--no-pre-clean-to-path` and `--pre-clean-smooth-win`.

`4.postprocess/post_process_reconnect.py` runs after reconnect. It splits disconnected pieces that share a label, drops very short fragments, absorbs short nearby pieces into longer neighboring bundles, skeletonizes each kept bundle, extracts its dominant path, smooths that path with a moving window, redraws the bundle as a slightly thicker polyline, and writes final labels, color preview, overlay, and a summary. The full runner then copies these into `final` and builds the DEM JSON.

Reconnect rejection logging can be enabled through the `debug.rejection_log_path` key in `3.reconnect/reconnect_config.yaml`; the straight evaluator writes candidate accept/reject metrics to that CSV path.

## Stringart Acceptance

The default stringart config is conservative again. A candidate line must now satisfy both gates: enough new residual pixels and enough support density on the original mask. The density gate rejects long fake chords that touch a few CNT pixels but mostly cross black background.

Tile-grid voting can be enabled with `--tile-grid-offsets`: omit it or pass `1` for a single grid, pass `4` for the previous half-tile four-grid pattern, or pass an explicit JSON list of `[oy,ox]` origins.

```powershell
python 1.stringart\stringart_tiles.py `
  --input path\to\mask.png `
  --output-root output\scratch `
  --output-folder-name 1.stringart `
  --hough-threshold 18 `
  --hough-min-line-length 4 `
  --hough-max-line-gap 5 `
  --min-accept-newpix 6 `
  --min-accept-density 0.45
```

## Environment Notes

The active stage scripts need the scientific Python stack used in `venv_cnt`: `numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, `pyyaml`, and `tifffile`.
