# FilaSeg

Tools for turning filament masks from microscopy images into reconnectable bundle instances.

The current CNT workflow starts from a binary mask and a greyscale/overlay image, vectorizes the mask with `1.stringart`, cleans the branch set, reconnects branch fragments, then exports smoothed/thickened bundle instances plus a DEM-ready JSON file.

## Repository Layout

```text
Tools/
  crop_pair_interactive.py      Interactive paired cropper for mask + overlay images.
  run_full_sem_pipeline.py      End-to-end SEM mask -> bundles runner.
  mask_edit.py                  Legacy/manual mask editor utility.

1.stringart/
  stringart_tiles.py            Tile-wise greedy vectorizer.
  stringart_tiles_curve.py      Curve-oriented variant.

2.preprocess/
  preprocess_stringart_branches.py
                                Cleans stringart branch images before reconnect.

3.reconnect/
  reconnect_run.py              CLI entry point, supports --version straight/curvy.
  reconnect_interactive.py      Interactive viewer/tuner for reconnect outputs.
  reconnect_debug.py            Inspection and rejection-log helper.
  reconnect_utils_straight.py   Standard straight-line evaluator.
  reconnect_utils_curvy.py      Arc-aware evaluator for curved filaments (CNT default).
  reconnect_config.yaml         Unified config; [curvy] section holds curvy-only keys.

4.postprocess/
  post_process_reconnect.py     Smooths/thickens reconnect labels and exports previews.
```

Generated pipeline artifacts are written under `output/full_pipeline/` and are ignored by git.

## End-To-End Pipeline

Run from the repo root:

```powershell
python Tools\run_full_sem_pipeline.py `
  --mask 1.stringart\input\SEM05\crops\sem_full_00006_mask255_crop.png `
  --background 1.stringart\input\SEM05\crops\sem_full_00006_overlay_crop.png
```

For the local CNT virtualenv used during development:

```powershell
C:\Repos\venv_cnt\Scripts\python.exe Tools\run_full_sem_pipeline.py `
  --mask 1.stringart\input\SEM05\crops\sem_full_00006_mask255_crop.png `
  --background 1.stringart\input\SEM05\crops\sem_full_00006_overlay_crop.png
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

The final folder uses the same naming convention as reconnect outputs, so it can be opened with:

```powershell
python 3.reconnect\reconnect_interactive.py `
  --input output\full_pipeline\<base>\final `
  --base <base> `
  --label full_pipeline
```

## DEM JSON

`<base>_bundles_dem.json` uses schema `filaseg.dem_bundles.v1`. Coordinates are stored in pixels with origin at the top-left. Each bundle includes id, area, bounding box, centroid, estimated length, mean width, endpoints, and centerline in both `rc` (`row, col`) and `xy` (`x=col, y=row`) forms.

## Preprocess And Postprocess

`2.preprocess/preprocess_stringart_branches.py` runs after stringart and before reconnect. It thresholds each per-angle branch image, removes tiny connected components, applies an angle-matched morphological close to bridge small gaps inside each branch, removes tiny components again, and writes a cleaned branch set plus a merged preview and JSON summary. It uses `run_config.json` from stringart to know the angle of each branch.

`4.postprocess/post_process_reconnect.py` runs after reconnect. It splits disconnected pieces that share a label, drops very short fragments, absorbs short nearby pieces into longer neighboring bundles, skeletonizes each kept bundle, extracts its dominant path, smooths that path with a moving window, redraws the bundle as a slightly thicker polyline, and writes final labels, color preview, overlay, and a summary. The full runner then copies these into `final` and builds the DEM JSON.

## Stringart Acceptance

The default stringart config is conservative again. A candidate line must now satisfy both gates: enough new residual pixels and enough support density on the original mask. The density gate rejects long fake chords that touch a few CNT pixels but mostly cross black background.

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

The active stage scripts need the scientific Python stack used in `venv_cnt`: `numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, and `pyyaml`.
