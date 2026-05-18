# filaments_quantification - Repo Summary

## Purpose

This repository is being shaped into a CNT bundle extraction pipeline:

1. Start with a binary SEM/UNet mask and a greyscale or overlay image.
2. Use `1.stringart` to convert the mask into branch-like primitives.
3. Use `2.preprocess` to remove obvious branch artifacts before reconnect.
4. Use staged `3.reconnect` to merge branch fragments into candidate bundles.
5. Use `4.postprocess` to smooth and slightly thicken bundle labels.
6. Export overlays, colored instance labels, and a DEM-oriented JSON bundle file.

## Current Layout

```text
Tools/
  crop_pair_interactive.py      Matplotlib paired cropper for aligned mask/overlay images.
  run_full_sem_pipeline.py      End-to-end runner for the current SEM bundle workflow.
  scale_helper.py               Physical-scale resolution + pixel-parameter auto-scaling.
  troubleshoot_reconnect.py     Consolidated reconnect-debug CLI.
  visualize_ids.py              Interactive viewer for label IDs in reconnect/final outputs.
  mask_edit.py                  Legacy OpenCV mask editing utility.

1.stringart/
  skeleton_decompose.py         Experimental/WIP skeleton-first orientation decomposition.
  stringart_tiles.py            Older tile-wise greedy Hough vectorizer.

2.preprocess/
  preprocess_stringart_branches.py
                                Branch cleanup between stringart and reconnect.

3.reconnect/
  reconnect_run.py              Main reconnect CLI.
  reconnect_debug.py            Inspection and rejection-log helper.
  reconnect_utils_straight.py   Standard straight-line evaluator.
  reconnect_config.yaml         Staged reconnect config.

4.postprocess/
  post_process_reconnect.py     Smooths/thickens reconnect labels and writes previews.

output/
  full_pipeline/                Generated runs, ignored by git.
```

## Reconnect Notes

The reconnect stage uses the straight evaluator with a staged config. The important behavior is:

- Clear short-gap merge handling for visually obvious reconnections.
- A narrow clear-merge overrun allowance for long, collinear two-tip paths whose endpoints slightly overshot each other.
- Strict then relaxed tip reconnect passes for progressively broader candidate gaps.
- Same-layer residual relaxation for fragments in the same orientation bin.
- The full runner rescales `stage_clear.clear_merge_backward_max_layer_gap` with branch count, using the YAML value as the `--angle-step-deg 15` baseline.
- Relabeling lets longer surviving trunks claim overlaps first.
- The evaluator can write candidate rejection/acceptance metrics to `debug.rejection_log_path`.
- Shared tip tracing and bridge sampling settings live under `geometry`; true pass gates live under each stage.

### Staged Reconnect Pipeline

The straight evaluator now runs **three ordered stages** (safest → broadest), each loaded from the config:

| Stage key | Name | Role |
|---|---|---|
| `stage_clear` | `clear_merge` | Tiny, straight, high-confidence endpoint gaps only |
| `stage_strict` | `strict_tip_reconnect` | Short-gap reconnect with full forward/opposition/residual/smoothness gates |
| `stage_relaxed` | `relaxed_tip_reconnect` | Wider-gap reconnect with relaxed gates |

Within `stage_strict` and `stage_relaxed`, a **same-layer relaxation** path applies when two components share the same orientation bin: `same_layer_max_line_resid_px`, `same_layer_min_forward_cos`, `same_layer_min_opposition` override the global residual gate to allow same-bin splits that would otherwise be blocked by a high residual score.

### Overlap Handling

Reconnect resolves heavy component overlaps with settings grouped under the `overlap` config block:

- `mode: kill` (legacy): drop the entire smaller component when overlap exceeds `kill_thr`.
- `mode: trim` (recommended): erase only the pixels that overlap the larger component, then re-CC the remainder. Every sub-component with area ≥ `min_keep_area` survives as its own fresh Component and can participate in downstream tip-bridging. Sub-components below the threshold are dropped.

`overlap.trim_dilate_px > 0` adds a temporary halo to the larger component during the overlap test only. The halo can make near-adjacent components qualify for trim/kill scoring, but the actual trim still subtracts only the larger component's real pixels. The halo is not persisted. Direct reconnect YAML currently sets this to `10`; the full runner overrides it from `--reco-trim-dilate-px`, whose default is `0`.

### Layered Multi-Label Output

After each reconnect run `reconnect_run.py` writes:

```text
<base>_reconnect_multilabel.npz   sparse per-ID pixel index arrays
<base>_reconnect_multilabel.tif   one page per ID (BigTIFF)
<base>_reconnect_multilabel_ids.json  page → ID mapping
<base>_reconnect_overlap.png      pixels covered by ≥ 2 IDs
```

## Stage 1 Options

The full runner currently defaults to `skeleton_decompose.py`, which skeletonizes the full mask and bins centerline pixels by local tangent orientation. Two algorithms keep smoothly-curving filaments together when the angle bins would otherwise fragment them:

- **Adaptive binning**: per-pixel binning runs first. If a single skeleton piece's pixels span only a few cyclically-adjacent bins (a smooth curve crossing one bin boundary), the entire piece is unified to its dominant bin. Pieces that span many non-adjacent bins (genuine sharp turns) keep the per-pixel split. Controlled by `ADAPTIVE_BIN_MAX_SPAN` (default `2`) and `ADAPTIVE_BIN_MIN_FRAC` (default `0.1`).
- **Smart junction merge**: at each 8-connected junction blob, the tangents of adjacent skeleton arms (walked outward `SMART_JUNCTION_WALK_STEPS` pixels) are compared. A pair is fused if both arms have at least `SMART_JUNCTION_MIN_ARM_PX` pixels and their tangents are anti-parallel with `cos < SMART_JUNCTION_COS_THR`. The whole junction blob is added to the merged piece as a bridge so its connected component physically spans both arms. `SMART_JUNCTION_HANDLE_X` extends the same rule to 4-arm X-crossings.

`run_config.json` records the CONFIG used per run and the resulting counts: `n_smart_junction_merges`, `n_through_junction_pixels`, `n_adaptive_unified_pieces`. Both algorithms are conservative by default; they will not fuse a 90-degree T-arm into a through-bundle. Pass `--no-skeleton-decompose` to use `stringart_tiles.py` instead.

## Stringart Acceptance

`1.stringart/stringart_tiles.py` is the older tiled Hough stage and uses conservative Hough defaults plus a mask-support density gate:

The goal is to reject fake long chords over black background. A line can still bridge tiny raster gaps, but most of the proposed line must lie on the original mask. `--min-accept-density`, `--residual-dilate-kernel`, and `--residual-dilate-iters` are exposed as CLI overrides.

### Multi-Grid Voting

The pipeline can run stringart at multiple tile-grid origins and keep only pixels that appear in at least `--tile-grid-vote-min` of them. Direct `stringart_tiles.py` runs default to a single grid; when the full runner uses `stringart_tiles.py`, it currently passes `4` generated offsets with vote minimum `2`. Pass an integer count (`1`, `2`, `3`, `4`, etc.) to generate offsets from the tile size; `4` restores the previous half-tile four-grid pattern. You can still pass an explicit JSON list of `[oy,ox]` pairs.

`stringart_tiles.py` also scales `MAX_LINES_PER_TILE` and `MAX_CANDIDATES_TO_TRY` by `(tile_size / 128)^2`, keeping those two workload limits proportional to tile area.

## Physical-Scale Auto-Scaling

All pixel-distance defaults in `run_full_sem_pipeline.py` and `reconnect_config.yaml` are tuned for **SEM08** (1.66 µm HFW / 1536 px = 0.001081 µm/px). `Tools/scale_helper.py` handles cross-magnification adaptation:

- `resolve_um_per_px()` tries (in order): `--um-per-px` CLI flag → filename FOV patterns (`_1p66_`, `20p7micron`, `3.45um`) → `DEFAULT_UM_PER_PX`.
- `scale_pipeline_args()` applies `sf = um_per_px / ref_um_per_px` to every distance/area/curvature arg: linear parameters scale as `ref / sf`, area parameters as `ref / sf²`, curvature as `ref × sf`.
- `write_scaled_reconnect_yaml()` writes a per-run scaled copy of `reconnect_config.yaml` without ever modifying the source.
- Pass `--no-scale` to disable all scaling and use raw defaults regardless of µm/px.

The runner always writes a per-run config copy (`reconnect_config_scaled.yaml` or `reconnect_config_active.yaml`) so the source YAML is never mutated even when only CLI overlap overrides are applied.

## End-To-End Runner

Use `Tools/run_full_sem_pipeline.py` from the repo root:

```powershell
python Tools\run_full_sem_pipeline.py `
  --mask input\sem_full_00000_1p66_crop512\mask.png `
  --background input\sem_full_00000_1p66_crop512\sem.png
```

Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--reconnect-version` | `straight` | Compatibility flag; straight evaluator only |
| `--reco-overlap-mode` | `trim` | `trim` or `kill` overlap handling in reconnect |
| `--reco-overlap-kill-thr` | `0.3` | Fraction threshold triggering kill/trim |
| `--reco-trim-dilate-px` | `0` | Halo radius (px) for the trim test |
| `--um-per-px` | auto | µm/px for scale-factor computation |
| `--no-scale` | off | Disable pixel-parameter scaling |
| `--skeleton-decompose` | on | Use skeleton decomposition for stage 1; pass `--no-skeleton-decompose` for tiled Hough stringart |
| `--smart-junction-merge` | on | Fuse anti-parallel arms at junction blobs in stage 1 |
| `--smart-junction-cos-thr` | `-0.90` | Cosine threshold for collinear-arm fusion (more negative = stricter) |
| `--smart-junction-walk-steps` | `10` | Pixels walked outward to estimate each arm's tangent |
| `--smart-junction-min-arm-px` | `25` | Minimum length each arm must have for fusion |
| `--smart-junction-handle-x` | on | Also process 4-arm X-crossings (pair-by-pair) |
| `--adaptive-bin-max-span` | `2` | Max adjacent bins a single piece may span before staying per-pixel |
| `--adaptive-bin-min-frac` | `0.1` | Fraction-of-dominant threshold for counting a bin as significant |
| `--angle-step-deg` | `15` | Stringart orientation-bin width; also sets branch count used to scale clear-merge backward layer gap |
| `--tile-grid-offsets` | `4` | Tiled stringart grid-origin count or explicit origins |
| `--tile-grid-vote-min` | `2` | Min tiled-stringart grids a pixel must appear in |
| `--smart-width` | on | Render postprocess IDs from SEM edge samples; those masks also drive absorb/trim cleanup |
| `--pre-fit-degree` | `2` | B-spline degree for preprocess skeleton smoothing (0=skip) |
| `--pre-fit-smoothing` | `1.5` | Spline smoothing factor multiplier |
| `--overlap-absorb-thr` | `0.6` | Post-thickening near-duplicate merge threshold |
| `--occlusion-trim-thr` | `0.25` | Post-thickening render-layer occlusion trim |
| `--occlusion-trim-min-px` | `50` | Minimum hidden rendered pixels before occlusion trim |

The runner creates:

```text
output\full_pipeline\<base>\1.stringart\
output\full_pipeline\<base>\2.preprocess\
output\full_pipeline\<base>\3.reconnect\
output\full_pipeline\<base>\4.postprocess\
output\full_pipeline\<base>\final\
output\full_pipeline\<base>\reconnect_config_scaled.yaml  (or _active.yaml)
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
- Optionally fits a parametric B-spline to the dominant skeleton path (`--fit-degree`, `--fit-smoothing`) to force a physically meaningful curve before handing off to reconnect.
- Writes cleaned branch PNGs, a merged branch mask, copied width metadata, and `pre_process_summary.json`.

The intent is conservative cleanup: keep the stringart branch identities, remove obvious specks, and make fragmented line masks less brittle before reconnect scoring.

## Postprocess

`4.postprocess/post_process_reconnect.py` prepares reconnect labels for the final handoff:

- Reads layered reconnect IDs when `*_reconnect_multilabel.npz` exists, else falls back to the flat `*_reconnect_labels.tif` image.
- Drops very short pieces below `--min-keep-len`.
- Skeletonizes each kept bundle and extracts a dominant centerline.
- Smooths that centerline with `--smooth-window`.
- Renders each kept path at one SEM-guided width per ID by sampling opposite-slope edge pairs normal to the path (`--smart-width`, default on). If edge evidence is too weak, it falls back to `--thicken-px`.
- **Overlap absorb** (`--overlap-absorb-thr`, default 0.6): on those smart-width masks, pairs whose intersection covers ≥ this fraction of the smaller mask are merged into the larger. Catches near-duplicate bundles that survive reconnect.
- **Occlusion trim** (`--occlusion-trim-thr`, `--occlusion-trim-min-px`): also on the smart-width masks, trims lower-priority rendered layers whose pixels are mostly already covered by earlier layers, keeps only substantial visible fragments, and hands off detached fragments that already overlap another rendered layer.
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
python Tools\troubleshoot_reconnect.py trace-component `
  --run output\full_pipeline\<base> `
  --step final `
  --ids 25 38 42
```

Reconnect troubleshooting now lives in one CLI instead of four one-off scripts:

```powershell
python Tools\troubleshoot_reconnect.py compare-followups --old-run <old> --new-run <new> --pairs 31,14 13,43
python Tools\troubleshoot_reconnect.py diagnose-missed --run <base> --pairs 31,14 13,43
python Tools\troubleshoot_reconnect.py check-coords --run <base> --coords "pairA|120,240|130,248"
python Tools\troubleshoot_reconnect.py trace-evolution --old-run <old> --new-run <new> --ids 8 9 11
python Tools\troubleshoot_reconnect.py visualize-pairs --run <base> --pairs 31,14 13,43
python Tools\troubleshoot_reconnect.py trace-component --run output\full_pipeline\<base> --step final --ids 25 38 42
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

The current workflow has been run with `C:\Repos\venv_cnt\Scripts\python.exe` and depends on `numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, `pyyaml`, and `tifffile`.
