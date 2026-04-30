# filaments_quantification — Repo Summary

## Purpose

Two-stage pipeline for quantifying filaments (e.g. CNTs) in microscopy images:

1. **stringart** — vectorize a binary filament mask into oriented line segments ("branches") using a tile-wise greedy Hough approach.
2. **reconnect** — stitch the resulting branch fragments back into whole filaments using geometry- and curvature-aware scoring.

---

## Repository layout

```
stringart/
  stringart_tiles.py          greedy Hough vectorizer (CONFIG dict at top)
  stringart_tiles_curve.py    variant with curve-fitting output

reconnect/
  reconnect_utils_v5.py       first Python port of the MATLAB reconnector
  reconnect_utils_v6.py       production evaluator (straight-line gates)
  reconnect_utils_v7.py       curvy-filament evaluator (monkey-patches v6)
  reconnect_config.yaml       default config (used by v5/v6)
  reconnect_config_v7.yaml    v7 config (two-stage, curvature-adaptive)
  reconnect_run.py            CLI entry point — --version v5/v6/v7
  input/                      per-image branch PNGs + merge TIFs
  output/                     label TIFs, previews, overlays
```

---

## Module responsibilities

### stringart_tiles.py
- Reads a binary mask, optionally skeletonizes it.
- Tiles the image (`TILE_SIZE`, default 128 px), runs `HoughLinesP` per tile per angle bin.
- Greedy acceptance: keeps lines that explain ≥ `MIN_ACCEPT_NEWPIX` new residual pixels.
- Outputs one PNG per line-bundle ("branch"), a merged PNG, and a `run_config.json`.
- Width-based autoscaling of Hough / acceptance thresholds.
- Optional experiment grid (`CONFIG["EXPERIMENT_GRID"]`) that runs parameter sweeps and prints a recall/precision/F1 table.

### reconnect_run.py
- Dynamically imports the requested utils version.
- Reads per-branch PNGs, binarizes, cleans, extracts components, skeletonizes.
- Optionally splits at branchpoints before reconnection.
- Supports **two-stage** reconnection when `stage1` / `stage2` keys are present in the config (strict first pass, relaxed second).
- Saves: raw label TIF, dilated label TIF, label previews, overlay PNG.
- `--compare` mode: compares extracted stats (filament count, arc length, angle, tortuosity) against a GT JSON and saves a 3-panel plot.

### reconnect_utils_v6.py
- Core reconnector: extracts tip geometry, traces along the skeleton, evaluates candidate tip-pairs with straight-line angular and residual gates, ranks by composite score, runs greedy merge passes.
- Key gate sequence: distance → forward cos → inward opposition → width ratio → **line residual** → curvature delta → Hermite bridge intrusion → smoothness RMS → turn angle.
- Arc prediction (`_arc_predicted_position`) was computed but came *after* all the hard gates, so curved pairs were already rejected before it ran.

### reconnect_utils_v7.py (changes vs v6)
See the next section.

### Debug script (reconnect/)

**[reconnect_debug.py](reconnect/reconnect_debug.py)** — single CLI with three subcommands:

```bash
# Print area, bbox, tips, axis angle for label ids
python reconnect_debug.py inspect v7/labels.tif "10,7,3"

# Find rejection-log rows by tip coordinates (tol defaults to 4 px)
python reconnect_debug.py grep-log output/reconnect_rejection_log.csv 178 132 175 126 5

# Print reason breakdown and override stats
python reconnect_debug.py log-summary output/reconnect_rejection_log.csv
```

---

## v7 vs v6 — what changed and why

### Root cause fixed by v7

In v6, slightly-curved filament gaps were rejected by `min_forward_cos` and `max_line_residual_px` before the arc-miss check ever ran. The arc prediction was a final cosmetic gate, not an early rescue mechanism.

### The seven changes

| # | Change | Effect |
|---|--------|--------|
| 1 | **Arc miss computed first** — before angular gates | Enables arc agreement to relax downstream gates |
| 2 | **Curvature-adaptive relaxation** — `curv_relax = min(max_curv_relax, mean_curv × curv_relax_factor)` lowers `min_forward_cos` and `min_inward_opposition` | Curved-tip pairs that are naturally misaligned from the gap vector survive |
| 3 | **OR-gate on line residual** — pair passes if `line_ok OR arc_ok` | The critical fix: curvy fragments that fail the straight-line residual pass via arc prediction |
| 4 | **Curvature-delta gated only when arc disagrees** | Two fragments with different local curvature still connect if their arcs land near each other |
| 5 | **Turn tolerance relaxed by `curv_relax × 40°`** | Hermite bridges over curved gaps inherently turn more |
| 6 | **`arc_miss` added as score term** (weight 0.8) | Ranker prefers pairs whose extrapolated arcs agree best |
| 7 | **Clear-merge override** — if `dist ≤ 15`, `line_resid ≤ 4 px`, `arc_miss ≤ 5 px` (or dist ≤ 2), and `opposition ≥ 0.85`, bypass `forward_cos`, `max_turn_deg`, and `bridge_intrusion` | Rescues true-positive merges killed by tip-direction noise on short rasterized skeleton traces |

### Implementation strategy

v7 monkey-patches `_v6._evaluate_tip_pair` inside `reconnect_components`, then restores it. The ~230-line greedy engine is not duplicated.

### New config keys in reconnect_config_v7.yaml

```yaml
thresholds:
  max_arc_miss_px: 15.0              # absolute arc-miss gate (px)
  clear_merge_max_dist_px: 15.0      # override window: max tip distance
  clear_merge_max_line_resid_px: 4.0 # override window: max line residual
  clear_merge_max_arc_miss_px: 5.0   # override window: max arc miss
  clear_merge_min_opposition: 0.85   # override window: min tip-face opposition
advanced:
  curv_relax_factor: 30.0      # curvature × factor → cosine slack
  max_curv_relax: 0.30         # cap on curvature-driven angular slack
weights:
  arc_miss: 0.8                # score penalty per unit arc_miss_norm
debug:
  rejection_log_path: 'output/reconnect_rejection_log.csv'  # null to disable
```

### Rejection log

When `debug.rejection_log_path` is set, every pair evaluation is written to a CSV with columns: `stage, base_id, tar_id, base_tip, tar_tip, base/tar tip coords, reason, dist, forward_base/tar, inward_opposition, width_ratio, line_resid, arc_miss_px, curv_delta, intrusion_frac, smooth_rms, max_turn_deg, eff_forward_cos, eff_opposition, eff_max_turn, mean_curv, curv_relax, arc_ok, line_ok, clear_merge`.

Use `_grep_log.py` to look up why a specific pair was rejected by its tip coordinates.

Two-stage config: stage1 is strict (tight thresholds, conservative relaxation), stage2 is relaxed (generous arc tolerance, aggressive curvature relaxation).

### Does v7 actually perform better?

**Yes, for curvy filament datasets — by design.** The fix is architecturally sound:
- Curved pairs that v6 silently discarded in the first two gates are now evaluated fully.
- The OR-gate prevents false rejections caused by the straight-line residual on genuinely curved gaps.
- The arc-miss score term means the ranker doesn't just accept more — it ranks curvy matches by geometric quality.

**Caveat:** No quantitative benchmark is checked into this repo. The `--compare` flag in `reconnect_run.py` will compute recall/precision against a GT JSON if you have one. For straight filaments, v6 and v7 should produce nearly identical results (curvature ≈ 0 → `curv_relax` ≈ 0 → gates identical).

---

## How to run

```bash
# Vectorize a mask into branches
python stringart/stringart_tiles.py          # edit CONFIG dict at top

# Reconnect with v7 (two-stage, curvy-aware)
cd reconnect
python reconnect_run.py \
    --version v7 \
    --config reconnect_config_v7.yaml \
    --input input/<image_folder> \
    --output output/<image_folder>

# Compare against GT JSON
python reconnect_run.py --version v7 --config reconnect_config_v7.yaml \
    --input input/<image_folder> --output output/<image_folder> --compare
```

## Environment

```bash
conda env create -f mtquant.yml
conda activate mtquant
```
