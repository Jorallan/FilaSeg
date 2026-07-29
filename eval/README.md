# FilaSeg evaluation (`eval/`)

Measures whether the pipeline's instance grouping matches ground truth. This is
the measurement foundation for fixing reconnect quality: without it, parameter
changes can't be judged objectively.

## What it measures

The **primary** metric is **fragment clustering**. After stage 1+2 the image is
a set of atomic fragments (the per-angle preprocess pieces). Ground truth
assigns each fragment to a true filament; reconnect assigns each to an output
id. Comparing those two partitions of the *same* fragment set answers "did the
right filaments emerge", and it is:

- **permutation invariant** — immune to id renumbering (so it ignores the
  cosmetic "inconsistent IDs" churn from `relabel_components` sorting);
- **decomposable** — pairwise **precision ↓ = over-merge** (false joins),
  pairwise **recall ↓ = under-merge** (missed joins);
- **cheap on GT** — needs grouping truth, not pixel-perfect masks.

Reported: pairwise P/R/F1, Adjusted Rand Index, Variation of Information split
into over-seg (`voi_split`) and under-seg (`voi_merge`) directions, and
plain-language split/merge counts.

A **secondary** pixel **Panoptic Quality** (PQ/SQ/RQ + F1@IoU) is reported as a
cross-check. It is harsh on 1-px-wide filaments, so it is not the objective.

## Files

| File | Role |
|---|---|
| `instance_io.py` | Load predicted instances (`*_multilabel.npz` CSR, or `*_labels.tif`), GT (label image, overlap-aware `gt_multilabel.npz`, or centerline JSON), and fragments (per-angle `2.preprocess/branches`). |
| `metrics.py` | Fragment-clustering metrics, panoptic quality, top-level `evaluate()`. |
| `eval_reconnect.py` | Eval CLI + synthetic self-test. |
| `synth_generator.py` | Scale-aware synthetic filament generator with realistic UNet degradation. |
| `run_batch.py` | Generate→run pipeline→eval→aggregate over many samples to CSV. |
| `diagnose_connections.py` | Root-cause attribution of each missed / wrong connection. |

## Usage

```powershell
$PY = "C:\Repos\venv_cnt\Scripts\python.exe"

# Validate the metric engine itself (no data needed):
& $PY eval\eval_reconnect.py --selftest

# Describe a run without GT (shows over-segmentation magnitude):
& $PY eval\eval_reconnect.py --describe --run output\full_pipeline\<base>

# Evaluate against ground truth:
& $PY eval\eval_reconnect.py --run output\full_pipeline\<base> --gt gt.tif --out report.json
```

## Development-only hybrid grouping factorial

`studies/hybrid_grouping_factorial.py` runs the predeclared 2x2
fragment-source by grouping-kernel study on the 20 `synthetic_thick`
development scenes. It compares input-skeleton versus stored Stage-2 Hough
fragments under minimum-turn versus FilaSeg staged grouping. Every cell is
scored on the same input-derived common fragments, uses singleton treatment
for GT-unassigned fragments, and uses a density-stratified paired block
bootstrap. Stage-4 rendering and overlap suppression are absent.

```powershell
& $PY eval\studies\hybrid_grouping_factorial.py
```

The default report is
`output/development_hybrid_grouping_factorial/results.json`. The command
refuses locked-evaluation paths and refuses to overwrite an existing study
directory. It is exploratory development evidence only; adopting a method
change requires a separately frozen method and fresh locked-v2 evaluation.

## Ground-truth formats

**Label image** (`.tif`/`.png`, single channel): `0` = background, `>0` =
filament id. Works for synthetic GT and painted real masks.

**Centerline JSON** (cheap manual annotation; run-independent and reusable):

```json
{
  "image_shape_rc": [512, 512],
  "default_width_px": 5.0,
  "filaments": [
    {"id": 1, "centerline_xy": [[12, 30], [40, 32], [90, 38]], "width_px": 8.0}
  ]
}
```

Coordinates are `[x=col, y=row]`. Centerlines are rasterized (and width-dilated)
so fragment-overlap assignment has area to vote on; for thin GT the assignment
also falls back to nearest-label within `--max-dist-px`.

## Synthetic data

`synth_generator.py` produces paired samples (no human labor) under
`<out>/synth_NNNN/`: `mask.png` (UNet-like), `mask_clean.png` (thin,
zero-degradation true skeleton — see below), `sem.png`, `gt_labels.tif`,
`gt_multilabel.npz` (overlap-aware GT — each filament owns its full stroke, so
crossings don't corrupt fragment attribution), `gt_meta.json`, optional
`preview.png` (now GT | mask | mask_clean | SEM).

```powershell
& $PY eval\synth_generator.py --out output\synthetic --n 20 --preview
& $PY eval\synth_generator.py --out output\synth_hires --n 5 --um-per-px 0.00054   # 2x resolution
```

Realism principle: the mask is thresholded from a synthetic intensity *ridge*
with low-frequency longitudinal modulation, so breaks appear at faint spots
(content-correlated), not as i.i.d. pixel noise. **Scale-aware**: linear sizes
scale by `REF_UM_PER_PX / um_per_px`, holding the physical scene constant across
resolutions, so the pipeline's scale-dependent reconnect params stay matched.
Degradation knobs: `--break-depth --break-prob --edge-jagged --false-pos
--occlusion-prob --curviness --n-filaments`.

**Smooth centerlines (2026-07-13):** turn increments are Gaussian-smoothed
along the path (`--curve-smooth`, default 20 px correlation length; `0` =
legacy per-step jitter) so filaments bend at long wavelengths like real ones,
and strokes are rendered sub-pixel (true distance to the polyline, anti-aliased
ridge edges) instead of integer disk stamping. Large-scale `curviness` meaning
is preserved (normalized kernel). Current stress sets built with this:
`input/synthetic_v3/dense40` + `dense60` (4 samples each; datasets live under
`input/`, results under `output/full_pipeline_v3/`).

**Clean skeleton mask, `mask_clean.png` (2026-07-14):** `mask.png`'s realistic
degradation (content-correlated fading, explicit cuts, crossing-occlusion
erasure, false positives, edge roughening) deliberately conflates two
different questions — "can the algorithm recover from a bad mask" and "can the
algorithm resolve topology/crossings given a good one." `mask_clean.png`
answers only the second: it's the true centerline network rasterized thin
(1px), fully connected end-to-end, every crossing a real geometric
intersection, **zero degradation of any kind**. Same pipeline `--mask`
contract (binary PNG) — width isn't encoded, so width would need to come from
`sem.png` (smart-width) or `gt_multilabel.npz` rather than the mask itself.
Generated automatically alongside `mask.png` for every sample; no extra flag.
Run it through `run_batch.py --mask-name mask_clean.png` (also plumbed).

**Result:** on the unmodified production pipeline, `mask_clean.png` scores
**+0.21 F1 over `mask.png`** on both dense40 and dense60 (0.675→0.884,
0.579→0.786) — most of the real-world error is mask-quality recovery, not
topology resolution. But the residual on a *perfect* mask is still dominated
by over-merge (~7:1 over under-merge on dense60), sharpening the standing
conclusion: junction-level ambiguity, not mask gaps, is the harder remaining
problem. Full table and root-cause breakdown in `../Experimentation.md`.

> Known gap: the synthetic is currently *harder* than the example real crop
> (more crossings / curvier filaments → more over-merge and angle-bin
> fragmentation). Tune `--curviness` / `--n-filaments` against real masks, and
> treat the **real manual GT as the anchor**; synthetic is for volume + stress.

## Batch run + aggregate

```powershell
& $PY eval\run_batch.py --samples output\synthetic --tag mybatch
& $PY eval\run_batch.py --samples output\synthetic --no-run   # re-eval existing runs
& $PY eval\run_batch.py --samples input\synthetic_v3\dense40 --mask-name mask_clean.png `
    --out-root output\full_pipeline_clean\dense40 --tag clean_dense40   # zero-degradation mask
```

Runs the pipeline on each sample, snapshots the reconnect rejection log into
the run folder, evaluates against `gt_labels.tif`, and writes
`output/full_pipeline/_eval_<tag>.csv` plus mean/median aggregates. `--mask-name`
selects which per-sample mask file to feed the pipeline (default `mask.png`;
use `mask_clean.png` for the zero-degradation variant — give it a distinct
`--out-root` so runs don't collide with the degraded-mask ones, sample names
are identical between the two).

## Root-cause diagnostic

Attributes every grouping error to a cause, so tuning is targeted:

```powershell
& $PY eval\diagnose_connections.py --run output\full_pipeline\<base> --gt <gt> --out diag.json
```

- **Missed** (under-merge): per gate that rejected it, with the median offending
  metric next to the configured threshold (so you see near-miss vs far), plus a
  `never_candidate` estimate (joins never evaluated).
- **Wrong** (over-merge): per stage that accepted it, plus `merged via chain`
  (filaments fused transitively / by overlap-absorb, not a single gate).

Needs the per-run rejection log at `<run>/3.reconnect/rejection_log.csv`
(`run_batch.py` snapshots it; otherwise copy it from
`3.reconnect/output/rejection_log_straight.csv` after a run).

**Overlap-aware joined test (2026-07-13).** Whether a tip pair is "joined" is
decided against each pred instance's full overlapping mask (like the metric),
not the flattened label image — flattening gives a disputed crossing pixel to
one layer, which made correctly-joined pairs look severed. On synthetic_v3
dense40 the flat test inflated missed pairs ~3x (556 → 192 after the fix) and
invented an `accepted_then_lost: 33` bucket (true value 0 at tip level; at the
fragment level postprocess breaks 3 pairs while fixing 103). GT-side and
wrong-merge attribution remain flat-nearest for comparability. If a diag
report predates this fix, re-run it before trusting the missed buckets.

## What "F1" means here

Every score below is the **fragment-clustering pairwise F1** on the **final**
pipeline output (the instance labels you actually use), vs ground truth:

- **Precision** — of all fragment *pairs* the pipeline put in the same instance,
  the fraction that are truly the same filament. Low precision = **over-merge**
  (wrong joins).
- **Recall** — of all fragment pairs that are truly the same filament, the
  fraction the pipeline grouped together. Low recall = **under-merge**
  (missed joins / fragmentation).
- **F1** = harmonic mean of precision and recall. **1.0 = perfect grouping**,
  0 = none. Permutation-invariant (ignores ID renumbering); needs only grouping
  GT, not pixel-perfect masks.

"**mean F1**" = average over the validation set (1 real manual-GT crop + 4
synthetic crops). Always validate on both — single-image wins overfit (we caught
one; see the reconnect table).

## Validated baselines (FINAL output, corrected 2026-06-08)

| Pipeline | real crop | synthetic (4) mean | 5-sample mean |
|---|---|---|---|
| **tiled-Hough (production, only stage-1 method)** | **0.709** | 0.689 | **0.693** |
| skeleton (experimental, removed 2026-07-14 — see Experimentation.md) | 0.429 | — | ~0.51 |

> Earlier numbers (~0.44–0.56) were a measurement error: they scored the **thin
> reconnect** stage, not the **final** postprocess output. Postprocess
> (smart-width render + overlap-absorb) does a second merge round worth **+0.15
> F1** (thin 0.556 → final 0.709). Always eval `--run <dir>` (defaults to
> `final/`), never `3.reconnect` directly, for user-facing quality.

## Experiment log (≈25 runs)

All FINAL-metric unless marked *(thin)*; best value **bold**.

**Stage 3 — reconnect** (detailed sweep: weights, overlap, all stage gates, geometry/bridge)
| Experiment | result | verdict |
|---|---|---|
| **`weights.forward` 2.0 → 3.0** | real 0.720 → **0.735**, synthetic neutral | **PORTABLE WIN (ported); precision 0.771→0.821; saturates at ≥3.0** |
| `weights.line_residual` 2.0 → 1.0 | real +0.015 but synth −0.011 mean | overfit, discarded (same real flip as forward but worse on synth) |
| `weights` opposition / bridge_intrusion / length_reward | inert | defaults optimal |
| `overlap.kill_thr` 0.2/0.3/0.5/0.7 | 0.680/**0.709**/0.667/0.671 | 0.3 optimal |
| relaxed `max_line_residual_px` 16/24/32 | 24 real +0.016 but synth tied | overfit, discarded |
| strict gates (`min_forward_cos`, `min_inward_opposition`, `max_line_residual_px`) | flat | optimal |
| clear gates (`clear_merge_max_dist_px` 16, `clear_merge_min_opposition` 0.6) | optimal | optimal |
| `search_size_px`, `max_component_tips`, `max_turn_deg`, `bridge_tangent_scale`, `tip_dedupe_sep_px`, `max_width_ratio` | inert | defaults optimal |
| tip-smoothing / spur-drop / global-matching *(thin, code)* | −0.009 / −0.006 / ±0.000 | reverted |

**Stage 1 — stringart (tiled-Hough)** — all optimal at defaults; auto-scale is essential
| Experiment | result | verdict |
|---|---|---|
| `--angle-step-deg` 12 / 15 / 20 | 0.473 / **0.709** / 0.550 | 15 optimal |
| `--tile-grid-vote-min` 1 / 2 | 0.543 / **0.709** | 2 optimal (denoising helps) |
| `--tile-size` 128 / 192 / 256 *(thin)* | **0.556** / 0.316 / 0.421 | 128 optimal |
| **auto-scale OFF** (fixed Hough values) | ~0.53 (real) | **auto-scale is ESSENTIAL — never disable** |
| `--min-accept-density` 0.3/0.45/0.6/0.75 | 0.720/**0.720**/0.578/0.571 | 0.45 optimal |
| `--hough-maxgap-mult` 0.6/1.0/1.5/2.0 | 0.584/**0.720**/0.559/0.437 | 1.0 optimal (auto-scale already right) |
| `--hough-threshold-mult` 0.7/1.0/1.3 | 0.542/**0.720**/0.517 | 1.0 optimal |
| `--newpix-mult` (coarse 0.6, fine 0.7/0.8/0.9) | real +0.013 (stacks to 0.748) BUT **synth3 crashes −0.131 at every value <1.0** | **discarded** — lowering the new-pixel threshold admits spurious short segments on dense data (sharp cliff) |

**Stage 2 — preprocess** (all optimal at defaults — no headroom)
| Experiment | result | verdict |
|---|---|---|
| `--pre-line-close-len` 4 / 10 / 18 | **0.709** / 0.451 / 0.547 | 4 optimal |
| `--pre-fit-smoothing` 0.5 / 1.5 / 3.0 | 0.659 / **0.709** / 0.709 | 1.5 optimal |
| `--pre-clean-to-path` on / off | **0.709** / 0.603 | ON essential (2-tip path reduction) |

**Stage 4 — postprocess**
| Experiment | result | verdict |
|---|---|---|
| `--overlap-absorb-thr` 0.35/0.45/0.55/0.6/0.7 | 0.678/0.697/**0.709**/0.709/0.682 | 0.55–0.6 optimal |
| **`--occlusion-trim-thr` 0.15/0.25/0.4/0.6** | 0.707/0.709/**0.720**/0.704 (real) | **0.4 = PORTABLE WIN: real +0.011, synthetic neutral** |
| `--min-keep-len` 8/20/35/50 | 0.703/**0.709**/0.693/0.660 | 20 optimal |
| `--occlusion-trim-min-px` 25/50/100 | 0.709 (inert) | default fine |
| thin reconnect → final | 0.556 → **0.709** | postprocess adds +0.15 |

**Skeleton-graph rewrite** (mean over 5 samples; sandbox)
| Step | mean F1 |
|---|---|
| production skeleton | ~0.51 (real 0.429) |
| aggressive junction merge | 0.586 |
| + generalize pairing (any arm count) | 0.592 |
| **+ mask pre-smoothing** | **0.622 (best skeleton)** |
| + robust PCA tangents | 0.618 (neutral, reverted) |
| `per_piece` binning | 0.463 (reverted) |
| spur-prune 3→8 | 0.566 (reverted) |
| width-aware graph traversal | real 0.532 (reverted) |
| stage-1 gap-bridge 12px (Hough maxLineGap analog) | real 0.43 (over-merge, reverted) |
| stage-1 gap-bridge 5px, naive | 0.623 (wash: real +0.035 but synth3 −0.077) |
| **+ smart gap-bridge 5px (width-consistency + mutual-best)** | **0.635 (best skeleton; real 0.632; 4/5 up; synth3 −0.057)** |
| + collinearity (line-residual) gate | 0.634 (neutral; synth3 unmoved → its wrong bridge is *genuinely collinear* = SEM territory; disabled) |

**Cross-pipeline transfer**
| Experiment | result | verdict |
|---|---|---|
| mask pre-smoothing on **tiled-Hough** | 0.709 → **0.483** | hurts — does NOT transfer |

**Dense v3 (smooth synthetic, 2026-07-13/14; sandbox A/B, 8 dense runs + val + real)**
| Experiment | result | verdict |
|---|---|---|
| orientation gate: tip-trace estimator (`orient_mismatch_estimator: path`) | mean −0.005, recall unchanged everywhere | junction tips genuinely ambiguous — reverted to default off |
| + trace skip 8 px | mean +0.001 | wash |
| tip-hook trim 10 px before tangent fit | worst −0.066 (d60) | trims real curvature — don't retry |
| stage-4 occlusion-trim OFF | −0.03 every sample | 0.4 setting confirmed again |
| stage-4 render extra skeleton branches (15 px) | 0.000 ×3, −0.046 | no effect — don't retry |
| **relaxed-stage `max_smooth_rms_px` 5.5 → 1.4** (pure config) | **real FINAL +0.020 (P+R both up), val 0/0/0/0 clean, dense neutral** | **PORT CANDIDATE** — first win found via the corrected diagnostic (truth-labeled accepted links); one-line config edit, no code; see sandbox/README.md |
| `max_smooth_rms_px` → 1.1 | val −0.065 | too tight — perturbs proposal scoring, discarded |
| `max_line_residual_px` tightening | kills 3–4x more true than wrong | residual is a poor discriminator — discarded |

Full narrative + root-cause findings: `sandbox/README.md` ("Dense-scene
investigation"). Headline: `accepted_then_lost` was a flat-label measurement
artifact (real postprocess damage: 3 broken vs 103 fixed pairs), and the
remaining dense-scene levers are junction-level joint assignment and SEM
evidence — not better pairwise tip measurements.

**Bottom line:** tiled-Hough is a strong, well-tuned local optimum. **Two
portable tuning wins found and ported (they stack, synthetic-neutral): real
final 0.709 → 0.720 (`occlusion-trim-thr` 0.25→0.4) → 0.735 (`weights.forward`
2.0→3.0).** Both raise precision by reducing wrong merges, with no synthetic
regression. Beyond these two, no parameter change across all four stages
produced a cross-validated gain (the tempting real-only candidates — relaxed
residual=22/24, occlusion-trim=0.50, newpix-mult=0.6–0.9, line_residual=1.0 — all
washed or regressed on synthetic, confirmed by **fine** sweeps around each). The
remaining error on *both* pipelines is recall = bridging real gaps in the UNet
mask, which geometry can't resolve → see SEM plan ([SEM_PLAN.md](SEM_PLAN.md)).
Always cross-validate a candidate on real **and** synthetic — single-image wins
overfit (e.g. relaxed-residual=24 looked +0.016 on real but washed on synthetic).

## Why tiled-Hough wins (what we learned)

1. **It gap-bridges at stage 1.** `HoughLinesP`'s `maxLineGap` joins collinear
   pixels across small raster gaps, so a filament broken by a few missing pixels
   is recovered as *one* segment immediately — before reconnect's 50px distance
   cap ever applies. Skeleton has no equivalent; it defers all bridging to the
   distance-capped reconnect.
2. **It absorbs jaggedness without fusing neighbors.** Line-fitting finds the
   dominant straight direction through a noisy edge; skeletonization turns every
   bump into a spur / false junction → fragmentation. *Proof:* mask
   pre-smoothing helps skeleton (+0.03) but hurts tiled (−0.23) — Hough never
   needed it, and closing only fuses adjacent filaments.
3. **It carries width.** Branch masks are real mask pixels, so downstream
   smart-width / overlap-absorb operate on true area evidence; skeleton branches
   are 1 px until re-dilated.
4. **Density gate + multi-grid voting denoise** (vote-min 2 > 1, confirmed).

**Transferable lesson:** the cheapest place to bridge gaps and absorb jaggedness
is the *earliest* stage, on real mask pixels — exactly where Hough does it.
Fixing fragmentation late (geometry-only, distance-capped reconnect) is strictly
weaker. This is also why SEM evidence should act as a *continuity* signal close
to where gaps are bridged, not only as a late reconnect gate (see SEM_PLAN.md).

## Status / next

Measurement foundation complete; tuning exhausted and validated on real +
synthetic. The one remaining lever is **SEM-bridge evidence** to bridge real
mask gaps / reject crossings — the universal bottleneck. Plan in
[SEM_PLAN.md](SEM_PLAN.md).

## Development-only density-by-width factorial study

`generators/synth_factorial.py` independently varies summed centreline length
density and physical bundle width. The default 3 by 3 grid uses length densities
0.020, 0.040, and 0.060 px per px squared, width centres 6, 11, and 16 px, and
four development seeds. Within each seed and length-density block, the degraded
one-pixel input, clean centreline, and overlap-aware centreline ground truth are
byte-identical across width cells. Width is therefore a negative control for
Stage-3 grouping and an active factor only for thick Stage-4 rendering.

```powershell
python eval/generators/synth_factorial.py
python eval/studies/factorial_density_width.py
```

The commands write only under
`output/development_factorial_density_width/`. Both programs reject paths that
contain `synthetic_locked_v1`. The report records achieved thick coverage,
one-pixel coverage, crossing density, filament count, and measured gap
statistics. These data are exploratory development evidence and must not be
combined with the existing locked evaluation.
