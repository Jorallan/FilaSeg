# Plan: SEM-evidence ("SEM-learned") reconnect

## Why

Across ~25 experiments (all 4 stages, both stage-1 algorithms), the residual
error on **both** pipelines is **recall = bridging real gaps in the UNet mask**,
which geometry cannot resolve: a real filament with a missing-pixel gap and two
*different* filaments crossing at a shallow angle look **identical** to the
geometric gates. The distinguishing evidence is in the **SEM image**, which
reconnect currently ignores entirely:

- A real filament across a gap usually shows **intensity continuity** in the SEM
  (the UNet dropped the pixels; the SEM didn't).
- A wrong crossing bridge **lacks** continuity (it crosses dark background or a
  perpendicular filament).

This lets us raise reach **only where the image justifies it** — resolving the
"don't raise the global distance cap" constraint.

> Placement note (from the "why Hough wins" analysis): the cheapest place to
> bridge gaps is the *earliest* stage, on real mask pixels. So evaluate SEM
> continuity **both** as a late reconnect gate AND as an early gap-bridge step
> (Phase 0 features support both); pick the stronger placement with the harness.

## Method: instrument → hand-gate → learn (de-risked, each measured on real+synth)

### Phase 0 — DONE (2026-06-08): GO ✅

Built `eval/sem_features.py` (`sample_bridge_sem`) + `eval/sem_phase0.py` (enumerate
geometrically-plausible candidate bridges from fragment endpoints, label by GT,
report per-feature AUC of good vs bad bridges). Result — SEM robustly separates
the bridges geometry cannot:

| test | best SEM AUC (feature) | geometry AUC |
|---|---|---|
| tiled real | 0.852 (`bridge_min_contrast`) | 0.26–0.56 |
| skeleton real | 0.711 (`bridge_min`) | 0.59–0.65 |
| skeleton synth3 (the wrong-bridge case) | 0.933 (`bridge_min_contrast`) | 0.85 |
| tiled synth1 | 0.858 (`bridge_min_contrast`) | 0.82 |

Strongest, most consistent features: **`bridge_min_contrast`** and
**`bridge_contrast`** (bridge centerline vs off-bridge background; high = real
continuity, low = crosses dark background), then `bridge_min`. SEM catches
synth3's wrong bridge (AUC 0.93) that every geometric criterion missed. Caveat:
small N on the real crop (15 tiled / 62 skeleton candidates) — re-confirm on more
real GT during Phase 1. **Verdict: proceed to Phase 1.**

Run: `python eval/sem_phase0.py --run <run> --sem <sem.png> --gt <gt>`.

### (original Phase 0 plan)
### Phase 0 — instrument & validate the hypothesis (1–2 days, do FIRST)

Add SEM-bridge feature extraction to the **sandbox** reconnect evaluator,
**logged only** (no decision change). For each candidate pair, sample the SEM
along the Hermite bridge (already computed in `_sample_hermite_bridge`) plus a
few normal offsets, and log features to the rejection CSV next to the geometry.

Candidate SEM-bridge features:
- `bridge_mean`, `bridge_min` intensity along the centerline (continuity; a real
  filament has no fully-dark point, a crossing-gap does).
- `bridge_contrast` = bridge mean − local off-bridge background (parallel offset
  samples).
- `ridge_score` = fraction of bridge samples that are a local max across the
  normal (is there an actual bright ridge?).
- `endpoint_match` = how well bridge intensity matches the two fragments' own
  mean intensities.
- `gap_len`, intensity-weighted "gap cost".

Then **measure separation before building anything**: join logged candidates
with GT (tip→GT lookup, reuse `diagnose_connections._nearest_label_field`) to
label each pair same/different filament, and compute each feature's AUC at
separating true-joins from false-joins, on **real + synthetic**. If SEM features
separate well → proceed; if not → stop and rethink (cheap kill switch).

Deliverables: `sandbox/sem_features.py` (`sample_bridge_sem(sem, bridge_pts)`),
SEM columns in the rejection log, `sandbox/analyze_sem_separation.py` (AUC report).

### Phase 1 — DONE (2026-06-08): rejection gate is net-NEGATIVE on real ⚠️

Built it: `eval/sem_features.sample_bridge_sem` threaded into the sandbox
reconnect (`_SEM_IMAGE`/`_SEM_CFG` globals, set by reconnect_run from
`--background`); a gate in `_evaluate_tip_pair` rejects non-clear bridges whose
`bridge_min_contrast < sem.min_contrast` (short bridges exempt via
`sem.apply_above_dist`, default 15). Opt-in: off unless `sem.min_contrast` set.
Measured on FINAL (reconnect+postprocess) with `sweep_reconnect.py --final`:

| set | F1 | prec | recall |
|---|---|---|---|
| real, off | 0.709 | 0.725 | 0.693 |
| real, −0.05 | 0.687 | 0.762 | 0.626 |
| real+3synth mean, off | 0.711 | 0.743 | 0.686 |
| real+3synth mean, −0.05 | 0.726 | 0.853 | 0.638 |

The gate reliably **cuts over-merge (precision up)** but the **recall cost sinks
the real crop** (faint real long-gap bridges rejected with the wrong ones). The
mean only rises because it helps the synthetic samples, whose SEM is cleaner/more
separable by construction → **sim-to-real gap**. A single-feature hard threshold
can't convert the Phase-0 separation (AUC 0.85, small N) into a real-data win.

**Conclusion: the blocker is real GT volume.** With one real crop (15–62
candidates) we cannot tune or trust an SEM gate for real data, and a learned
model (Phase 2) on one crop would overfit. The gate is kept as an **opt-in
precision tool** (helps crossing-heavy data) but is OFF by default. **Next:
collect a few more manual-GT crops, THEN revisit the SEM recall-lever (extend
reach for SEM-continuous far gaps) and Phase 2 learned affinity.**

### (original Phase 1 plan)
### Phase 1 — hand-crafted SEM gate (Tier-2, no training; ~1 week)

Use the top features from Phase 0 as an extra reconnect gate/score term:
- **Extend reach when SEM-continuous:** allow longer bridges only if
  `bridge_min`/`bridge_contrast`/`ridge_score` clear a threshold → recovers real
  gaps without globally raising `max_tip_distance_px`.
- **Reject crossings:** drop collinear pairs whose bridge lacks SEM support →
  kills the crossing over-merges.
- Tune thresholds on the harness (real+synth), same as the parameter sweeps.
- Target: beat 0.709 (real) / 0.689 (synth). Plumbing: thread `--background`
  SEM into `_evaluate_tip_pair` (sandbox first).

### Phase 2 — learned edge affinity (the "SEM-learned" step; weeks)

Replace the hand gate with a small model `P(same filament | features)`:
- **Labels for free:** every candidate pair in the rejection log is labelled by
  GT (same/diff filament). Synthetic gives unlimited volume; real manual GT
  anchors. Build a tabular dataset: rows = candidate pairs, X = [geometry metrics
  already logged + Phase-0 SEM features], y = same/diff.
- **Model:** gradient-boosted trees (lightgbm/xgboost) or a tiny MLP — **tabular,
  not a deep net**; hundreds–thousands of rows suffice, and feature importance
  tells us what actually matters. (Respects the "no NN hand-wave" constraint.)
- **Integration:** model outputs an affinity → use as the reconnect edge cost
  (augment/replace the hand-weighted `score_scalar`); keep distance as a cheap
  candidate prefilter only.
- **Validation:** train on synthetic + a portion of real, test on **held-out
  real**; compare to Phase-1 and the 0.71 baseline. Watch the sim-to-real gap
  (synthetic vs real SEM texture) — featurize with *relative* contrasts, weight
  real higher, and collect a few more real GT crops if needed.

### Phase 3 — global consistency (optional)

Feed learned affinities into a min-cost path-cover matching for globally
consistent grouping. Only if Phase 2 plateaus.

## Risks & mitigations

- **Sim-to-real gap:** synthetic SEM ≠ real SEM. → relative/normalised features,
  validate on real, more real GT.
- **Placement:** late reconnect gate vs early gap-bridge — decide empirically
  (Phase 0 features serve both).
- **Real GT volume:** 1 manual crop is thin for training. → a few more manual
  centerline annotations (cheap; `eval/` already reads centerline JSON).

## First concrete step

Build `sandbox/sem_features.py` + log SEM features (Phase 0) and run the AUC
separation analysis on the existing real+synthetic runs. One day; tells us
whether SEM evidence is worth the full build before committing.
