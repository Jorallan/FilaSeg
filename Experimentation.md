# Experimentation Log

A single reference for every parameter, gate, algorithm, and architecture change that has been
**tried** on this pipeline (roughly May–July 2026), whether it was **kept, ported, reverted, or
rejected**, and **why**. Its purpose is to stop the same experiment from being re-run and getting
the same (negative) answer a second time.

**How to read this doc:**
- **PORTED** = live in production (`3.reconnect/`, `4.postprocess/`, `Tools/run_full_sem_pipeline.py`).
- **PORT CANDIDATE** = validated in the sandbox, not yet in production (rare — check current state
  before trusting the label; grep production configs to confirm).
- **Reverted / discarded / don't retry** = tried, measured, did not survive cross-validation, and
  should not be re-proposed without new evidence.
- All results are the **fragment-clustering pairwise F1** on the pipeline's **FINAL** output vs
  ground truth, unless marked *(thin)* = the `3.reconnect` stage before postprocess. See
  `eval/README.md` for what F1 means and how to reproduce any number here.
- Historical output paths (`output/full_pipeline/sem00000_crop512_manual`, `output/synthetic_val`,
  `output/audit`, `output/synthetic_v2`, etc.) may no longer exist — `output/` has been cleared
  more than once. They're kept here for provenance; treat any specific old path as possibly stale
  and re-derive from current `input/` + a fresh pipeline run if you need to reproduce a number.
- The current manual GT anchor lives at
  `sem_full_00000_1p66_crop512_manual/sem_full_00000_1p66_crop512_manual_labels.tif` (paired with
  `input/sem_full_00000_1p66_crop512/`).

---

## Quick reference: what's actually PORTED to production today

| # | Change | Where | Real-crop effect |
|---|---|---|---|
| 1 | `weights.forward` 2.0 → 3.0 | `3.reconnect/reconnect_config.yaml` | 0.709 → 0.735 |
| 2 | `occlusion-trim-thr` 0.25 → 0.4 | `Tools/run_full_sem_pipeline.py` default | 0.709 → 0.720 |
| 3 | `stage_relaxed.min_inward_opposition` 0.4 → 0.85 | `3.reconnect/reconnect_config.yaml` | suppresses wrong long bridges |
| 4 | Crossing-prevention gate (`max_orientation_mismatch_deg: 40`, all 3 stages) | `3.reconnect/reconnect_config.yaml` + `reconnect_utils_straight.py` | 0.767 → 0.803 (dev), 0.792 → 0.839 (unseen) |
| 5 | `reconnect_repair` (candidate coverage fix, mutual-best, long-bridge veto, contact/boundary continuation passes) | `3.reconnect/reconnect_config.yaml` (`reconnect_repair.enabled: true`) + `reconnect_utils_straight.py` | mean 0.696 → 0.728 |
| 6 | Instance-recovery co-primary metric (well-recovered / deleted / false_pred) | `eval/metrics.py` | measurement only |
| 7 | `stage_relaxed.max_smooth_rms_px` 5.5 → 1.4 | `3.reconnect/reconnect_config.yaml` | 0.755 → 0.775 (+0.020) |
| 8 | Smooth synthetic generator (`--curve-smooth`, sub-pixel strokes) | `eval/synth_generator.py` | realism, not F1 |
| 9 | Overlap-aware tip attribution in the diagnostic | `eval/diagnose_connections.py` | measurement only — fixed a ~3x inflated error budget |
| 10 | Skeleton-decompose stage-1 **removed** (never beat tiled Hough) | deleted `1.stringart/skeleton_decompose.py` + sandbox mirror + CLI flag | simplification; `stringart_tiles.py` is now the only stage-1 |
| 11 | `mask_clean.png` — zero-degradation synthetic mask variant | `eval/synth_generator.py` + `eval/run_batch.py --mask-name` | isolates topology error from mask-recovery error: **+0.21 F1 on dense40/60** — see below |

Stack effect on the real crop across items 1–2: **0.709 → 0.735** (both synthetic-neutral).
Items 4–5 pushed dev F1 further (0.735 → ~0.78 range depending on sample). Item 7 is the most
recent, +0.020 on top of everything else.

**Bottom line so far:** the architecture (tiled-Hough → preprocess → 3-stage reconnect →
smart-width postprocess) is a strong, thoroughly-tuned local optimum. Every remaining gap traces
to one of two things: (a) real UNet mask gaps that pure geometry cannot bridge safely (needs SEM
continuity evidence — see [SEM-evidence investigation](#sem-evidence-investigation)), or (b)
junction-level ambiguity at dense crossings where a locally-correct join and a locally-plausible
wrong join are geometrically indistinguishable (needs joint, not pairwise, assignment — see
[Dense-scene investigation](#dense-scene-investigation-2026-07-13-14)). **The clean-mask experiment
(2026-07-14, item 11) decouples these two for the first time: (a) accounts for most of the F1 gap
but (b) is the harder, more dominant one — over-merge stays ~7:1 vs under-merge even with a
topologically perfect input mask.**

---

## Stage 1 — stringart (mask → per-angle branches)

### Tiled-Hough (`stringart_tiles.py`) — the production method, all knobs optimal at defaults

| Experiment | Result | Verdict |
|---|---|---|
| `--angle-step-deg` 12 / **15** / 20 | 0.473 / **0.709** / 0.550 | 15 optimal |
| `--tile-grid-vote-min` 1 / **2** | 0.543 / **0.709** | 2 optimal (multi-grid denoising helps) |
| `--tile-size` **128** / 192 / 256 *(thin)* | **0.556** / 0.316 / 0.421 | 128 optimal |
| auto-scale **ON** vs OFF (fixed Hough values) | 0.72 vs ~0.53 | **auto-scale is essential — never disable** |
| `--min-accept-density` 0.3 / **0.45** / 0.6 / 0.75 | 0.720 / **0.720** / 0.578 / 0.571 | 0.45 optimal |
| `--hough-maxgap-mult` 0.6 / **1.0** / 1.5 / 2.0 | 0.584 / **0.720** / 0.559 / 0.437 | 1.0 optimal |
| `--hough-threshold-mult` 0.7 / **1.0** / 1.3 | 0.542 / **0.720** / 0.517 | 1.0 optimal |
| `--newpix-mult` 0.6–0.9 | real +0.013 (stacks to 0.748) BUT **synth3 crashes −0.131 at every value <1.0** | **discarded** — admits spurious short segments on dense data (sharp cliff on cross-validation) |

**Auto-scale is the load-bearing mechanism**: it adapts Hough parameters to filament width per
image; disabling it tanks F1 regardless of any other tuning. Never disable in production.

### Skeleton-decompose (`skeleton_decompose.py`) — experimental alternative, REMOVED 2026-07-14

**Hypothesis:** tiled-Hough is fundamentally a line-fitting/chopping approach that scatters curvy
filaments across tile seams and angle bins — exactly the fragmentation reconnect then has to
undo. A connectivity-preserving stage-1 (skeletonize → split at junctions → bin by local tangent →
smart-merge collinear arms) should produce cleaner input.

**What was built and tuned, in order (mean F1 unless noted):**

| Step | mean F1 | Note |
|---|---|---|
| naive per-pixel angle-bin skeleton | 0.394 (real), ~0.35 mean | worse than tiled by −0.16 to start |
| junction-merge (cos_thr −0.90, min-arm 25) baseline | ~0.51 (real 0.429) | too strict — only 22/273 junctions merged |
| **loosen to cos_thr −0.75, min-arm 10** (first real win) | real 0.429 → 0.487 | recall gain (0.349→0.455), slight precision dip |
| + generalize pairing to any arm count ≥2 | 0.592 | best so far |
| `per_piece` binning (whole-piece vote instead of per-pixel) | 0.463 | **reverted** — loses smooth-curve splitting |
| spur-prune 3→8 px | 0.566 | **reverted** |
| **+ mask pre-smoothing** (binary_closing 3×3 before skeletonize) | **0.622 (best skeleton)** | real 0.483→0.573; confirms jagged-skeleton noise was the real-crop cap |
| + robust PCA arm tangents | 0.618 | **neutral, reverted** — junction *pairing* quality was not the bottleneck |
| width-aware graph traversal (`GRAPH_TRACE`, true skeleton-graph + width-aware junction matching) | real 0.532 (width-on) / 0.517 (width-off) | **worse than smart_merge (0.573) — reverted.** DT-based width on ~3px filaments is noisy and rejects valid pairs |
| stage-1 gap-bridge 12px (Hough `maxLineGap` analog) | real 0.43 | **over-merge, reverted** |
| gap-bridge 5px, naive | 0.622→0.623 (wash) | real +0.035 but synth3 −0.077 |
| **+ smart gap-bridge 5px (width-consistency + mutual-best)** | **0.635 (new best skeleton)** | real 0.573→0.632, 4/5 samples up |
| + collinearity (line-residual) gate on the bridge | 0.634 | **neutral** — synth3's wrong bridge is *genuinely collinear*; geometry cannot separate it from a real gap |

**Final skeleton verdict: 0.635 mean, still −0.06 to −0.10 behind tiled-Hough (0.693–0.735
depending on era).** Every remaining avenue was tried: junction-pairing sophistication (robust
tangents, graph traversal) made it *worse*; the only real wins were skeleton-cleanliness
(mask-smoothing) and a smart gap-bridge. The residual gap is bridging real UNet mask gaps — the
same SEM-evidence-territory conclusion as the reconnect side.

**Why tiled-Hough wins (the transferable lesson):**
1. **It gap-bridges at stage 1.** `HoughLinesP`'s `maxLineGap` joins collinear pixels across small
   raster gaps immediately — before reconnect's 50px distance cap ever applies. Skeleton has no
   equivalent; it defers all bridging to the distance-capped reconnect.
2. **It absorbs jaggedness without fusing neighbors.** Line-fitting finds the dominant straight
   direction through a noisy edge; skeletonization turns every bump into a spur / false junction →
   fragmentation. *Proof it doesn't transfer:* mask pre-smoothing helps skeleton (+0.03) but
   **hurts tiled-Hough (0.709 → 0.483)** — Hough never needed it, and closing just fuses adjacent
   filaments. Any "clean the mask first" idea should be tested per-pipeline, not assumed universal.
3. **It carries width.** Branch masks are real mask pixels, so downstream smart-width /
   overlap-absorb operate on true area evidence; skeleton branches are 1px until re-dilated.
4. **Density gate + multi-grid voting denoise** (vote-min 2 > 1, confirmed).

**Decision (2026-07-14):** removed entirely — `1.stringart/skeleton_decompose.py`, its sandbox
mirror, the `--skeleton-decompose` / `--no-skeleton-decompose` CLI flag, and the dedicated sandbox
A/B harnesses (`sandbox/ab_stage1.py`, `sandbox/run_full_skel.py`). `stringart_tiles.py` is now the
only stage-1 method — the flag had defaulted to the *worse* experimental path
(`DEFAULT_SKELETON_DECOMPOSE = True`), which was an active footgun: any direct pipeline invocation
that forgot `--no-skeleton-decompose` silently ran the inferior method. `eval/run_batch.py` always
passed the flag internally, so all batch-eval numbers in this doc were on tiled-Hough regardless.

**Don't retry:** skeleton-first stage-1 in any form, without first bridging real UNet mask gaps —
that gap is the actual bottleneck on both architectures, and better skeleton bookkeeping alone
provably cannot close it (graph traversal made it worse, not better).

---

## Stage 2 — preprocess (per-branch cleanup)

All optimal at defaults — no headroom found.

| Experiment | Result | Verdict |
|---|---|---|
| `--pre-line-close-len` **4** / 10 / 18 | **0.709** / 0.451 / 0.547 | 4 optimal — longer fuses parallel same-bin filaments |
| `--pre-fit-smoothing` 0.5 / **1.5** / 3.0 | 0.659 / **0.709** / 0.709 | 1.5 optimal |
| `--pre-clean-to-path` **on** / off | **0.709** / 0.603 | ON essential — reduces multi-tip components to 2 tips |

---

## Stage 3 — reconnect (tip-tip merging)

### Early tip-geometry experiments (May–June 2026) — all reverted

The first hypothesis was that reconnect's tip-tangent geometry was noisy and needed hardening.
**Empirically wrong** — reverted every time:

| Experiment | Result | Verdict |
|---|---|---|
| Tip trace smoothing | F1 −0.009 (P 0.752→0.789, R 0.441→0.419) | trades recall for precision, net negative — misses are **distance-limited** (rejected before tangent gates run), so better tangents can't recover them |
| Spur/hook outlier-drop | F1 −0.006 | same story |
| Global matching (mutual-best gate replacing greedy selection) | dF1 = 0.000, **identical** | wrong merges are already **mutual-best** pairs (two filaments crossing at a shallow angle are each other's best collinear partner) — matching/competition cannot fix a wrong *edge*, only a wrong *selection among valid edges* |
| Targeted "collinear-clear" rule (dist≤16, resid≤2, |fwd|≥0.85, layer_gap≤1) | synth1 +0.006 but **real −0.018** | over-merged the real crop; reverted |

**Conclusion that redirected the whole investigation:** recall was capped by the **50px distance
gate** (misses have ~95px median gap — the user vetoed raising the cap, correctly, since it causes
wrong long-range connects) and by wrong merges being **structurally transitive** ("via chain":
253 chain-propagated vs 6 direct in the original diagnosis) rather than single-gate mistakes.

### `reconnect_repair` — PORTED 2026-06-10

Fixed four Step-3 defects as one coordinated switch (`reconnect_repair.enabled: true`):
1. Candidate discovery now covers the full active 50px relaxed distance gate (a genuine bug — the
   old search stopped at 40px even though the relaxed gate accepted up to 50px).
2. Normal proposals require reciprocal best matches (mutual-best), reducing greedy chain merges.
3. Long bridges crossing ≥2 component IDs rejected unless endpoint extrapolation is exceptionally
   collinear (≤6.5px residual).
4. Two topology-checked local passes (`contact_continuation`, `boundary_continuation`) recover
   overlap overshoot and 0–2.1px endpoint-to-boundary contacts, both SEM-free, both requiring
   bilateral endpoint support + mutual-best pairing + a connected two-ended virtual path.

Validated across 8 samples (5 dev + 3 unseen), **zero regressions**, mean F1 0.696 → 0.728
(synthetic mean ~0.689 → ~0.730). Biggest single gain +0.134 on unseen data.

### Crossing-prevention gate + opposition tightening — PORTED 2026-06-11

Deep id-by-id error analysis found crossings dominate both error directions — a filament splits
where it crosses another (median facing gap 3.4px, pieces *touch*) **and** two crossing filaments
fuse, from the *same* ambiguous-junction event.

**What failed first (don't retry):**
- Post-merge topology split: −0.055 (over-merges are 93% path-like, invisible to branch-point
  detection).
- Max-path-turn guard: no separating signal (24° vs 15°, distributions overlap).
- Clear-merge forward-floor: −0.01 to −0.04.
- SEM-based rejection gate (tried at this point in the investigation): **regressed real** (recall
  cost sank it) — see [SEM-evidence investigation](#sem-evidence-investigation) for the full,
  later, more careful attempt.
- "Split-recovery exemption" (un-block the orientation gate when no third filament is present):
  **NO-GO** after formal phase-0 characterization — only 71/352 blocked same-GT pairs have no third
  filament, they sit in a pool with 668 true different-filament L-meetings at 9:1 against, and no
  logged signal (dist recovers 27/71 while leaking 15 wrong fusions) separates them.

**What worked:**
1. `stage_relaxed.min_inward_opposition` 0.4 → 0.85 — long bridges must be strongly collinear,
   suppresses wrong-bridge over-merges.
2. **The big win — merge-time crossing-PREVENTION gate.** Reject a join when the two pieces'
   **local orientation at the join** (PCA over a small window, `orient_mismatch_radius_px: 16`)
   differs by ≥ `max_orientation_mismatch_deg` (**40**, validated optimum — see the density-ceiling
   sweep below). *Local*, not whole-piece, orientation is essential: it lets a smooth curve
   continue while blocking a crossing. Prevention beats a post-hoc split because blocking a wrong
   join also *frees the arms* to merge correctly — one gate fixes over- and under-merge together.

   Validated: dev (4 samples) F1 0.767→0.803, unseen (3 samples, seed 999) 0.792→0.839; precision
   *and* recall both up, zero regressions. Known limitation: sharply-curving single filaments
   (≥40° local bend) can fragment — rare for stiff CNTs, mitigate with threshold 50 if real curvy
   data suffers.

Also ported at this point: the **instance-recovery co-primary metric** (`well_recovered` /
`deleted` / `false_pred` in `eval/metrics.py`) — pairwise F1 alone weighted a full filament
deletion the same as a minor crossing leak, which under-penalized deletions.

### Detailed parameter sweep (2026-06-09) — one more portable win

Swept weights, overlap handling, all three stage gates, geometry/bridge parameters.

| Experiment | Result | Verdict |
|---|---|---|
| **`weights.forward` 2.0 → 3.0** | real 0.720 → **0.735** (precision 0.771→0.821), synthetic neutral | **PORTABLE WIN, PORTED — saturates at ≥3.0** |
| `weights.line_residual` 2.0 → 1.0 | real +0.015 but synth −0.011 mean | overfit, discarded |
| `weights` opposition / bridge_intrusion / length_reward | inert | defaults optimal |
| `overlap.kill_thr` 0.2 / **0.3** / 0.5 / 0.7 | 0.680 / **0.709** / 0.667 / 0.671 | 0.3 optimal |
| relaxed `max_line_residual_px` 16 / 24 / 32 | 24: real +0.016 but synth tied | overfit, discarded |
| strict/clear gates, `search_size_px`, `max_component_tips`, `max_turn_deg`, `bridge_tangent_scale`, `tip_dedupe_sep_px`, `max_width_ratio` | inert | defaults optimal |

**Fine sweeps around the winners** confirmed both are robust local optima, no further gain:
forward flips to 0.735 anywhere in [2.5, 6.0]; occlusion-trim real peak is actually 0.50 (0.727)
but that **regresses synthetic** (mean 0.712 < 0.720 at 0.40) — overfit, kept at 0.40; relaxed
residual=22 reaches the same real merge-flip as forward but via a synthetic-riskier lever, skipped.

### Density-ceiling re-confirmation (2026-07-07)

Cross-validated sweep of the two gates the density diagnosis blamed most, on a 4-tier density
stress set (n=25/40/60 filaments; F1 0.78→0.647→0.603 as density rises):

| Sweep | Result | Verdict |
|---|---|---|
| `stage_relaxed.max_tip_distance_px` 50→60/70 | real F1 −0.025/−0.027 (both precision AND recall drop — wrong long bridges consume tips + chain damage) | **the June distance-cap veto re-confirmed, even with the orientation gate + opposition 0.85 already in place — dead, don't retry** |
| `max_orientation_mismatch_deg` 40→45/50 (all stages) | loses F1 in *every* group, including the dense sets it was hypothesized to help | **40°/16px confirmed robust optimum at all densities — don't re-sweep** |

**Conclusion:** the density degradation is an information ceiling, not mis-tuning. The blocked
joins are geometrically indistinguishable from crossings; loosening any gate loses more via chain
over-merge than it recovers.

**Operational lesson from this pass:** the sandbox reconnect copy had **drifted** from production
(missing the crossing-prevention gate that had been ported). Sweeps run against a drifted sandbox
are silently invalid. **Whenever something is ported to `3.reconnect/`, re-sync the sandbox copy**
— or better, use `ab_reconnect.py`, which reads each run's own `reconnect_config_active.yaml`
(mirrors whatever is actually running) rather than trusting a static sandbox config file.

### FWHM width — investigated, deliberately REJECTED (2026-07-07)

Production width estimation is `estimate_sem_guided_width()` ("smart-width": gradient-edge scan
normal to the path) in `4.postprocess/post_process_reconnect.py`. FWHM (background-subtracted
half-max) was ported as an opt-in alternative and compared.

**Why FWHM lost:** on the real crop, smart-width=13px vs FWHM=15px — FWHM *over-read* the real
image relative to the thin manual-centerline GT (~8.9px). On synthetic data FWHM was numerically
closer to the generative-stroke GT (0.73–0.82× vs smart's 1.22–1.37×), but that synthetic
convention differs from the real manual-annotation convention, so synthetic-closeness didn't
transfer. Width mode barely moved final F1 (largest Δ 0.048) — no segmentation-quality reason to
switch — and smart-width simply looked better on the real crop the user cares about.

**Decision: smart-width is committed. Do not re-propose FWHM.** The FWHM port, `sandbox/width_fwhm.py`,
and its comparison artifacts were removed.

*(Kept from that investigation: the denser `synthetic_v2` stress set — n=25/40/60 — which
reconfirmed the recall/density degradation above, and the smart-width diagnostic tool
`sandbox/width_diag.py`.)*

---

## Dense-scene investigation (2026-07-13/14)

Context: the synthetic generator was made realistic first (see
[Synthetic data generator](#synthetic-data-generator-eval-synth_generatorpy) below), producing new
stress sets `input/synthetic_v3/dense40` + `dense60`. Production baselines on them: dense40 F1
0.677 (P 0.760 / R 0.617), dense60 F1 0.585 (P 0.628 / R 0.549) — under-merge dominant.

### Round 1 — tip-geometry fixes, all WASH (don't retry)

The diagnostic (pre-fix, see below) pointed at the orientation gate as the top lever: it
"falsely" rejected 128 true pairs. Every measurement fix tried:

| Experiment | Result | Verdict |
|---|---|---|
| Tip-trace orientation estimator instead of mask-window PCA (`orient_mismatch_estimator: path`) | mean −0.005; **recall unchanged in every sample**; 41/42 diagnosed pairs still rejected | the trace itself is hooked at junction tips — measurement is not the fixable problem |
| + skip first 8px of trace | mean +0.001 | wash |
| Adaptive tip-hook trim before tangent fit (`tip_hook_trim_px: 10`) | 4×0.000 on dense40; **−0.066** worst on dense60 (recall collapse) | trims genuine curvature — regression |
| hook-trim + path estimator combined | mean −0.013 | worst combination |

**Root-cause finding:** true pairs rejected by the orientation gate have a *real* GT tangent
mismatch of only ~8° median — but every estimator tried (mask PCA, skeleton trace, hook-trimmed
trace) reads 30°+, because the fragment's end geometry genuinely *is* bent at the crossing. Fixing
the measurement doesn't help: rescuing those joins admits equally-plausible wrong joins at the
same junctions, so recall gains cancel against precision losses. **Pairwise local geometry cannot
break this tie — only junction-level context (deciding all arms of a crossing jointly) or SEM
continuity evidence can.**

### Round 1b — stage-4 experiments

| Experiment | Result | Verdict |
|---|---|---|
| stage-4 occlusion-trim OFF | −0.03 on every sample | 0.4 setting re-confirmed — keep ON |
| stage-4 overlap-absorb OFF | ~0.000 | neutral at these densities |
| stage-4 render leftover skeleton branches (union side-branches, not just the dominant path) | 3× 0.000, one −0.046 | unions are path-shaped in practice; extra branches only add noise — don't retry |

### The diagnostic itself was the real bug (2026-07-13/14)

`eval/diagnose_connections.py` attributed tip pairs by looking them up in the **flattened** label
image. At a crossing, flattening gives the disputed pixel to exactly one overlapping layer — so a
pair that reconnect had correctly joined, sitting under another filament's stroke, looked "in
different ids": a phantom miss, blamed on whichever gate happened to reject it earlier in the log.

**Fixed:** the "joined" test is now overlap-aware — a pair counts as joined when some single pred
instance has pixels within `--max-dist` of *both* tips, tested against that instance's full
(possibly overlapping) mask, matching the metric's own convention (new `_tip_instance_sets`
helper). GT-side and wrong-merge attribution deliberately stay flat-nearest for comparability.

**Effect on the dense40 error budget (4 samples):**

| Bucket | Before (flat) | After (overlap-aware) |
|---|---|---|
| missed pairs total | 556 | **192** |
| blamed on orientation gate | 128 | **28** |
| `accepted_then_lost` | 33 | **0** |
| distance-gate misses (median gap) | 400 (~120px) | 152 (**~200px**) |

At the fragment level (the metric's own unit), postprocess was found to break only **3** same-GT
pairs across 4 images while *fixing* **103** — the "accepted_then_lost" bucket had been almost
entirely a flat-label artifact, not a real postprocess defect.

**Lesson:** ~65% of the round-1 diagnosed budget was phantom. This fully explains why round 1's
fixes washed — they were chasing a mostly-fictitious problem. **Any diagnostic conclusion drawn
before 2026-07-14 should be re-derived before being trusted**, especially anything involving
crossings/dense scenes.

### Round 2 — the corrected budget led to a real, portable win

With the true budget (missed 192 = ~152 chain-collapse-gap "distance" misses at ~200px median +
28 genuine orientation-gate rejections + noise), a different method was used: **label every
ACCEPTED link** in the rejection logs with overlap-aware GT truth, instead of only studying
rejections.

**Finding:** `stage_clear` accepts are 99% correct (2 wrong / 393). **All real wrong links enter
at `stage_relaxed`** (29 wrong / 168 true, ~4/image) and are long, wobbly bridges: `smooth_rms`
median 0.94 vs 0.52 for true joins, distance 32px vs 24px. `line_residual` separates *poorly*
(tightening it kills 3–4× more true links than wrong ones) — **smoothness is the only usable
discriminator.**

The relaxed stage's `max_smooth_rms_px` was 5.5 (scaled) — far above the ~0.5–1.0px true joins
actually need.

| Experiment | Result | Verdict |
|---|---|---|
| Distance-conditional code gate (bridges >26px must have rms ≤1.1) | real FINAL +0.018 | worked, but added new code for no extra benefit — see next row |
| **Plain config tighten: `stage_relaxed.max_smooth_rms_px` 5.5 → 1.4** | **real FINAL +0.020 (P and R both up), synthetic_val 0/0/0/0 clean, dense mean ~neutral** | **PORTED 2026-07-14 — matches the code gate's result with zero new code (user's explicit preference: prefer a config tweak over a new gate)** |
| `max_smooth_rms_px` → 1.1 (more aggressive) | val −0.065 on one sample | too tight — tightening this value also inflates the `smooth_rms/max_smooth_rms_px` term used in proposal *scoring* (not just the reject gate), and at 1.1 that side-effect flips rankings; 1.4 is the safe point |
| `max_line_residual_px` tightening (any value) | kills 3–4× more true than wrong | discarded — same conclusion as the June residual sweep |

Validated on 13 datasets (8 dense v3 + 4 synthetic_val + real crop): real crop +0.020,
synthetic_val perfectly clean (no regression on independent data), dense mean roughly neutral (one
−0.05 outlier on the most extreme tier, offset by a +0.03 elsewhere) — same "real-positive,
synthetic-neutral" profile as every other portable win in this log. At the extreme dense tiers,
true and wrong long bridges overlap in every logged metric — those two regressions are irreducible
pairwise, consistent with the round-1 junction-ambiguity finding.

**Verification note:** while porting, a pipeline invocation accidentally used the (default-on,
now-deleted) skeleton-decompose stage-1 instead of tiled-Hough, producing a spuriously low score
that looked like a regression. See the "skeleton-decompose default" entry in the quick-reference
table — this is why that flag was removed rather than just fixed.

### Remaining levers (evidence-ordered, as of 2026-07-14)

1. **Junction-level joint assignment** — decide all arms of a crossing together (mutual-exclusive,
   continuity-cost matching) instead of independent pairwise gates. This is the structural
   response to the "pairwise geometry cannot break the tie" finding above, and the
   [clean-mask experiment below](#clean-mask-experiment-result--decouples-topology-error-from-mask-recovery-error-2026-07-14)
   confirms it's the **larger** of the two remaining error sources, independent of mask quality.
2. **SEM continuity evidence** — the orthogonal signal for the (now proven smaller, but still real)
   mask-gap-bridging side; see below.

---

## Synthetic data generator (`eval/synth_generator.py`)

**Design principle (from the start):** the mask is derived by thresholding a synthetic intensity
*ridge* with low-frequency longitudinal modulation, so breaks are content-correlated (faint spots)
like a real UNet failure mode, not i.i.d. pixel noise. Scale-aware: linear sizes scale by
`REF_UM_PER_PX / um_per_px` so reconnect's scale-dependent params stay matched across resolutions.

**Overlap-aware GT** (`gt_multilabel.npz`): each filament owns its full stroke even under
crossings, so fragment-to-filament attribution isn't corrupted by which filament happened to be
drawn on top.

**Smooth centerlines + sub-pixel rendering (2026-07-13):** the original generator added
independent random heading noise every step, producing visibly jagged, unrealistic filaments.
Fixed two ways:
- Turn increments are now Gaussian-smoothed along the path (`--curve-smooth`, default 20px
  correlation length; `0` = legacy jitter) so filaments bend at long wavelengths like real ones.
  The smoothing kernel is normalized so `curviness`'s large-scale meaning is preserved.
- Strokes render by true distance to a sub-pixel polyline (KDTree) with an anti-aliased ridge edge,
  instead of integer-rounded disk stamping — removes stair-stepping and width quantization.

Current stress sets built with this generator: `input/synthetic_v3/dense40` + `dense60` (4 samples
each). See the [Dense-scene investigation](#dense-scene-investigation-2026-07-13-14) above for what
they were used to find.

**Clean skeleton mask, `mask_clean.png` (2026-07-14).** The realistic `mask.png` combines several
degradations (content-correlated fading, explicit hard cuts, **crossing occlusion** — locally
erasing the lower-priority filament's pixels at a crossing to simulate a UNet losing the dimmer
filament — spurious blobs, morphological roughening). Crossing-occlusion in particular erases a
real crossing's evidence entirely, which directly opposes what reconnect's job is (resolving
crossings) rather than merely testing recovery from a degraded mask — this conflates two different
questions in one artifact. Added `mask_clean.png`: the true centerline network rasterized thin
(1px, `render_clean_skeleton_mask()`), fully connected end-to-end, every crossing a real geometric
intersection, **zero degradation of any kind** — no fading, cuts, occlusion, false positives, or
roughening. Generated automatically alongside `mask.png` (additive, same binary-mask pipeline
contract; width not encoded — use `sem.png`/`gt_multilabel.npz` for width). Backfilled into
`input/synthetic_v3/dense40` + `dense60` with the original seeds (verified `mask.png`/`sem.png`/
`gt_labels.tif` stayed byte-identical). `eval/run_batch.py` gained `--mask-name` to run either
variant through the same harness.

### Clean-mask experiment result — decouples topology error from mask-recovery error (2026-07-14)

Ran the **unmodified production pipeline** on `mask_clean.png` for all 8 dense-v3 samples
(`output/full_pipeline_clean/`) and compared against the same-seed `mask.png` (degraded) runs:

| tier | mask | F1 | P | R | missed (4 samples) | wrong (4 samples) |
|---|---|---|---|---|---|---|
| dense40 | degraded | 0.675 | 0.768 | 0.608 | 207 | 600 |
| dense40 | **clean** | **0.884** | 0.891 | 0.877 | **59** | **377** |
| dense60 | degraded | 0.579 | 0.646 | 0.527 | 372 | 1297 |
| dense60 | **clean** | **0.786** | 0.766 | 0.807 | **143** | **1008** |

**Big, clean effect: +0.21 F1 (dense40) / +0.21 F1 (dense60) from mask quality alone**, with the
identical reconnect/postprocess config on both sides. This quantifies, for the first time
decoupled from any other variable, how much of the pipeline's real-world error is "recovering from
a degraded mask" vs "resolving instance topology." Most of it is mask recovery.

**But the residual — same config, topologically perfect input — is not zero, and its shape is
diagnostic.** `missed` (under-merge) collapsed 71% (dense40) / 61% (dense60), confirming most
recall loss really was mask-gap-bridging. `wrong` (over-merge) dropped far less (−37% / −22%) and
**stayed completely dominant** — on clean dense60 the wrong:missed ratio is ~7:1 (1008:143), *more*
lopsided than on the degraded mask (~3.5:1). Two things follow:

1. **Reconnect still has real work to do even on a perfect mask.** The small residual `missed`
   bucket (median gap 150–240px on `dist`) is stage-1's own fragmentation (tile seams / angle-bin
   chopping — see the Stage 1 section above) reasserting itself; it is not zero even with no UNet
   mask gaps at all.
2. **Over-merge/junction-ambiguity is the harder, more fundamental problem — confirmed with the
   mask-quality confound fully removed.** This sharpens the earlier "SEM evidence is the universal
   next lever" conclusion: SEM continuity would attack the (now proven smaller) `missed`/recall
   side. The (now proven larger, especially at higher density) `wrong`/precision side needs
   junction-level joint assignment — pairwise gates cannot resolve it regardless of how clean the
   mask is, because the ambiguity is genuinely in the crossing geometry itself, not in mask noise.

**Reproduce:**
```powershell
& $PY eval\run_batch.py --samples input\synthetic_v3\dense40 --mask-name mask_clean.png `
    --out-root output\full_pipeline_clean\dense40 --tag clean_dense40
& $PY eval\diagnose_connections.py --run output\full_pipeline_clean\dense40\synth_0000 `
    --gt input\synthetic_v3\dense40\synth_0000\gt_labels.tif --out diag.json
```

(Also fixed in the same pass: `diagnose_connections.py` crashed with `_csv.Error: field larger
than field limit` on the larger dense60 rejection logs — raised `csv.field_size_limit()`.)

---

## Evaluation subsystem (`eval/`)

Built 2026-06-05 to solve "can't measure reconnect quality — no ground truth." Design choice:
**synthetic generator first**, not real annotation (real GT was added later as the anchor).

**Primary metric: fragment clustering.** After stage 1+2 the image is a set of atomic fragments;
GT assigns each to a true filament, the pipeline assigns each to a predicted id. Comparing the two
partitions of the *same* fragment set is permutation-invariant (immune to ID-renumbering churn),
decomposable (pairwise precision↓ = over-merge, recall↓ = under-merge), and cheap on GT (needs
grouping truth, not pixel-perfect masks). Secondary: pixel Panoptic Quality (harsh on 1px-wide
filaments — not the objective).

**Two measurement bugs found and fixed early, both significant:**
1. **Thin vs final confusion (2026-06-08).** Nearly every early number (~0.44–0.56) had scored the
   **thin `3.reconnect`** stage, not the **final** postprocess output the user actually uses.
   Postprocess (smart-width render + overlap-absorb) does a second merge round worth **+0.15 F1**
   (thin 0.556 → final 0.709 on the same run). **Always eval `--run <dir>` (defaults to `final/`),
   never `3.reconnect` directly, for user-facing quality claims.**
2. **Stale-directory read.** `run_batch.py` pre-created the output directory before invoking the
   runner; the runner then appended a timestamp on collision, so eval silently read a stale run.
   Fixed by parsing the actual run directory from the runner's own stdout.

**Overlap-aware tip attribution fix (2026-07-14)** — see the
[Dense-scene investigation](#dense-scene-investigation-2026-07-13-14) above; this is the most
consequential fix to the diagnostic tool itself.

**Metric caveats to remember:** pairwise P/R is brutal on small clusters and singleton splits;
crossing-attribution noise is real in dense synthetic scenes (mitigated, not eliminated, by
overlap-aware GT). The real manual GT crop is the anchor — synthetic is for volume and stress, not
as a substitute ground truth.

---

## SEM-evidence investigation

The one lever that keeps recurring as "the next real gain" across almost every section above:
geometry alone cannot distinguish a real long gap from a wrong crossing, because the mask is
missing pixels that only exist in the underlying SEM intensity.

**Phase 0 (2026-06-08) — clear GO.** Built `eval/sem_features.py` / `eval/sem_phase0.py`:
enumerate geometrically-plausible candidate bridges, label by GT, measure per-feature AUC
(good-bridge vs bad-bridge). Results: tiled-real `bridge_min_contrast` AUC 0.852 (vs geometry's
0.26–0.56), skeleton-real 0.711, and — the headline result — **`bridge_min_contrast` = 0.933 on
exactly the wrong bridge that every geometric criterion in the skeleton investigation had failed
to catch** (synth3). SEM information genuinely resolves cases geometry cannot. Caveat flagged at
the time: small N on real data (15–62 candidates).

**Phase 1 (2026-06-08) — net-negative on real, blocked on GT volume.** Threaded SEM contrast into
the sandbox reconnect as an opt-in rejection gate (`sem.min_contrast`). Measured: real F1
0.709 → 0.687 at threshold −0.05 (precision up 0.725→0.762 but **recall sank** 0.693→0.626 — good
long faint-gap bridges got rejected alongside bad ones). The synthetic-mean *looked* like a gain
(0.711→0.726) but that was traced to synthetic SEM being cleaner/more-separable than real SEM
(sim-to-real inflation), not a real signal. **A single hard threshold cannot convert a
promising-but-small-N Phase-0 AUC into a real win — good faint bridges overlap wrong ones in
SEM-contrast space.** Kept as an opt-in precision tool (off by default), not adopted.

**Stated blocker, still open:** more real manual-GT volume. One crop cannot both tune and validate
an SEM-based gate or a learned Phase-2 affinity model without overfitting to that one crop. This
remains the recommended next investment if SEM evidence is revisited.

**Don't re-attempt** a single-feature SEM threshold gate without either (a) meaningfully more real
GT, or (b) treating it as a *recall lever* (extend reach for SEM-continuous far gaps) rather than a
precision veto, which is the untried framing.

---

## Crossing / quantification measurement audit (2026-06-15)

A separate audit asked: even where grouping (reconnect) is imperfect, are the *downstream physical
quantities* (length, orientation, crossing count) still trustworthy?

**General thesis, confirmed across three independent measurements:** measurements that are
**grouping-free** (computed on the foreground mask or local pixel neighborhoods, not on
instance-level groupings) are accurate *right now*; measurements that are **grouping-dependent**
(per-instance/per-filament) carry the full reconnect error as bias.

- **Orientation:** per-pixel local tangent (structure-tensor-like) overlap with GT = 0.94 —
  accurate. Whole-instance PCA orientation = 0.74 — biased by grouping errors.
- **Total length:** extracted/GT = 0.97 — accurate (grouping-free, it's just total foreground
  length).
- **Per-filament length:** median ratio to GT = 0.82 — biased (~18% short), inflated instance count
  from residual splits (+14%). Perfect-grouping ceiling is only 0.93 (~7% is a
  rendering/skeleton-length-vs-GT-stroke offset unrelated to grouping).
- **Junction count** (skeleton degree≥3 nodes, grouping-free): ext/GT 0.92–0.98 — accurate.
- **Instance-overlap crossing count** (grouping-dependent): ext/GT 1.37 — over-counts, because
  fragmentation/splits manufacture extra pairwise overlaps.

**Practical recommendation:** report network-level statistics (total length, orientation
distribution, junction density) from grouping-free measurements — they're trustworthy today.
Flag any *per-filament* statistic (individual length, individual crossing degree) as carrying the
reconnect grouping ceiling as bias until the junction-ambiguity problem above is solved.

---

## Don't-retry list (fast lookup)

- Raising `stage_relaxed.max_tip_distance_px` past 50 — re-confirmed harmful twice (June veto,
  July density re-sweep), even with every other gate improvement stacked on top.
- Loosening `max_orientation_mismatch_deg` past 40 — confirmed harmful at every tested density.
- Tightening `max_line_residual_px` anywhere in reconnect — a consistently poor discriminator that
  kills 3–4× more true links than wrong ones every time it's been tried.
- Any single-feature SEM hard-threshold rejection gate (as a *precision veto*) without materially
  more real GT — net-negative on real, gains were sim-to-real inflation.
- Tip-geometry hardening (trace smoothing, spur/hook drop, tip-trace orientation estimator, hook
  trimming) as a fix for junction-tip ambiguity — tried in at least two independent investigations
  (June and July), always a wash or regression. The ambiguity is real geometry, not a measurement
  artifact, at true crossings.
- Global/mutual-best matching as a fix for wrong merges — proven identical to greedy (wrong merges
  are already mutual-best pairs).
- Post-merge topology splitting, max-path-turn guards, clear-merge forward-floors — all tested
  net-negative for over-merge control.
- Mask pre-smoothing as a universal preprocessing step — helps skeleton-decompose, **actively
  hurts tiled-Hough** (0.709→0.483). Pipeline-specific, not a general win.
- FWHM width estimation — deliberately rejected in favor of smart-width after direct comparison.
- Skeleton-decompose as a stage-1 replacement — fully explored (junction merge, mask-smoothing,
  robust tangents, graph traversal, gap-bridging), best case 0.635 vs tiled's 0.71–0.735, and
  removed from the codebase 2026-07-14.
- Rendering postprocess's leftover skeleton side-branches (not just the dominant path) — no effect
  or slight regression; unions are path-shaped in practice.
- Diagnosing dense-scene errors with flat-label (non-overlap-aware) tip attribution — inflates the
  error budget ~3× at crossings; always use the overlap-aware `diagnose_connections.py` (fixed
  2026-07-14).

---

## Where to look for current state

- `eval/README.md` — the metric definition, current experiment-log tables (may be a subset of this
  doc, kept close to the code), and the "why tiled-Hough wins" section.
- `sandbox/README.md` — sandbox workflow, the most recent dense-scene / long-bridge investigation
  narrative, and the `ab_reconnect.py` / `ab_postprocess.py` harness docs.
- `eval/SEM_PLAN.md` — the phased plan for the SEM-evidence step, if picked back up.
- `repo_summary.md` — architecture/file-layout reference.
- This file — the historical *why*, so the same experiment isn't re-run for the same negative
  answer.
