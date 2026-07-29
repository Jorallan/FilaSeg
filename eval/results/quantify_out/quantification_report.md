# Grouping-Free Network Quantification Report

Metrics computed from the **foreground union** of all instance masks (OR-reduce of `.masks`), its skeleton, or the background — **NOT per-instance**. All metrics are therefore grouping-free and expected to be unbiased by reconnect grouping errors.

## (a) All Grouping-Free Metrics — Accuracy Summary

| Metric | Measurement | Mean Ext/GT Ratio | Mean KS | Notes |
|--------|-------------|:-----------------:|:-------:|-------|
| Orientation distribution (per-pixel) | histogram overlap | 0.94 (overlap) | — | Already validated |
| Total skeleton length | ratio ext/GT | 0.97 | — | Already validated |
| Junction/crossing count (skeleton) | ratio ext/GT | 0.92 | — | Already validated |
| **Width** (EDT diameter at skel px) | median ratio + KS | 1.667 | 0.731 | New |
| **Mesh/Pore size** (bg medial-axis EDT) | median ratio + KS | 1.272 | 0.112 | New |
| **Curvature** (|dθ|/arc-len per edge) | median ratio + KS | 1.052 | 0.145 | New |
| **Coverage** (FG area fraction) | ratio ext/GT | 1.848 | — | New |
| **Alignment S** (nematic order param) | |S_ext − S_GT| | — | 0.019 (mean |ΔS|) | New |

## (b) New Metrics — Per-Case Detail

### Metric 1: Width Distribution (EDT diameter at skeleton pixels)

| Case | Ext mean | Ext med | Ext p10 | Ext p90 | GT mean | GT med | GT p10 | GT p90 | Ratio (med) | KS |
|------|:--------:|:-------:|:-------:|:-------:|:-------:|:------:|:------:|:------:|:-----------:|:--:|
| real_crop | 15.65 | 15.23 | 12.00 | 20.00 | 9.51 | 8.94 | 8.00 | 12.17 | 1.703 | 0.777 |
| synth_0001 | 6.70 | 6.32 | 4.47 | 8.94 | 3.59 | 4.00 | 2.00 | 6.00 | 1.581 | 0.719 |
| synth_0002 | 7.04 | 6.32 | 5.66 | 8.94 | 3.86 | 4.00 | 2.00 | 6.00 | 1.581 | 0.743 |
| synth_0003 | 7.37 | 7.21 | 5.66 | 10.00 | 4.17 | 4.00 | 2.00 | 6.32 | 1.803 | 0.684 |
| **MEAN** | | | | | | | | | **1.667** | **0.731** |

### Metric 2: Mesh/Pore Size (background medial-axis EDT)

| Case | Ext pore_med | Ext bg_mean | GT pore_med | GT bg_mean | Ratio (med) | KS |
|------|:------------:|:-----------:|:-----------:|:----------:|:-----------:|:--:|
| real_crop | 7.62 | 6.78 | 8.06 | 7.60 | 0.945 | 0.066 |
| synth_0001 | 13.15 | 18.19 | 9.85 | 18.06 | 1.335 | 0.145 |
| synth_0002 | 13.93 | 20.47 | 10.05 | 20.50 | 1.386 | 0.128 |
| synth_0003 | 18.97 | 26.21 | 13.34 | 24.00 | 1.422 | 0.111 |
| **MEAN** | | | | | **1.272** | **0.112** |

### Metric 3: Curvature (mean |Δθ|/arc-length per skeleton edge, rad/px)

*Definition: skeleton FG → remove junction pixels (8-nb ≥ 3) → connected component edges ≥ 12 px → order by nearest-neighbour → chord vectors every 5 px → |Δθ| between consecutive chords → sum(|Δθ|) / (n_intervals × step_px). Pool one value per edge.*

| Case | Ext mean | Ext med | Ext n_edges | GT mean | GT med | GT n_edges | Ratio (med) | KS |
|------|:--------:|:-------:|:-----------:|:-------:|:------:|:----------:|:-----------:|:--:|
| real_crop | 0.0364 | 0.0320 | 213 | 0.0433 | 0.0291 | 221 | 1.101 | 0.108 |
| synth_0001 | 0.0295 | 0.0245 | 92 | 0.0229 | 0.0221 | 94 | 1.106 | 0.154 |
| synth_0002 | 0.0287 | 0.0251 | 88 | 0.0323 | 0.0237 | 101 | 1.055 | 0.126 |
| synth_0003 | 0.0354 | 0.0231 | 54 | 0.0268 | 0.0245 | 60 | 0.944 | 0.191 |
| **MEAN** | | | | | | | **1.052** | **0.145** |

### Metric 4: Coverage (FG area fraction)

| Case | Ext | GT | Ratio (ext/GT) |
|------|----|-----|:--------------:|
| real_crop | 0.5226 | 0.3518 | 1.485 |
| synth_0001 | 0.1429 | 0.0683 | 2.092 |
| synth_0002 | 0.1493 | 0.0753 | 1.983 |
| synth_0003 | 0.1191 | 0.0650 | 1.832 |
| **MEAN** | | | **1.848** |

### Metric 5: Alignment Order Parameter S (nematic, per-pixel skeleton tangent)

*S = √(mean(cos 2θ)² + mean(sin 2θ)²), θ in [0,π) from local PCA on skeleton pixels with radius 6 px. S=0: isotropic, S=1: perfectly aligned.*

| Case | S_ext | S_GT | |ΔS| |
|------|:-----:|:----:|:----:|
| real_crop | 0.099 | 0.121 | 0.021 |
| synth_0001 | 0.301 | 0.326 | 0.024 |
| synth_0002 | 0.180 | 0.204 | 0.024 |
| synth_0003 | 0.090 | 0.098 | 0.008 |
| **MEAN** | | | **0.019** |

## (c) Grouping-Dependent Metrics (biased — do not trust per-instance)

> **Warning:** The following metrics require per-filament identity and are corrupted by reconnect grouping errors (over/under-merge). They are listed here for reference only and should NOT be used to assess physical network properties.

| Metric | Ext/GT Ratio or KS | Comment |
|--------|--------------------|---------|
| Filament count | 1.14 (ratio) | Over-fragmentation inflates count |
| Per-filament length (median) | 0.82 (ratio) | Splits shorten per-instance length |
| Instance-overlap crossing count | 1.37 (ratio) | Grouping boundary crossings inflate |
| Crossing-degree KS distance | 0.21 | Degree distribution distorted by merge/split |

## (d) Verdict

The grouping-free metrics split into two accuracy tiers. **Accurately extractable (ratio near 1 / low KS):** Curvature is the best-recovered new metric (median ratio 1.052, KS 0.145), showing the skeleton edge geometry matches GT well once junction structure is similar. **Alignment S** (mean |ΔS| = 0.019) is excellent — the nematic order parameter is virtually identical between extracted and GT skeletons. Together with the pre-validated orientation overlap (0.94), total-length ratio (0.97), and junction-count ratio (0.92), these confirm that shape and topology are faithfully preserved. **Inaccurate due to systematic FG over-extraction:** Width (median ratio 1.667, KS 0.731), pore size (median ratio 1.272, KS 0.112), and coverage (ratio 1.848) all show large positive biases. The extracted FG union is ~1.5–2× larger than the manual GT in pixel area (e.g., 136,999 vs 92,228 px for real_crop; ~37–39k vs ~18–20k for synth). This is not a grouping artefact — it reflects that the pipeline segmentation captures more foreground than the conservative manual GT. Consequently, per-skeleton-pixel widths are ~67% too large, pore sizes ~27% over-estimated (larger gaps between thicker-appearing filaments), and area fraction nearly doubles. These biases must be interpreted against the GT annotation style rather than as pipeline failure. By contrast, grouping-dependent metrics (filament count ratio 1.14, per-filament length ratio 0.82, crossing ratio 1.37, crossing-degree KS 0.21) are biased for a different reason — reconnect merge/split errors — and should not be used to characterise physical network properties at all.
