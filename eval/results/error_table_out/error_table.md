# Categorized error table — actionable failure analysis

## Headline metrics + complementary instance-recovery

`F1/P/R` = fragment-pairwise (current metric). `well-recovered` = GT filaments whose dominant pred id covers ≥80% of the filament AND is ≥80% pure — a human-meaningful 'basically got it' count that ignores small fragment leaks.

| case | GT | pred | F1 | P | R | well-recovered | filaments deleted |
|---|--:|--:|--:|--:|--:|--:|--:|
| real_crop | 59 | 70 | 0.766 | 0.871 | 0.684 | 35/58 (60%) | 5 |
| synth_0001 | 25 | 31 | 0.761 | 0.84 | 0.695 | 16/25 (64%) | 1 |
| synth_0002 | 25 | 29 | 0.831 | 0.875 | 0.791 | 18/25 (72%) | 1 |
| synth_0003 | 25 | 24 | 0.855 | 0.855 | 0.855 | 17/23 (74%) | 3 |

## UNDER-MERGE categories (failed connects)

| case | U1_crossing_split | U2_wide_break | U3_mid_break | U4_dropped | U5_minor_leak |
|---|---|---|---|---|---|
| real_crop | 8 | 3 | 3 | 1 | 1 |
| synth_0001 | 4 | 0 | 3 | 0 | 1 |
| synth_0002 | 4 | 1 | 1 | 0 | 1 |
| synth_0003 | 3 | 0 | 1 | 2 | 0 |
| **TOTAL** | **19** | **4** | **8** | **3** | **3** |

## OVER-MERGE categories (wrong connects)

| case | O1_wrong_bridge | O2_crossing_fusion | O3_crossing_leak |
|---|---|---|---|
| real_crop | 4 | 7 | 0 |
| synth_0001 | 0 | 4 | 0 |
| synth_0002 | 2 | 2 | 0 |
| synth_0003 | 1 | 3 | 0 |
| **TOTAL** | **7** | **16** | **0** |

## Over-merge SEVERITY (the real metric refinement)

A wrong-connect has two very different outcomes. Pairwise-F1 weights both by fragment-pair count, not by instance severity — that is the metric weakness:

- **ABSORBED — a filament DELETED: 10.** The minority filament's own dominant id *is* the shared pred, so it has no separate instance. A true, severe instance error (lost a filament).
- **partial leak — both filaments survive: 13.** The minority filament still has its own id elsewhere; only some crossing fragments leaked. Mild at the instance level, but pairwise-F1 penalizes it like a deletion.

So precision is depressed partly by ~half-severity leaks. An **instance-level metric** (well-recovered % + filaments-deleted count) tracks the real goal — counting/measuring distinct filaments — better than fragment-pair precision alone. Crossing leaks (O3, only 0) are rare, so the issue is leak *severity weighting*, not leak count.


## real_crop — wrong-connect → damaged-id links

- pred 5: gA=27 + gB=31 [O1_wrong_bridge, bridge n/a] -> gB=31 leaked 20% here; main id elsewhere (pred 23)
- pred 5: gA=27 + gB=112 [O1_wrong_bridge, bridge n/a] -> gB=112 ABSORBED into gA=27 (lost as distinct id)
- pred 13: gA=83 + gB=12 [O2_crossing_fusion, crossing 61deg] -> gB=12 leaked 20% here; main id elsewhere (pred 46)
- pred 19: gA=44 + gB=37 [O2_crossing_fusion, crossing 17deg] -> gB=37 ABSORBED into gA=44 (lost as distinct id)
- pred 22: gA=41 + gB=37 [O2_crossing_fusion, crossing 42deg] -> gB=37 leaked 50% here; main id elsewhere (pred 19)
- pred 25: gA=20 + gB=6 [O2_crossing_fusion, crossing 12deg] -> gB=6 leaked 25% here; main id elsewhere (pred 2)
- pred 26: gA=79 + gB=41 [O1_wrong_bridge, bridge 0deg] -> gB=41 leaked 20% here; main id elsewhere (pred 22)
- pred 36: gA=2 + gB=47 [O1_wrong_bridge, bridge n/a] -> gB=47 ABSORBED into gA=2 (lost as distinct id)
- pred 37: gA=99 + gB=36 [O2_crossing_fusion, crossing 2deg] -> gB=36 leaked 50% here; main id elsewhere (pred 53)
- pred 37: gA=99 + gB=67 [O2_crossing_fusion, crossing 23deg] -> gB=67 ABSORBED into gA=99 (lost as distinct id)
- pred 43: gA=25 + gB=60 [O2_crossing_fusion, crossing 89deg] -> gB=60 ABSORBED into gA=25 (lost as distinct id)

## synth_0001 — wrong-connect → damaged-id links

- pred 4: gA=8 + gB=13 [O2_crossing_fusion, crossing 30deg] -> gB=13 leaked 20% here; main id elsewhere (pred 6)
- pred 5: gA=18 + gB=9 [O2_crossing_fusion, crossing 6deg] -> gB=9 leaked 33% here; main id elsewhere (pred 15)
- pred 7: gA=4 + gB=15 [O2_crossing_fusion, crossing 3deg] -> gB=15 leaked 40% here; main id elsewhere (pred 17)
- pred 15: gA=18 + gB=9 [O2_crossing_fusion, crossing 6deg] -> gB=9 ABSORBED into gA=18 (lost as distinct id)

## synth_0002 — wrong-connect → damaged-id links

- pred 3: gA=5 + gB=23 [O2_crossing_fusion, crossing 14deg] -> gB=23 leaked 25% here; main id elsewhere (pred 11)
- pred 6: gA=21 + gB=25 [O1_wrong_bridge, bridge n/a] -> gB=25 ABSORBED into gA=21 (lost as distinct id)
- pred 15: gA=15 + gB=8 [O1_wrong_bridge, bridge 7deg] -> gB=8 leaked 33% here; main id elsewhere (pred 10)
- pred 15: gA=15 + gB=24 [O2_crossing_fusion, crossing 30deg] -> gB=24 leaked 20% here; main id elsewhere (pred 9)

## synth_0003 — wrong-connect → damaged-id links

- pred 2: gA=18 + gB=22 [O1_wrong_bridge, bridge n/a] -> gB=22 ABSORBED into gA=18 (lost as distinct id)
- pred 3: gA=3 + gB=20 [O2_crossing_fusion, crossing 31deg] -> gB=20 leaked 33% here; main id elsewhere (pred 9)
- pred 5: gA=2 + gB=4 [O2_crossing_fusion, crossing 40deg] -> gB=4 ABSORBED into gA=2 (lost as distinct id)
- pred 9: gA=3 + gB=20 [O2_crossing_fusion, crossing 31deg] -> gB=20 ABSORBED into gA=3 (lost as distinct id)