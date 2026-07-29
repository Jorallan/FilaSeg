# Categorized error table — actionable failure analysis

## Headline metrics + complementary instance-recovery

`F1/P/R` = fragment-pairwise (current metric). `well-recovered` = GT filaments whose dominant pred id covers ≥80% of the filament AND is ≥80% pure — a human-meaningful 'basically got it' count that ignores small fragment leaks.

| case | GT | pred | F1 | P | R | well-recovered | filaments deleted |
|---|--:|--:|--:|--:|--:|--:|--:|

## UNDER-MERGE categories (failed connects)

| case | U1_crossing_split | U2_wide_break | U3_mid_break | U4_dropped | U5_minor_leak |
|---|---|---|---|---|---|
| **TOTAL** | **0** | **0** | **0** | **0** | **0** |

## OVER-MERGE categories (wrong connects)

| case | O1_wrong_bridge | O2_crossing_fusion | O3_crossing_leak |
|---|---|---|---|
| **TOTAL** | **0** | **0** | **0** |

## Over-merge SEVERITY (the real metric refinement)

A wrong-connect has two very different outcomes. Pairwise-F1 weights both by fragment-pair count, not by instance severity — that is the metric weakness:

- **ABSORBED — a filament DELETED: 0.** The minority filament's own dominant id *is* the shared pred, so it has no separate instance. A true, severe instance error (lost a filament).
- **partial leak — both filaments survive: 0.** The minority filament still has its own id elsewhere; only some crossing fragments leaked. Mild at the instance level, but pairwise-F1 penalizes it like a deletion.

So precision is depressed partly by ~half-severity leaks. An **instance-level metric** (well-recovered % + filaments-deleted count) tracks the real goal — counting/measuring distinct filaments — better than fragment-pair precision alone. Crossing leaks (O3, only 0) are rare, so the issue is leak *severity weighting*, not leak count.
