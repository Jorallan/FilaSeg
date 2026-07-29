# Id-by-id audit: pipeline output vs ground truth

Per-GT-filament status and root cause across the validation cases. Status: OK / SPLIT (under-merge) / MERGED (over-merge) / DROPPED (lost upstream).

| case | GT | pred | F1 | P | R | OK | SPLIT | MERGED | S+M | DROP | FP-pred |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clean40_synth0003 | 40 | 42 | 0.751 | 0.743 | 0.758 | 27 | 3 | 4 | 6 | 0 | 0 |

## Failure causes by stage (aggregated over all cases)

- **10** — stage3 reconnect over-merge (transitive chain, no single gate)
- **9** — stage3 reconnect gate rejected the join
- **5** — stage3 reconnect over-merge (direct gate accept)

Reconnect gates most responsible for SPLITs:
  - `distance`: 9
  - `orientation_mismatch`: 2

## clean40_synth0003 — failing filaments

| gt | frags | npred | dom | status | shares | root cause |
|--:|--:|--:|--:|---|---|---|
| 1 | 6 | 3 | 2 | SPLIT+MERGED | 17 | gate:distancex10; merge@transitive(stage3)x28; merge@stage_clearx1 |
| 2 | 2 | 2 | 9 | SPLIT+MERGED | 15 30 | gate:distancex1; merge@transitive(stage3)x16; merge@stage_relaxedx1 |
| 15 | 3 | 2 | 24 | SPLIT |  | gate:distancex1 |
| 17 | 5 | 2 | 2 | SPLIT+MERGED | 1 | gate:distancex2; gate:orientation_mismatchx1; merge@transitive(stage3)x26; merge@stage_clearx2 |
| 23 | 4 | 2 | 17 | SPLIT |  | gate:distancex4 |
| 30 | 4 | 2 | 9 | SPLIT+MERGED | 2 15 | gate:distancex1; merge@transitive(stage3)x29; merge@stage_clearx2 |
| 34 | 6 | 2 | 3 | SPLIT+MERGED | 33 | gate:distancex4; gate:orientation_mismatchx1; merge@transitive(stage3)x17 |
| 35 | 4 | 2 | 25 | SPLIT |  | gate:distancex1 |
| 40 | 4 | 2 | 8 | SPLIT+MERGED | 23 | gate:distancex4; merge@transitive(stage3)x15 |
| 3 | 4 | 1 | 7 | MERGED | 30 | merge@transitive(stage3)x34; merge@stage_clearx5 |
| 10 | 1 | 1 | 13 | MERGED | 2 | merge@transitive(stage3)x6 |
| 32 | 2 | 1 | 20 | MERGED | 1 | merge@transitive(stage3)x3 |
| 33 | 2 | 1 | 3 | MERGED | 34 | merge@transitive(stage3)x16 |