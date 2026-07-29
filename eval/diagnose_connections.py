"""Root-cause diagnostic for reconnect misses and wrong merges.

Given a run, its ground truth, and the reconnect rejection log, this attributes
every grouping error to a concrete cause so tuning is targeted instead of blind:

  MISSED connections (under-merge): two tips on the SAME true filament that ended
  in DIFFERENT pred ids. Classified by the gate that rejected them
  (forward_cos / opposition / line_residual / distance / curv_delta / arc_miss /
  bridge_intrusion / smooth_rms / max_turn / ...), or "never_candidate" when no
  log row ever evaluated that join. For each gate, the median of the offending
  metric is reported next to the configured threshold, so you can see e.g.
  "forward_cos misses cluster at 0.78 vs gate 0.80 -> loosen a hair".

  WRONG connections (over-merge): two tips on DIFFERENT true filaments that landed
  in the SAME pred id. Classified by the stage/mechanism that accepted them.

How tips map to filaments: each logged tip coordinate is looked up against the
GT and final-pred label fields (nearest label within a tolerance). The JOINED
test is OVERLAP-AWARE (2026-07-13): two tips count as joined when some single
pred instance has pixels within `--max-dist` of BOTH, voting against each
instance's full (possibly overlapping) mask — the metric's convention. The
previous flat test (`pred_a == pred_b` on the flattened image) mis-scored
crossings: flattening gives a disputed pixel to one layer only, so correctly-
joined pairs under an overlapping stroke looked severed. On synthetic_v3
dense40 that inflated `accepted_then_lost` from ~0 to 33 (metric-level truth:
postprocess breaks 3 fragment pairs while fixing 103) and polluted every
missed-gate bucket. GT-side and wrong-merge attribution stay flat-nearest so
those buckets remain comparable with the experiment history.

Usage
-----
    python eval/diagnose_connections.py --run output/full_pipeline/<base> \
        --gt <gt_labels.tif> [--rejection-log <csv>] [--out diag.json] [--pairs-csv pairs.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A dense run's rejection log can have a very long single field (e.g. many
# candidate branch regions on one row); the 131072-byte default trips on it.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import instance_io as iio          # noqa: E402
import metrics as M                # noqa: E402

# reason -> (human label, metric column most responsible, gate config key)
REASON_METRIC = {
    "distance":          ("dist",          "dist",              "max_tip_distance_px"),
    "forward_cos":       ("forward",        "forward_min",       "min_forward_cos"),
    "opposition":        ("opposition",     "opposition_pos",    "min_inward_opposition"),
    "line_residual":     ("line residual",  "line_resid",        "max_line_residual_px"),
    "curv_delta":        ("curvature",      "curv_delta",        "max_curvature_delta"),
    "arc_miss_frac":     ("arc miss",       "arc_miss_frac",     "max_arc_miss_frac"),
    "bridge_intrusion":  ("intrusion",      "intrusion_frac",    "max_bridge_intrusion_frac"),
    "smooth_rms":        ("smoothness",     "smooth_rms",        "max_smooth_rms_px"),
    "max_turn":          ("turn angle",     "max_turn_deg",      "max_turn_deg"),
    "width_ratio":       ("width ratio",    "width_ratio",       "max_width_ratio"),
    "not_clear_merge":   ("not clear merge", "dist",             None),
    "tip_trace_too_short": ("tip trace short", "dist",           None),
    "clear_merge_multitip_forward": ("clear multitip fwd", "forward_min", None),
    "clear_merge_backward_layer_gap": ("clear backward layer", "layer_gap", None),
}


def _nearest_label_field(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import distance_transform_edt
    nz = lab != 0
    if nz.any() and not nz.all():
        dist, (ir, ic) = distance_transform_edt(~nz, return_indices=True)
        return lab[ir, ic], dist
    return lab.copy(), np.zeros(lab.shape)


def _label_at(field: np.ndarray, dist: np.ndarray, r: float, c: float, max_dist: float) -> int:
    rr = int(min(max(round(r), 0), field.shape[0] - 1))
    cc = int(min(max(round(c), 0), field.shape[1] - 1))
    if dist[rr, cc] > max_dist:
        return 0
    return int(field[rr, cc])


def _fnum(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _tip_instance_sets(
    instances: "iio.InstanceSet",
    tips: List[Tuple[int, int]],
    max_dist: float,
) -> List[set]:
    """For each tip (r, c): the set of instance ids with a pixel within max_dist.

    Overlap-aware replacement for flat nearest-label lookup. At a crossing the
    flattened label image gives the disputed pixel to a single layer, so a tip
    sitting under an overlapping stroke was attributed to whichever instance won
    the flattening — making correctly-joined pairs look severed. Voting against
    each instance's full mask matches how the metric assigns fragments.
    """
    R = int(np.ceil(max_dist))
    oy, ox = np.ogrid[-R:R + 1, -R:R + 1]
    disk = (oy * oy + ox * ox) <= float(max_dist) ** 2
    H, W = instances.shape
    out: List[set] = [set() for _ in tips]
    if not tips:
        return out
    trc = np.asarray(tips, dtype=np.int64)
    for iid, mask in instances.masks.items():
        rows = np.any(mask, axis=1)
        if not rows.any():
            continue
        cols = np.any(mask, axis=0)
        r0, r1 = int(np.argmax(rows)), int(H - 1 - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(W - 1 - np.argmax(cols[::-1]))
        near = np.flatnonzero(
            (trc[:, 0] >= r0 - R) & (trc[:, 0] <= r1 + R)
            & (trc[:, 1] >= c0 - R) & (trc[:, 1] <= c1 + R)
        )
        for k in near:
            tr, tc = int(trc[k, 0]), int(trc[k, 1])
            wr0, wr1 = max(0, tr - R), min(H, tr + R + 1)
            wc0, wc1 = max(0, tc - R), min(W, tc + R + 1)
            sub = mask[wr0:wr1, wc0:wc1]
            dsub = disk[wr0 - (tr - R):wr1 - (tr - R), wc0 - (tc - R):wc1 - (tc - R)]
            if np.any(sub & dsub):
                out[int(k)].add(int(iid))
    return out


def diagnose(run_dir: Path, gt_path: Path, rejection_log: Path,
             max_dist: float = 8.0, min_area: int = 15) -> dict:
    pred = iio.load_pred_instances(run_dir)
    gt, _ = iio.load_gt(gt_path, shape=pred.shape)
    gt_lab = gt.label_image()
    pred_lab = pred.label_image()
    frags = iio.load_fragments(run_dir, min_area=min_area)

    gt_field, gt_dist = _nearest_label_field(gt_lab)
    pred_field, pred_dist = _nearest_label_field(pred_lab)

    # Collapse the multi-pass log to one record per geometric candidate pair.
    pairs: Dict[Tuple, dict] = {}
    with open(rejection_log, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                br, bc = _fnum(row["base_tip_r"]), _fnum(row["base_tip_c"])
                tr, tc = _fnum(row["tar_tip_r"]), _fnum(row["tar_tip_c"])
            except KeyError:
                continue
            if any(np.isnan(v) for v in (br, bc, tr, tc)):
                continue
            key = tuple(sorted([(round(br), round(bc)), (round(tr), round(tc))]))
            rec = pairs.get(key)
            if rec is None:
                ga = _label_at(gt_field, gt_dist, br, bc, max_dist)
                gb = _label_at(gt_field, gt_dist, tr, tc, max_dist)
                pa = _label_at(pred_field, pred_dist, br, bc, max_dist)
                pb = _label_at(pred_field, pred_dist, tr, tc, max_dist)
                rec = {"gt_a": ga, "gt_b": gb, "pred_a": pa, "pred_b": pb,
                       "reasons": Counter(), "by_reason_metrics": {},
                       "tips": key, "accepted_stage": None}
                pairs[key] = rec
            reason = (row.get("reason") or "").strip()
            rec["reasons"][reason] += 1
            if reason == "accepted" and rec["accepted_stage"] is None:
                rec["accepted_stage"] = (row.get("stage") or "").strip()
            if reason not in rec["by_reason_metrics"]:
                fb, ft = _fnum(row.get("forward_base", "nan")), _fnum(row.get("forward_tar", "nan"))
                opp = _fnum(row.get("inward_opposition", "nan"))
                rec["by_reason_metrics"][reason] = {
                    "dist": _fnum(row.get("dist", "nan")),
                    "forward_min": np.nanmin([fb, ft]) if not (np.isnan(fb) and np.isnan(ft)) else float("nan"),
                    "opposition_pos": -opp if not np.isnan(opp) else float("nan"),
                    "line_resid": _fnum(row.get("line_resid", "nan")),
                    "curv_delta": _fnum(row.get("curv_delta", "nan")),
                    "arc_miss_frac": _fnum(row.get("arc_miss_frac", "nan")),
                    "intrusion_frac": _fnum(row.get("intrusion_frac", "nan")),
                    "smooth_rms": _fnum(row.get("smooth_rms", "nan")),
                    "max_turn_deg": _fnum(row.get("max_turn_deg", "nan")),
                    "width_ratio": _fnum(row.get("width_ratio", "nan")),
                    "layer_gap": _fnum(row.get("layer_gap", "nan")),
                }

    # Overlap-aware "joined" test (pred side only): the set of pred instances
    # touching each tip's neighborhood. The flat label image gives a disputed
    # crossing pixel to one layer, which made correctly-joined pairs look
    # severed (phantom accepted_then_lost / inflated missed buckets). GT and
    # wrong-merge attribution stay flat-nearest so buckets remain comparable
    # with the experiment history (see _tip_instance_sets).
    uniq_tips = sorted({t for rec in pairs.values() for t in rec["tips"]})
    tip_idx = {t: i for i, t in enumerate(uniq_tips)}
    pred_sets = _tip_instance_sets(pred, uniq_tips, max_dist)

    # Classify each unique candidate pair.
    missed_by_reason: Counter = Counter()
    missed_metric_vals: Dict[str, List[float]] = defaultdict(list)
    accepted_then_lost = 0
    wrong_by_stage: Counter = Counter()
    wrong_no_direct_log = 0
    missed_pairs: List[dict] = []
    wrong_pairs: List[dict] = []
    explained_boundaries: Dict[int, set] = defaultdict(set)

    for rec in pairs.values():
        ga, gb, pa, pb = rec["gt_a"], rec["gt_b"], rec["pred_a"], rec["pred_b"]
        tip_a, tip_b = rec["tips"]
        reasons = rec["reasons"]
        if ga == 0 or gb == 0:
            continue  # a tip not on any GT filament (spurious / background)
        # Joined = some pred instance covers BOTH tip neighborhoods. Overlap-
        # aware: at crossings the flat image gives the pixel to one layer only.
        joined = bool(pred_sets[tip_idx[tip_a]] & pred_sets[tip_idx[tip_b]])
        if ga == gb:
            if joined:
                continue  # same filament, spanned by one pred instance -> correctly joined
            if "accepted" in reasons:
                accepted_then_lost += 1
                continue
            # genuine miss: pick the dominant (most frequent) reject reason
            reason = reasons.most_common(1)[0][0]
            missed_by_reason[reason] += 1
            if pa and pb and pa != pb:
                explained_boundaries[ga].add(frozenset((pa, pb)))
            mcol = REASON_METRIC.get(reason, (None, None, None))[1]
            mval = rec["by_reason_metrics"].get(reason, {}).get(mcol, float("nan")) if mcol else float("nan")
            if mcol and not np.isnan(mval):
                missed_metric_vals[reason].append(float(mval))
            missed_pairs.append({"gt": int(ga), "pred_a": int(pa), "pred_b": int(pb),
                                 "reason": reason, "metric": mcol,
                                 "value": None if np.isnan(mval) else round(float(mval), 3),
                                 "tips": rec["tips"]})
        else:
            if pa != 0 and pa == pb:  # different filaments, same pred id -> wrong merge
                if "accepted" in reasons:
                    stage = rec["accepted_stage"] or "unknown"
                    wrong_by_stage[stage] += 1
                else:
                    wrong_no_direct_log += 1
                wrong_pairs.append({"gt_a": int(ga), "gt_b": int(gb), "pred": int(pa),
                                    "stage": rec["accepted_stage"], "tips": rec["tips"]})

    # never_candidate estimate: fragmentation not explained by any rejected join.
    # Overlap-aware fragment assignment (same convention as the metric).
    gt_of_frag = M.assign_fragments_to_instances(frags, gt)
    pred_of_frag = M.assign_fragments_to_instances(frags, pred)
    gt_to_preds: Dict[int, set] = defaultdict(set)
    for g, p in zip(gt_of_frag, pred_of_frag):
        if g != 0 and p != 0:
            gt_to_preds[int(g)].add(int(p))
    never_candidate = 0
    per_gt_frag = []
    for g, preds in gt_to_preds.items():
        needed = max(0, len(preds) - 1)
        explained = len(explained_boundaries.get(g, set()))
        nc = max(0, needed - explained)
        never_candidate += nc
        if needed > 0:
            per_gt_frag.append({"gt": g, "pred_ids": len(preds),
                                "explained_rejections": explained, "never_candidate": nc})

    # Per-reason metric medians next to configured gate thresholds.
    cfg_thr = _load_stage_thresholds(run_dir)
    reason_summary = []
    for reason, count in missed_by_reason.most_common():
        label, mcol, cfgkey = REASON_METRIC.get(reason, (reason, None, None))
        vals = missed_metric_vals.get(reason, [])
        median = round(float(np.median(vals)), 3) if vals else None
        thr = cfg_thr.get(cfgkey) if cfgkey else None
        reason_summary.append({"reason": reason, "label": label, "count": count,
                               "metric": mcol, "median": median, "gate_threshold": thr})

    per_gt_frag.sort(key=lambda d: -d["never_candidate"])
    return {
        "run": run_dir.name,
        "attribution": "overlap_aware_tip_sets",
        "n_candidate_pairs": len(pairs),
        "missed": {
            "total": int(sum(missed_by_reason.values())) + never_candidate,
            "by_gate": reason_summary,
            "never_candidate_est": never_candidate,
            "accepted_then_lost": accepted_then_lost,
        },
        "wrong": {
            "total": int(sum(wrong_by_stage.values())) + wrong_no_direct_log,
            "by_stage": dict(wrong_by_stage),
            "indirect_no_log": wrong_no_direct_log,
        },
        "top_fragmented_gt": per_gt_frag[:10],
        "missed_pairs": missed_pairs,
        "wrong_pairs": wrong_pairs,
    }


def _load_stage_thresholds(run_dir: Path) -> Dict[str, float]:
    """Pull gate thresholds from the run's active reconnect config (union of stages)."""
    try:
        import yaml
    except Exception:
        return {}
    cfg_path = None
    for cand in ("reconnect_config_active.yaml", "reconnect_config_scaled.yaml"):
        p = run_dir / cand
        if p.exists():
            cfg_path = p
            break
    if cfg_path is None:
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for stage in ("stage_relaxed", "stage_strict", "stage_clear"):
        thr = (cfg.get(stage, {}) or {}).get("thresholds", {}) or {}
        for k, v in thr.items():
            if isinstance(v, (int, float)):
                out.setdefault(k, v)  # first (relaxed) wins = the broadest gate
    return out


def _print_summary(d: dict) -> None:
    print(f"=== Root-cause diagnostic: {d['run']} ===")
    print(f"candidate pairs evaluated: {d['n_candidate_pairs']}")
    print(f"\nMISSED connections (under-merge): {d['missed']['total']}")
    print(f"  {'gate':<18}{'count':>6}  {'median':>8}  {'threshold':>10}")
    for r in d["missed"]["by_gate"]:
        med = "-" if r["median"] is None else f"{r['median']:.3f}"
        thr = "-" if r["gate_threshold"] is None else f"{r['gate_threshold']}"
        print(f"  {r['label']:<18}{r['count']:>6}  {med:>8}  {thr:>10}  ({r['metric']})")
    print(f"  {'never candidate*':<18}{d['missed']['never_candidate_est']:>6}   (est: no join ever evaluated)")
    if d["missed"]["accepted_then_lost"]:
        print(f"  {'accepted-then-lost':<18}{d['missed']['accepted_then_lost']:>6}   (merged then trimmed/relabeled apart)")
    print(f"\nWRONG connections (over-merge): {d['wrong']['total']}")
    for stage, n in d["wrong"]["by_stage"].items():
        print(f"  {stage:<22}{n:>6}")
    if d["wrong"]["indirect_no_log"]:
        print(f"  {'(merged via chain)':<22}{d['wrong']['indirect_no_log']:>6}")
    if d["top_fragmented_gt"]:
        print("\nMost fragmented GT filaments (gt: pred_ids / explained / never):")
        for g in d["top_fragmented_gt"][:6]:
            print(f"  gt {g['gt']}: {g['pred_ids']} ids  "
                  f"explained={g['explained_rejections']}  never={g['never_candidate']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconnect miss/wrong-merge root-cause diagnostic")
    ap.add_argument("--run", required=True, help="pipeline run folder")
    ap.add_argument("--gt", required=True, help="GT label image or centerline JSON")
    ap.add_argument("--rejection-log", default=None, dest="rejection_log",
                    help="rejection CSV (default: <run>/3.reconnect/rejection_log.csv)")
    ap.add_argument("--out", default=None, help="write JSON report")
    ap.add_argument("--max-dist", type=float, default=8.0, dest="max_dist",
                    help="max px from a logged tip to a label for attribution")
    ap.add_argument("--min-area", type=int, default=15, dest="min_area")
    args = ap.parse_args()

    run_dir = Path(args.run)
    log = Path(args.rejection_log) if args.rejection_log else run_dir / "3.reconnect" / "rejection_log.csv"
    if not log.exists():
        print(f"ERROR: rejection log not found: {log}\n"
              "Run the pipeline via eval/run_batch.py (it snapshots the log), or "
              "point --rejection-log at 3.reconnect/output/rejection_log_straight.csv",
              file=sys.stderr)
        return 2

    d = diagnose(run_dir, Path(args.gt), log, max_dist=args.max_dist, min_area=args.min_area)
    _print_summary(d)
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
