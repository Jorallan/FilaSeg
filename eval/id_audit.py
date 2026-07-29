"""Per-instance (id-by-id) audit: GT filament -> pred id(s), with status + root cause.

The primary metric (fragment-clustering pairwise F1) is permutation-invariant and
tells you *how much* over/under-merge there is, but not *which* filament failed or
*why*. This tool produces the missing per-id sheet:

For every GT filament it reports the predicted id(s) its fragments landed in and a
status:

  OK             one pred id, not shared with any other GT          (perfect)
  SPLIT          fragments scattered across >1 pred id              (under-merge)
  MERGED         its pred id also holds another GT's fragments      (over-merge)
  SPLIT+MERGED   both of the above
  DROPPED        no fragment survived into any pred id              (lost upstream)

Then it attaches a ROOT CAUSE traced to a stage, by joining the reconnect
rejection log (via diagnose_connections):

  SPLIT  -> the gate that rejected the join (stage 3 reconnect), or
            "never_candidate" = the join was never even proposed, which means the
            two pieces were not facing tips within search range => the break was
            introduced in stage 1/2 (skeleton/preprocess fragmentation).
  MERGED -> the stage that accepted the wrong join (clear/strict/relaxed/repair),
            or a postprocess overlap-absorb chain (stage 4).

Usage
-----
    python eval/id_audit.py --run <run_dir> --gt <gt.npz|tif|json> --name <label>
    python eval/id_audit.py --batch   # all 4 audit cases, writes the combined sheet
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import instance_io as iio          # noqa: E402
import metrics as M                # noqa: E402
import diagnose_connections as DC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "audit_out"


# ── per-GT contingency ────────────────────────────────────────────────────


def _per_gt_rows(gt_frag: np.ndarray, pred_frag: np.ndarray,
                 gt_ids: List[int]) -> Tuple[List[dict], Dict[int, set]]:
    """Build one row per GT id from the fragment->instance assignments.

    gt_frag/pred_frag are aligned int vectors over the same fragment set.
    Returns (rows, pred_to_gts) where pred_to_gts[p] = set of GT ids sharing p.
    """
    gt_to_pred_counts: Dict[int, Counter] = defaultdict(Counter)
    pred_to_gts: Dict[int, set] = defaultdict(set)
    for g, p in zip(gt_frag, pred_frag):
        if g == 0:
            continue
        gt_to_pred_counts[int(g)][int(p)] += 1
        if p != 0:
            pred_to_gts[int(p)].add(int(g))

    rows = []
    for g in gt_ids:
        counts = gt_to_pred_counts.get(g, Counter())
        total = sum(counts.values())
        nonzero = {p: c for p, c in counts.items() if p != 0}
        n_pred = len(nonzero)
        dom_pred = max(nonzero, key=nonzero.get) if nonzero else 0
        # other GTs sharing this GT's dominant pred id (over-merge partners)
        share = sorted(pred_to_gts.get(dom_pred, set()) - {g}) if dom_pred else []
        if total == 0 or n_pred == 0:
            status = "DROPPED"
        else:
            split = n_pred > 1
            merged = len(share) > 0
            status = ("SPLIT+MERGED" if split and merged else
                      "SPLIT" if split else
                      "MERGED" if merged else "OK")
        rows.append({
            "gt_id": g,
            "n_frags": total,
            "n_pred_ids": n_pred,
            "dom_pred": dom_pred,
            "frags_in_dom": nonzero.get(dom_pred, 0),
            "shares_pred_with": share,
            "status": status,
        })
    return rows, pred_to_gts


# ── join with the reconnect rejection log for root cause ────────────────────


def _causes_from_diag(diag: dict) -> Tuple[Dict[int, Counter], Dict[int, set], Dict[int, Counter]]:
    """From a diagnose() report, group causes by GT id.

    Returns:
      split_gates[g]    Counter of gate-reason -> count for missed joins of GT g
      split_boundaries  carries never_candidate count via top_fragmented_gt
      merge_stages[g]   Counter of accepting-stage -> count for wrong merges on g
    """
    split_gates: Dict[int, Counter] = defaultdict(Counter)
    for mp in diag.get("missed_pairs", []):
        split_gates[int(mp["gt"])][mp["reason"]] += 1

    never_by_gt: Dict[int, int] = {}
    for tf in diag.get("top_fragmented_gt", []):
        never_by_gt[int(tf["gt"])] = int(tf.get("never_candidate", 0))

    merge_stages: Dict[int, Counter] = defaultdict(Counter)
    for wp in diag.get("wrong_pairs", []):
        # No direct accept row => the two filaments were fused transitively by a
        # chain of pairwise reconnect joins (A-B, B-C => A,B,C), NOT by a single
        # gate and NOT by postprocess (thin-vs-final shows stage 4 adds ~0).
        stage = wp.get("stage") or "transitive(stage3)"
        merge_stages[int(wp["gt_a"])][stage] += 1
        merge_stages[int(wp["gt_b"])][stage] += 1

    return split_gates, never_by_gt, merge_stages


# never_candidate / accepted_then_lost map to a pipeline stage:
def _stage_for(cause: str) -> str:
    if cause == "never_candidate":
        return "stage1/2 (fragmentation: no facing tips in range)"
    if cause in ("accepted_then_lost",):
        return "stage4 postprocess (trim/relabel split a merge)"
    return "stage3 reconnect gate"


def audit_one(run_dir: Path, gt_path: Path, name: str,
              min_area: int = 15) -> dict:
    pred = iio.load_pred_instances(run_dir)
    gt, gtmeta = iio.load_gt(gt_path, shape=pred.shape)
    frags = iio.load_fragments(run_dir, min_area=min_area)

    # overlap-aware fragment assignment (matches the primary metric)
    gt_frag = M.assign_fragments_to_instances(frags, gt)
    pred_frag = M.assign_fragments_to_instances(frags, pred)
    gt_ids = sorted(gt.masks)
    rows, pred_to_gts = _per_gt_rows(gt_frag, pred_frag, gt_ids)

    # headline metric
    rep = M.evaluate(pred, gt, frags, gt_kind=gtmeta["kind"], do_panoptic=False)
    c = rep.clustering

    # root cause via the rejection log (if present)
    log = run_dir / "3.reconnect" / "rejection_log.csv"
    diag = None
    if log.exists():
        diag = DC.diagnose(run_dir, gt_path, log, min_area=min_area)
        split_gates, never_by_gt, merge_stages = _causes_from_diag(diag)
    else:
        split_gates, never_by_gt, merge_stages = {}, {}, {}

    # attach cause/stage to each failing row
    for r in rows:
        g = r["gt_id"]
        causes: List[str] = []
        if r["status"] in ("SPLIT", "SPLIT+MERGED", "DROPPED"):
            gates = split_gates.get(g, Counter())
            nc = never_by_gt.get(g, 0)
            if gates:
                top = gates.most_common(2)
                causes += [f"gate:{k}x{v}" for k, v in top]
            if nc > 0:
                causes.append(f"never_candidate x{nc} -> stage1/2")
            if not gates and nc == 0 and r["status"] != "OK":
                causes.append("upstream/unlogged -> stage1/2")
        if r["status"] in ("MERGED", "SPLIT+MERGED"):
            stages = merge_stages.get(g, Counter())
            if stages:
                causes += [f"merge@{k}x{v}" for k, v in stages.most_common(2)]
            else:
                causes.append("merge: transitive(stage3)")
        r["root_cause"] = "; ".join(causes) if causes else ""

    status_counts = Counter(r["status"] for r in rows)
    # false-positive pred ids: pred instances no GT fragment mapped to
    pred_ids = set(pred.masks)
    used_pred = {r["dom_pred"] for r in rows if r["dom_pred"]}
    fp_pred = sorted(pred_ids - set().union(*pred_to_gts.keys()) if False else pred_ids - set(pred_to_gts.keys()))

    return {
        "name": name,
        "run": str(run_dir),
        "gt": str(gt_path),
        "n_gt": len(gt_ids),
        "n_pred": pred.n,
        "n_frags": len(frags),
        "f1": round(c.pairwise_f1, 3),
        "precision": round(c.pairwise_precision, 3),
        "recall": round(c.pairwise_recall, 3),
        "rows": rows,
        "status_counts": dict(status_counts),
        "fp_pred_ids": fp_pred,
        "diag": diag,
    }


# ── reporting ───────────────────────────────────────────────────────────────


def _print_case(res: dict) -> None:
    print(f"\n{'='*78}\nCASE: {res['name']}   F1={res['f1']}  P={res['precision']}  R={res['recall']}")
    print(f"  GT instances={res['n_gt']}  pred={res['n_pred']}  fragments={res['n_frags']}")
    sc = res["status_counts"]
    order = ["OK", "SPLIT", "MERGED", "SPLIT+MERGED", "DROPPED"]
    print("  status: " + "  ".join(f"{k}={sc.get(k,0)}" for k in order))
    print(f"  false-positive pred ids (no GT): {len(res['fp_pred_ids'])}")
    print(f"\n  {'gt':>4} {'frags':>5} {'npred':>5} {'dom':>4} {'status':<13} root_cause")
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*4} {'-'*13} {'-'*40}")
    for r in sorted(res["rows"], key=lambda x: (x["status"] == "OK", -x["n_pred_ids"], x["gt_id"])):
        if r["status"] == "OK":
            continue
        share = f" <shares {r['shares_pred_with']}>" if r["shares_pred_with"] else ""
        print(f"  {r['gt_id']:>4} {r['n_frags']:>5} {r['n_pred_ids']:>5} {r['dom_pred']:>4} "
              f"{r['status']:<13} {r['root_cause']}{share}")


def _write_csv(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_id", "n_frags", "n_pred_ids", "dom_pred", "frags_in_dom",
                    "shares_pred_with", "status", "root_cause"])
        for r in sorted(res["rows"], key=lambda x: x["gt_id"]):
            w.writerow([r["gt_id"], r["n_frags"], r["n_pred_ids"], r["dom_pred"],
                        r["frags_in_dom"], " ".join(map(str, r["shares_pred_with"])),
                        r["status"], r["root_cause"]])


def _write_markdown(results: List[dict], path: Path) -> None:
    L = ["# Id-by-id audit: pipeline output vs ground truth", ""]
    L.append("Per-GT-filament status and root cause across the validation cases. "
             "Status: OK / SPLIT (under-merge) / MERGED (over-merge) / DROPPED (lost upstream).")
    L.append("")
    # headline table
    L.append("| case | GT | pred | F1 | P | R | OK | SPLIT | MERGED | S+M | DROP | FP-pred |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for res in results:
        sc = res["status_counts"]
        L.append(f"| {res['name']} | {res['n_gt']} | {res['n_pred']} | {res['f1']} | "
                 f"{res['precision']} | {res['recall']} | {sc.get('OK',0)} | {sc.get('SPLIT',0)} | "
                 f"{sc.get('MERGED',0)} | {sc.get('SPLIT+MERGED',0)} | {sc.get('DROPPED',0)} | "
                 f"{len(res['fp_pred_ids'])} |")
    # aggregate cause breakdown (stage attribution)
    L += ["", "## Failure causes by stage (aggregated over all cases)", ""]
    stage_tally: Counter = Counter()
    gate_tally: Counter = Counter()
    for res in results:
        for r in res["rows"]:
            if r["status"] == "OK":
                continue
            rc = r["root_cause"]
            if "never_candidate" in rc or "stage1/2" in rc:
                stage_tally["stage1/2 fragmentation (never a candidate)"] += 1
            if rc.startswith("gate:") or "; gate:" in rc:
                stage_tally["stage3 reconnect gate rejected the join"] += 1
                for tok in rc.split(";"):
                    tok = tok.strip()
                    if tok.startswith("gate:"):
                        gate_tally[tok.split(":", 1)[1].split("x")[0]] += 1
            if "merge@stage" in rc:
                stage_tally["stage3 reconnect over-merge (direct gate accept)"] += 1
            if "transitive(stage3)" in rc:
                stage_tally["stage3 reconnect over-merge (transitive chain, no single gate)"] += 1
    for k, v in stage_tally.most_common():
        L.append(f"- **{v}** — {k}")
    if gate_tally:
        L += ["", "Reconnect gates most responsible for SPLITs:"]
        for k, v in gate_tally.most_common():
            L.append(f"  - `{k}`: {v}")
    # per-case failing rows
    for res in results:
        L += ["", f"## {res['name']} — failing filaments", "",
              "| gt | frags | npred | dom | status | shares | root cause |",
              "|--:|--:|--:|--:|---|---|---|"]
        any_fail = False
        for r in sorted(res["rows"], key=lambda x: (-x["n_pred_ids"], x["gt_id"])):
            if r["status"] == "OK":
                continue
            any_fail = True
            sh = " ".join(map(str, r["shares_pred_with"]))
            L.append(f"| {r['gt_id']} | {r['n_frags']} | {r['n_pred_ids']} | {r['dom_pred']} | "
                     f"{r['status']} | {sh} | {r['root_cause']} |")
        if not any_fail:
            L.append("| — | | | | (all OK) | | |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# ── batch over the 4 audit cases ────────────────────────────────────────────


AUDIT_CASES = [
    ("real_crop", ROOT / "output/audit/real_crop",
     ROOT / "output/full_pipeline/sem00000_crop512_manual/modified/sem_full_00000_1p66_crop512_manual_multilabel.npz"),
    ("synth_0001", ROOT / "output/audit/synth_0001", ROOT / "output/synthetic_val/synth_0001/gt_multilabel.npz"),
    ("synth_0002", ROOT / "output/audit/synth_0002", ROOT / "output/synthetic_val/synth_0002/gt_multilabel.npz"),
    ("synth_0003", ROOT / "output/audit/synth_0003", ROOT / "output/synthetic_val/synth_0003/gt_multilabel.npz"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-instance id-by-id audit with root cause")
    ap.add_argument("--run")
    ap.add_argument("--gt")
    ap.add_argument("--name", default="case")
    ap.add_argument("--batch", action="store_true", help="run all 4 audit cases")
    ap.add_argument("--min-area", type=int, default=15, dest="min_area")
    args = ap.parse_args()

    cases = AUDIT_CASES if args.batch else [(args.name, Path(args.run), Path(args.gt))]
    results = []
    for name, run, gt in cases:
        if not Path(run).exists():
            print(f"SKIP {name}: run not found {run}", file=sys.stderr)
            continue
        res = audit_one(Path(run), Path(gt), name, min_area=args.min_area)
        _print_case(res)
        _write_csv(res, OUT_DIR / f"{name}_ids.csv")
        results.append(res)

    if results:
        _write_markdown(results, OUT_DIR / "audit_summary.md")
        print(f"\nwrote {OUT_DIR / 'audit_summary.md'} and per-case CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
