"""FilaSeg reconnect-quality evaluation CLI.

Measures how well the pipeline's instance grouping matches ground truth, using
the fragment-clustering metric as the primary objective (see metrics.py).

Usage
-----
Describe a run (no GT needed) - validates loaders, shows over-seg magnitude:
    python eval/eval_reconnect.py --describe --run output/full_pipeline/<base>

Evaluate against ground truth (label image or centerline JSON):
    python eval/eval_reconnect.py --run output/full_pipeline/<base> --gt gt.tif
    python eval/eval_reconnect.py --run output/full_pipeline/<base> --gt gt.json --out report.json

Validate the metric engine itself on synthetic known cases:
    python eval/eval_reconnect.py --selftest

Ground-truth formats
--------------------
- Label image (.tif/.png, single channel): 0 = background, >0 = filament id.
- Centerline JSON:
    {"image_shape_rc": [H, W], "default_width_px": 5.0,
     "filaments": [{"id": 1, "centerline_xy": [[x,y], ...], "width_px": 8.0}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)

import instance_io as iio          # noqa: E402
import metrics as M                # noqa: E402


def cmd_describe(args) -> int:
    pred = iio.load_pred_instances(Path(args.pred or args.run))
    frags = iio.load_fragments(Path(args.run), min_area=args.min_area)
    print(f"pred artifact : {iio.find_pred_artifact(Path(args.pred or args.run))}")
    print(f"pred instances: {pred.n}")
    print(f"fragments     : {len(frags)} (pre-reconnect pieces, min_area={args.min_area})")
    if pred.n and frags:
        ratio = len(frags) / max(1, pred.n)
        print(f"fragments/instance = {ratio:.2f}  "
              f"(>~1 means reconnect merged little; ~1 means it barely grouped)")
    return 0


def cmd_evaluate(args) -> int:
    pred = iio.load_pred_instances(Path(args.pred or args.run))
    gt, gtmeta = iio.load_gt(Path(args.gt), shape=pred.shape)
    if gt.shape != pred.shape:
        print(f"ERROR: GT shape {gt.shape} != pred shape {pred.shape}", file=sys.stderr)
        return 2
    frags = iio.load_fragments(Path(args.run), min_area=args.min_area)
    if not frags:
        print("ERROR: no fragments found (need 2.preprocess/branches or a "
              "merge/reconstructed mask under the run dir).", file=sys.stderr)
        return 2

    report = M.evaluate(
        pred, gt, frags,
        gt_kind=gtmeta["kind"],
        min_overlap_frac=args.min_overlap_frac,
        max_dist_px=args.max_dist_px,
        do_panoptic=not args.no_panoptic,
    )
    print(report.summary())
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


# ── self-test ─────────────────────────────────────────────────────────────


def _make_case(pred_groups):
    """Build a 64x64 synthetic case.

    GT: 4 filaments, each = 3 horizontal fragments -> 12 fragments.
    `pred_groups[f]` lists the pred id assigned to each of filament f's 3
    fragments, letting us script perfect / split / merged partitions.
    """
    H, W = 64, 64
    fragments = []
    gt_lab = np.zeros((H, W), dtype=np.int32)
    pred_lab = np.zeros((H, W), dtype=np.int32)
    for f in range(4):
        row = 8 + f * 14
        for s in range(3):
            c0 = 4 + s * 20
            frag = np.zeros((H, W), dtype=bool)
            frag[row:row + 4, c0:c0 + 16] = True
            fragments.append(frag)
            gt_lab[frag] = f + 1
            pred_lab[frag] = pred_groups[f][s]
    return fragments, gt_lab, pred_lab


def _run_selftest() -> int:
    from instance_io import instances_from_label_image

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # Case A: perfect grouping (each filament -> one consistent pred id).
    frags, gt, pred = _make_case({0: [1, 1, 1], 1: [2, 2, 2], 2: [3, 3, 3], 3: [4, 4, 4]})
    gf = M.assign_fragments_to_labels(frags, gt)
    pf = M.assign_fragments_to_labels(frags, pred)
    r = M.fragment_clustering_metrics(gf, pf)
    print("Case A (perfect):")
    check("all 12 fragments assigned a GT label", int((gf != 0).sum()) == 12)
    check("precision == 1", abs(r.pairwise_precision - 1.0) < 1e-9)
    check("recall == 1", abs(r.pairwise_recall - 1.0) < 1e-9)
    check("ARI == 1", abs(r.adjusted_rand - 1.0) < 1e-9)
    check("VOI == 0", r.voi < 1e-9)
    check("no splits / merges", r.n_gt_split == 0 and r.n_pred_overmerged == 0)

    # Case B: pure under-merge (each filament split into 2 pred ids).
    frags, gt, pred = _make_case({0: [1, 1, 9], 1: [2, 2, 10], 2: [3, 3, 11], 3: [4, 4, 12]})
    gf = M.assign_fragments_to_labels(frags, gt)
    pf = M.assign_fragments_to_labels(frags, pred)
    r = M.fragment_clustering_metrics(gf, pf)
    print("Case B (under-merge / over-seg):")
    check("precision == 1 (no false joins)", abs(r.pairwise_precision - 1.0) < 1e-9)
    check("recall < 1 (missed joins)", r.pairwise_recall < 1.0)
    check("4 GT filaments split", r.n_gt_split == 4)
    check("voi_merge == 0", r.voi_merge < 1e-9)
    check("voi_split > 0", r.voi_split > 0)

    # Case C: pure over-merge (filaments 1+2 fused, 3+4 fused).
    frags, gt, pred = _make_case({0: [1, 1, 1], 1: [1, 1, 1], 2: [2, 2, 2], 3: [2, 2, 2]})
    gf = M.assign_fragments_to_labels(frags, gt)
    pf = M.assign_fragments_to_labels(frags, pred)
    r = M.fragment_clustering_metrics(gf, pf)
    print("Case C (over-merge / under-seg):")
    check("recall == 1 (nothing missed)", abs(r.pairwise_recall - 1.0) < 1e-9)
    check("precision < 1 (false joins)", r.pairwise_precision < 1.0)
    check("2 pred ids over-merged", r.n_pred_overmerged == 2)
    check("voi_split == 0", r.voi_split < 1e-9)
    check("voi_merge > 0", r.voi_merge > 0)

    # Panoptic sanity on the perfect case.
    frags, gt, pred = _make_case({0: [1, 1, 1], 1: [2, 2, 2], 2: [3, 3, 3], 3: [4, 4, 4]})
    pq = M.panoptic_quality(instances_from_label_image(pred), instances_from_label_image(gt))
    print("Panoptic (perfect case):")
    check("PQ == 1", abs(pq.pq - 1.0) < 1e-9)
    check("TP == 4, FP == 0, FN == 0", pq.tp == 4 and pq.fp == 0 and pq.fn == 0)

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="FilaSeg reconnect-quality evaluation")
    ap.add_argument("--run", help="pipeline run folder (output/full_pipeline/<base>)")
    ap.add_argument("--pred", help="override predicted-instance file/folder")
    ap.add_argument("--gt", help="ground-truth label image or centerline JSON")
    ap.add_argument("--out", help="write JSON report to this path")
    ap.add_argument("--describe", action="store_true",
                    help="load pred + fragments and report counts (no GT)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the metric engine on synthetic cases")
    ap.add_argument("--min-area", type=int, default=15, dest="min_area",
                    help="min fragment area (px)")
    ap.add_argument("--min-overlap-frac", type=float, default=0.2, dest="min_overlap_frac",
                    help="fragment direct-overlap fraction before nearest-label fallback")
    ap.add_argument("--max-dist-px", type=float, default=12.0, dest="max_dist_px",
                    help="max distance for nearest-label fallback assignment")
    ap.add_argument("--no-panoptic", action="store_true", help="skip pixel PQ")
    args = ap.parse_args()

    if args.selftest:
        return _run_selftest()
    if args.describe:
        if not args.run:
            ap.error("--describe needs --run")
        return cmd_describe(args)
    if args.run and args.gt:
        return cmd_evaluate(args)
    ap.error("provide --selftest, or --describe --run, or --run --gt")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
