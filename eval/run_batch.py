"""Batch driver: run the pipeline over many samples and aggregate eval metrics.

For each sample folder containing <mask-name> + sem.png + gt_labels.tif, this:
  1. runs Tools/run_full_sem_pipeline.py,
  2. snapshots the per-run reconnect rejection log into the run folder,
  3. evaluates the output against gt_labels.tif (in-process),
and writes a per-sample CSV plus mean/median aggregates.

Usage
-----
    python eval/run_batch.py --samples output/synthetic_v2 --tag synthbatch
    python eval/run_batch.py --samples output/synthetic_v2 --no-run   # eval only

    # Run on the zero-degradation clean-skeleton mask instead of the
    # realistic degraded one (see eval/README.md "Clean skeleton mask"):
    python eval/run_batch.py --samples input/synthetic_v3/dense40 \
        --mask-name mask_clean.png --out-root output/full_pipeline_clean/dense40 \
        --tag clean_dense40

`--no-run` skips the pipeline and just (re)evaluates existing run folders, which
is handy after a long batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import instance_io as iio          # noqa: E402
import metrics as M                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path(r"C:\Repos\venv_cnt\Scripts\python.exe")
RUNNER = ROOT / "Tools" / "run_full_sem_pipeline.py"
# Reconnect writes its rejection log here (cwd-relative to 3.reconnect).
SHARED_REJECTION_LOG = ROOT / "3.reconnect" / "output" / "rejection_log_straight.csv"


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def run_pipeline(mask: Path, sem: Path, base: str, out_root: Path,
                 um_per_px: Optional[float]) -> Path:
    cmd = [
        _python(), str(RUNNER),
        "--mask", str(mask), "--background", str(sem),
        "--base", base, "--output-root", str(out_root),
    ]
    if um_per_px is not None:
        cmd += ["--um-per-px", str(um_per_px)]
    # Do NOT pre-create out_root/base: the runner appends a timestamp if the dir
    # already exists, which would desync the eval target. Capture stdout and
    # parse the actual run dir the runner reports.
    out_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (out_root / f"{base}_stdout.log").write_text(out_text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline failed for {base} (see {out_root / (base + '_stdout.log')})")
    run_dir = None
    for line in out_text.splitlines():
        if "final:" in line:
            run_dir = Path(line.split("final:", 1)[1].strip()).parent
    return run_dir if run_dir is not None else (out_root / base)


def snapshot_rejection_log(run_dir: Path) -> Optional[Path]:
    if SHARED_REJECTION_LOG.exists():
        dst = run_dir / "3.reconnect" / "rejection_log.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SHARED_REJECTION_LOG, dst)
        return dst
    return None


def eval_one(run_dir: Path, gt_path: Path, min_area: int) -> dict:
    pred = iio.load_pred_instances(run_dir)
    gt, gtmeta = iio.load_gt(gt_path, shape=pred.shape)
    frags = iio.load_fragments(run_dir, min_area=min_area)
    report = M.evaluate(pred, gt, frags, gt_kind=gtmeta["kind"])
    c = report.clustering
    row = {
        "sample": run_dir.name,
        "n_fragments": report.n_fragments,
        "gt_instances": c.n_gt_instances,
        "pred_instances": c.n_pred_instances,
        "precision": round(c.pairwise_precision, 4),
        "recall": round(c.pairwise_recall, 4),
        "f1": round(c.pairwise_f1, 4),
        "ari": round(c.adjusted_rand, 4),
        "voi": round(c.voi, 4),
        "voi_split": round(c.voi_split, 4),
        "voi_merge": round(c.voi_merge, 4),
        "gt_split": c.n_gt_split,
        "excess_splits": c.excess_splits,
        "pred_overmerged": c.n_pred_overmerged,
        "excess_merges": c.excess_merges,
    }
    if report.instance is not None:
        ir = report.instance
        row["well_recovered"] = ir.well_recovered
        row["gt_present"] = ir.n_gt_present
        row["recovery_rate"] = round(ir.recovery_rate, 4)
        row["filaments_deleted"] = ir.deleted
        row["false_pred"] = ir.false_pred
    if report.panoptic is not None:
        row["pq"] = round(report.panoptic.pq, 4)
    return row


def discover_samples(samples_dir: Path, mask_name: str = "mask.png") -> List[Path]:
    out = []
    for d in sorted(samples_dir.iterdir()):
        if d.is_dir() and (d / mask_name).exists() and (d / "gt_labels.tif").exists():
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FilaSeg batch run + eval aggregator")
    ap.add_argument("--samples", required=True,
                    help="dir of sample folders (<mask-name>, sem.png, gt_labels.tif)")
    ap.add_argument("--mask-name", default="mask.png", dest="mask_name",
                    help="mask filename within each sample dir (default mask.png; "
                         "use mask_clean.png for the zero-degradation skeleton variant)")
    ap.add_argument("--out-root", default=str(ROOT / "output" / "full_pipeline"))
    ap.add_argument("--tag", default="batch", help="name for the aggregate report")
    ap.add_argument("--no-run", action="store_true", help="skip pipeline, eval only")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0=all)")
    ap.add_argument("--min-area", type=int, default=15, dest="min_area")
    args = ap.parse_args()

    samples = discover_samples(Path(args.samples), args.mask_name)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        print(f"no samples with {args.mask_name}+gt_labels.tif under {args.samples}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root)
    rows: List[dict] = []
    for i, s in enumerate(samples, 1):
        base = s.name
        gt_path = s / "gt_labels.tif"
        um_per_px = None
        meta = s / "gt_meta.json"
        if meta.exists():
            try:
                um_per_px = float(json.loads(meta.read_text())["params"]["um_per_px"])
            except Exception:
                um_per_px = None
        print(f"[{i}/{len(samples)}] {base} ...", flush=True)
        run_dir = out_root / base
        if not args.no_run:
            run_dir = run_pipeline(s / args.mask_name, s / "sem.png", base, out_root, um_per_px)
            snapshot_rejection_log(run_dir)
        try:
            row = eval_one(run_dir, gt_path, args.min_area)
        except Exception as e:
            print(f"    eval failed: {e}", file=sys.stderr)
            continue
        rows.append(row)
        inst = (f" | inst {row.get('well_recovered','?')}/{row.get('gt_present','?')} "
                f"well, deleted={row.get('filaments_deleted','?')}"
                if 'well_recovered' in row else "")
        print(f"    F1={row['f1']} P={row['precision']} R={row['recall']} "
              f"splits={row['gt_split']} merges={row['pred_overmerged']} "
              f"({row['pred_instances']} pred / {row['gt_instances']} GT){inst}")

    if not rows:
        print("no successful evaluations", file=sys.stderr)
        return 1

    # Write per-sample CSV.
    csv_path = out_root / f"_eval_{args.tag}.csv"
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Aggregate.
    def agg(metric):
        vals = [r[metric] for r in rows if metric in r]
        return (statistics.mean(vals), statistics.median(vals)) if vals else (0, 0)

    print("\n=== AGGREGATE over", len(rows), "samples ===")
    for m in ("f1", "precision", "recall", "ari", "voi_split", "voi_merge"):
        mean, med = agg(m)
        print(f"  {m:12s} mean={mean:.3f}  median={med:.3f}")
    tot_split = sum(r["gt_split"] for r in rows)
    tot_gt = sum(r["gt_instances"] for r in rows)
    tot_merge = sum(r["pred_overmerged"] for r in rows)
    print(f"  fragmented GT filaments: {tot_split}/{tot_gt} "
          f"({100.0 * tot_split / max(1, tot_gt):.0f}%)")
    print(f"  over-merged pred ids:    {tot_merge}")
    if any("well_recovered" in r for r in rows):
        tot_well = sum(r.get("well_recovered", 0) for r in rows)
        tot_present = sum(r.get("gt_present", 0) for r in rows)
        tot_deleted = sum(r.get("filaments_deleted", 0) for r in rows)
        tot_false = sum(r.get("false_pred", 0) for r in rows)
        print(f"  --- instance-level (co-primary) ---")
        print(f"  well-recovered filaments: {tot_well}/{tot_present} "
              f"({100.0 * tot_well / max(1, tot_present):.0f}%)")
        print(f"  filaments deleted (swallowed by a wrong-connect): {tot_deleted}")
        print(f"  false (spurious) pred ids: {tot_false}")
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
