"""
reconnect_debug.py — inspection and log-analysis utilities.

Subcommands
-----------
inspect  <labels.tif> <id[,id,...]>
    Print area, bbox, skeleton tips, and PCA axis angle for each label id.

grep-log <log.csv> <r1> <c1> <r2> <c2> [tol=4]
    Find rejection-log rows where the tip-pair coordinates are within `tol`
    pixels of (r1,c1)<->(r2,c2) in either order.

log-summary <log.csv>
    Print total evaluations, accepted counts, override stats, and per-reason
    breakdown.

Examples
--------
python reconnect_debug.py inspect v7/labels.tif "10,7,3"
python reconnect_debug.py grep-log output/reconnect_rejection_log.csv 178 132 175 126 5
python reconnect_debug.py log-summary output/reconnect_rejection_log.csv
"""

import csv
import sys
from collections import Counter

import numpy as np
from skimage import io
from skimage.morphology import skeletonize


# ── inspect ──────────────────────────────────────────────────────────────────

def cmd_inspect(argv):
    if len(argv) < 2:
        sys.exit("usage: inspect <labels.tif> <id[,id,...]>")
    lbl_path, ids_str = argv[0], argv[1]
    ids = [int(x) for x in ids_str.split(",")]
    lbl = io.imread(lbl_path).astype(np.int32)
    print(f"shape={lbl.shape}, unique ids={np.unique(lbl)[1:].size}")
    for cid in ids:
        print(_analyze_id(lbl, cid))


def _analyze_id(lbl, cid):
    m = (lbl == cid)
    if not m.any():
        return f"id{cid}: NOT FOUND"
    pts = np.argwhere(m)
    r0, c0 = pts.min(0)
    r1, c1 = pts.max(0)
    sk = skeletonize(m)
    sk_pts = np.argwhere(sk)
    tips = []
    for r, c in sk_pts:
        y0, y1 = max(0, r - 1), min(sk.shape[0], r + 2)
        x0, x1 = max(0, c - 1), min(sk.shape[1], c + 2)
        if int(sk[y0:y1, x0:x1].sum()) - 1 == 1:
            tips.append((int(r), int(c)))
    xy = sk_pts.astype(np.float32)
    xy -= xy.mean(0)
    C = (xy.T @ xy) / max(1, xy.shape[0] - 1)
    w, v = np.linalg.eigh(C)
    d = v[:, int(np.argmax(w))]
    ang = float(np.degrees(np.arctan2(d[0], d[1])))
    return (
        f"id{cid}: area={int(m.sum())}, bbox=r[{r0}:{r1}] c[{c0}:{c1}], "
        f"tips={tips}, axis={ang:+.1f}deg"
    )


# ── grep-log ──────────────────────────────────────────────────────────────────

def cmd_grep_log(argv):
    if len(argv) < 5:
        sys.exit("usage: grep-log <log.csv> <r1> <c1> <r2> <c2> [tol=4]")
    path = argv[0]
    r1, c1, r2, c2 = [float(x) for x in argv[1:5]]
    tol = float(argv[5]) if len(argv) > 5 else 4.0

    def near(a, b):
        return abs(a - b) <= tol

    hits = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                br = float(row["base_tip_r"]); bc = float(row["base_tip_c"])
                tr = float(row["tar_tip_r"]);  tc = float(row["tar_tip_c"])
            except (KeyError, ValueError):
                continue
            if (near(br, r1) and near(bc, c1) and near(tr, r2) and near(tc, c2)) or \
               (near(br, r2) and near(bc, c2) and near(tr, r1) and near(tc, c1)):
                hits.append(row)

    print(f"hits={len(hits)} for ({r1},{c1})<->({r2},{c2}) tol={tol}")
    keys = ("stage", "base_id", "tar_id", "reason", "dist",
            "forward_base", "forward_tar", "inward_opposition",
            "line_resid", "arc_miss_px", "curv_delta",
            "intrusion_frac", "smooth_rms", "max_turn_deg",
            "eff_forward_cos", "eff_opposition", "eff_max_turn",
            "arc_ok", "line_ok", "clear_merge")
    for row in hits:
        def fmt(k):
            v = row.get(k, "")
            try: return f"{float(v):.2f}"
            except (ValueError, TypeError): return v
        print(
            f"  [{fmt('stage')}] ids={fmt('base_id')},{fmt('tar_id')} "
            f"REASON={fmt('reason'):22s} dist={fmt('dist')} "
            f"fwd=({fmt('forward_base')},{fmt('forward_tar')}) opp={fmt('inward_opposition')} "
            f"line={fmt('line_resid')} arc={fmt('arc_miss_px')} "
            f"curv_d={fmt('curv_delta')} turn={fmt('max_turn_deg')}/{fmt('eff_max_turn')} "
            f"smooth={fmt('smooth_rms')} intrusion={fmt('intrusion_frac')} "
            f"eff_fwd={fmt('eff_forward_cos')} eff_opp={fmt('eff_opposition')} "
            f"arc_ok={fmt('arc_ok')} line_ok={fmt('line_ok')} clear={fmt('clear_merge')}"
        )


# ── log-summary ───────────────────────────────────────────────────────────────

def cmd_log_summary(argv):
    if not argv:
        sys.exit("usage: log-summary <log.csv>")
    with open(argv[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    accepted = [r for r in rows if r.get("reason") == "accepted"]
    override = [r for r in accepted if r.get("clear_merge") == "1"]
    print(f"total evaluations : {len(rows)}")
    print(f"accepted          : {len(accepted)}")
    print(f"  via clear_merge : {len(override)}")
    print(f"  normally        : {len(accepted) - len(override)}")
    print()
    print("reason counts:")
    for reason, count in sorted(Counter(r["reason"] for r in rows).items(), key=lambda x: -x[1]):
        print(f"  {reason:<30s} {count}")


# ── dispatch ──────────────────────────────────────────────────────────────────

COMMANDS = {
    "inspect":     cmd_inspect,
    "grep-log":    cmd_grep_log,
    "log-summary": cmd_log_summary,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])
