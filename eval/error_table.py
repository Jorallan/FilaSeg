"""Categorized, actionable error table over the validation runs.

For each case it classifies every error into a workable category, separates GROSS
errors from minor crossing LEAKS (which the pairwise-F1 metric over-penalizes),
and links each wrong-connect to the instance it damaged.

UNDER-MERGE (a GT filament not assembled into one id), by why it broke:
  U1_crossing_split  pieces touch/overlap at a crossing (facing gap <=12px)
  U2_wide_break      pieces separated by a real gap (>40px) — bridging evidence
  U3_mid_break       12-40px gap (borderline)
  U4_dropped         no surviving fragment (small/faint) — lost stage1/2
  U5_minor_leak      dominant id holds >=80% of the filament; the "split" is a
                     few leaked fragments, not a real break (metric-sensitive)

OVER-MERGE (a pred id fusing >=2 GT filaments), by what happened + severity:
  O1_wrong_bridge    two SEPARATE filaments joined across a gap (GT disjoint)
  O2_crossing_fusion two truly-crossing filaments substantially fused
  O3_crossing_leak   only a few fragments of the 2nd filament leaked in
                     (minority share <20%) — metric-sensitive, not a gross fusion

Causal link: for each over-merge (predP: gA dominant + gB minority) we report
gB's fate — fully absorbed into gA (gB lost as a distinct id) or partially
leaked (gB still has its own id elsewhere).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_io as iio   # noqa: E402
import metrics as M         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "error_table_out"
CASES = [
    ("real_crop", ROOT / "output/audit/real_crop",
     ROOT / "output/full_pipeline/sem00000_crop512_manual/modified/sem_full_00000_1p66_crop512_manual_multilabel.npz"),
    ("synth_0001", ROOT / "output/audit/synth_0001", ROOT / "output/synthetic_val/synth_0001/gt_multilabel.npz"),
    ("synth_0002", ROOT / "output/audit/synth_0002", ROOT / "output/synthetic_val/synth_0002/gt_multilabel.npz"),
    ("synth_0003", ROOT / "output/audit/synth_0003", ROOT / "output/synthetic_val/synth_0003/gt_multilabel.npz"),
]
LEAK_FRAC = 0.20    # minority share below this = a leak, not a gross error
RECOVER_FRAC = 0.80 # dominant id holds this much of a GT = basically recovered


def local_angle(mask, center, r=14):
    rr, cc = center
    sub = np.zeros_like(mask)
    r0, r1 = max(0, rr - r), min(mask.shape[0], rr + r + 1)
    c0, c1 = max(0, cc - r), min(mask.shape[1], cc + r + 1)
    sub[r0:r1, c0:c1] = mask[r0:r1, c0:c1]
    pts = np.argwhere(sub)
    if pts.shape[0] < 2:
        return None
    pts = pts - pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    mj = v[:, int(np.argmax(w))]
    return np.degrees(np.arctan2(mj[0], mj[1])) % 180.0


def pair_geom(gt, ga, gb):
    """Return (type, angle, gap) for a fused/over-merged GT pair."""
    ma, mb = gt.masks[ga], gt.masks[gb]
    if (ma & mb).any():
        ctr = tuple(int(v) for v in np.argwhere(ma & mb).mean(0))
        typ, gap = "crossing", 0.0
    else:
        d = distance_transform_edt(~mb)
        apts = np.argwhere(ma)
        gap = float(d[ma].min())
        ctr = tuple(int(v) for v in apts[int(np.argmin(d[ma]))])
        typ = "bridge"
    aa, ab = local_angle(ma, ctr), local_angle(mb, ctr)
    ang = None
    if aa is not None and ab is not None:
        ang = abs(aa - ab); ang = min(ang, 180 - ang)
    return typ, ang, gap


def split_gap(frags, idxA, idxB, shape):
    mA = np.zeros(shape, bool)
    for k in idxA:
        mA |= frags[k]
    mB = np.zeros(shape, bool)
    for k in idxB:
        mB |= frags[k]
    if (mA & mB).any():
        return 0.0
    return float(distance_transform_edt(~mB)[mA].min())


def analyze(name, run, gtp):
    pred = iio.load_pred_instances(run)
    gt, _ = iio.load_gt(gtp, shape=pred.shape)
    frags = iio.load_fragments(run, min_area=15)
    shape = pred.shape
    gf = M.assign_fragments_to_instances(frags, gt)
    pf = M.assign_fragments_to_instances(frags, pred)

    cell = defaultdict(list)              # (g,p) -> frag idxs
    gt_tot = defaultdict(int)
    pred_tot = defaultdict(int)
    for k, (g, p) in enumerate(zip(gf, pf)):
        if g and p:
            cell[(int(g), int(p))].append(k)
            gt_tot[int(g)] += 1
            pred_tot[int(p)] += 1

    gt_dom_pred = {}
    for g in gt.masks:
        ps = {p: len(cell[(g, p)]) for (gg, p) in cell if gg == g}
        gt_dom_pred[g] = max(ps, key=ps.get) if ps else 0
    pred_dom_gt = {}
    for p in pred.masks:
        gs = {g: len(cell[(g, p)]) for (g, pp) in cell if pp == p}
        pred_dom_gt[p] = max(gs, key=gs.get) if gs else 0

    # ---- UNDER-MERGE rows (per GT) ----
    gt_rows = []
    ucat = defaultdict(int)
    for g in sorted(gt.masks):
        preds = sorted({p for (gg, p) in cell if gg == g})
        tot = gt_tot[g]
        if tot == 0:
            gt_rows.append({"gt": g, "n_frags": 0, "n_pred": 0, "dom_pred": 0,
                            "recovery": 0.0, "category": "U4_dropped", "detail": "no surviving fragment"})
            ucat["U4_dropped"] += 1
            continue
        dom = gt_dom_pred[g]
        recovery = len(cell[(g, dom)]) / tot
        if len(preds) == 1:
            continue  # fully recovered, not an under-merge
        if recovery >= RECOVER_FRAC:
            cat, detail = "U5_minor_leak", f"dom id holds {recovery*100:.0f}%; {len(preds)-1} leaked frag-group(s)"
        else:
            # classify the worst split: gap between dom and the next-biggest pred
            others = sorted([p for p in preds if p != dom], key=lambda p: -len(cell[(g, p)]))
            p2 = others[0]
            gap = split_gap(frags, cell[(g, dom)], cell[(g, p2)], shape)
            if gap <= 12:
                cat = "U1_crossing_split"
            elif gap > 40:
                cat = "U2_wide_break"
            else:
                cat = "U3_mid_break"
            detail = f"dom {recovery*100:.0f}%, split into {len(preds)} ids, worst gap {gap:.0f}px"
        ucat[cat] += 1
        gt_rows.append({"gt": g, "n_frags": tot, "n_pred": len(preds), "dom_pred": dom,
                        "recovery": round(recovery, 2), "category": cat, "detail": detail})

    # ---- OVER-MERGE rows (per pred) + causal links ----
    pred_rows = []
    ocat = defaultdict(int)
    o_absorbed = 0   # wrong-connect that DELETED a filament (gB lost as distinct id)
    o_survives = 0   # wrong-connect where gB still has its own id (partial leak)
    links = []
    for p in sorted(pred.masks):
        gts = sorted({g for (g, pp) in cell if pp == p})
        if len(gts) < 2:
            continue
        dom = pred_dom_gt[p]
        for gb in gts:
            if gb == dom:
                continue
            share_in_pred = len(cell[(gb, p)]) / max(1, pred_tot[p])
            share_of_gb = len(cell[(gb, p)]) / max(1, gt_tot[gb])
            typ, ang, gap = pair_geom(gt, dom, gb)
            if share_of_gb < LEAK_FRAC:
                cat = "O3_crossing_leak"
            elif typ == "bridge":
                cat = "O1_wrong_bridge"
            else:
                cat = "O2_crossing_fusion"
            ocat[cat] += 1
            # causal: gb's fate
            gb_home = gt_dom_pred[gb]
            if gb_home == p:
                o_absorbed += 1
                fate = f"gB={gb} ABSORBED into gA={dom} (lost as distinct id)"
            else:
                o_survives += 1
                fate = f"gB={gb} leaked {share_of_gb*100:.0f}% here; main id elsewhere (pred {gb_home})"
            angs = f"{ang:.0f}deg" if ang is not None else "n/a"
            pred_rows.append({"pred": p, "dom_gt": dom, "minority_gt": gb,
                              "minority_share_of_pred": round(share_in_pred, 2),
                              "share_of_minority_gt": round(share_of_gb, 2),
                              "type": typ, "angle": angs, "gap": round(gap, 1),
                              "category": cat, "fate": fate})
            links.append(f"pred {p}: gA={dom} + gB={gb} [{cat}, {typ} {angs}] -> {fate}")

    # ---- complementary instance-recovery metric ----
    well = 0
    for g in gt.masks:
        tot = gt_tot[g]
        if tot == 0:
            continue
        dom = gt_dom_pred[g]
        recovery = len(cell[(g, dom)]) / tot
        purity = len(cell[(g, dom)]) / max(1, pred_tot[dom])
        if recovery >= RECOVER_FRAC and purity >= RECOVER_FRAC:
            well += 1
    n_gt_present = sum(1 for g in gt.masks if gt_tot[g] > 0)

    rep = M.evaluate(pred, gt, frags, do_panoptic=False).clustering
    return {
        "name": name, "n_gt": len(gt.masks), "n_pred": pred.n,
        "f1": round(rep.pairwise_f1, 3), "prec": round(rep.pairwise_precision, 3),
        "rec": round(rep.pairwise_recall, 3),
        "well_recovered": well, "n_gt_present": n_gt_present,
        "ucat": dict(ucat), "ocat": dict(ocat),
        "o_absorbed": o_absorbed, "o_survives": o_survives,
        "gt_rows": gt_rows, "pred_rows": pred_rows, "links": links,
    }


def write_case_csv(res):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{res['name']}_undermerge.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gt", "n_frags", "n_pred", "dom_pred", "recovery", "category", "detail"])
        w.writeheader(); w.writerows(res["gt_rows"])
    with open(OUT / f"{res['name']}_overmerge.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pred", "dom_gt", "minority_gt", "minority_share_of_pred",
                                           "share_of_minority_gt", "type", "angle", "gap", "category", "fate"])
        w.writeheader(); w.writerows(res["pred_rows"])


UCATS = ["U1_crossing_split", "U2_wide_break", "U3_mid_break", "U4_dropped", "U5_minor_leak"]
OCATS = ["O1_wrong_bridge", "O2_crossing_fusion", "O3_crossing_leak"]


def write_md(results):
    L = ["# Categorized error table — actionable failure analysis", ""]
    L += ["## Headline metrics + complementary instance-recovery", "",
          "`F1/P/R` = fragment-pairwise (current metric). `well-recovered` = GT filaments whose "
          "dominant pred id covers ≥80% of the filament AND is ≥80% pure — a human-meaningful "
          "'basically got it' count that ignores small fragment leaks.", "",
          "| case | GT | pred | F1 | P | R | well-recovered | filaments deleted |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in results:
        L.append(f"| {r['name']} | {r['n_gt']} | {r['n_pred']} | {r['f1']} | {r['prec']} | {r['rec']} | "
                 f"{r['well_recovered']}/{r['n_gt_present']} ({100*r['well_recovered']/max(1,r['n_gt_present']):.0f}%) | "
                 f"{r['o_absorbed']} |")

    L += ["", "## UNDER-MERGE categories (failed connects)", "",
          "| case | " + " | ".join(UCATS) + " |", "|---|" + "---|" * len(UCATS)]
    tot_u = defaultdict(int)
    for r in results:
        L.append(f"| {r['name']} | " + " | ".join(str(r['ucat'].get(c, 0)) for c in UCATS) + " |")
        for c in UCATS:
            tot_u[c] += r['ucat'].get(c, 0)
    L.append(f"| **TOTAL** | " + " | ".join(f"**{tot_u[c]}**" for c in UCATS) + " |")

    L += ["", "## OVER-MERGE categories (wrong connects)", "",
          "| case | " + " | ".join(OCATS) + " |", "|---|" + "---|" * len(OCATS)]
    tot_o = defaultdict(int)
    for r in results:
        L.append(f"| {r['name']} | " + " | ".join(str(r['ocat'].get(c, 0)) for c in OCATS) + " |")
        for c in OCATS:
            tot_o[c] += r['ocat'].get(c, 0)
    L.append(f"| **TOTAL** | " + " | ".join(f"**{tot_o[c]}**" for c in OCATS) + " |")

    tot_abs = sum(r["o_absorbed"] for r in results)
    tot_surv = sum(r["o_survives"] for r in results)
    L += ["", "## Over-merge SEVERITY (the real metric refinement)", "",
          "A wrong-connect has two very different outcomes. Pairwise-F1 weights both by "
          "fragment-pair count, not by instance severity — that is the metric weakness:", "",
          f"- **ABSORBED — a filament DELETED: {tot_abs}.** The minority filament's own dominant id "
          "*is* the shared pred, so it has no separate instance. A true, severe instance error "
          "(lost a filament).",
          f"- **partial leak — both filaments survive: {tot_surv}.** The minority filament still has "
          "its own id elsewhere; only some crossing fragments leaked. Mild at the instance level, "
          "but pairwise-F1 penalizes it like a deletion.", "",
          "So precision is depressed partly by ~half-severity leaks. An **instance-level metric** "
          "(well-recovered % + filaments-deleted count) tracks the real goal — counting/measuring "
          "distinct filaments — better than fragment-pair precision alone. Crossing leaks (O3, only "
          f"{tot_o['O3_crossing_leak']}) are rare, so the issue is leak *severity weighting*, not leak count.", ""]

    for r in results:
        L += ["", f"## {r['name']} — wrong-connect → damaged-id links", ""]
        if not r["links"]:
            L.append("(none)")
        for ln in r["links"]:
            L.append(f"- {ln}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "error_table.md").write_text("\n".join(L), encoding="utf-8")


def main():
    results = []
    for name, run, gtp in CASES:
        if not run.exists():
            print(f"skip {name}", file=sys.stderr); continue
        r = analyze(name, run, gtp)
        write_case_csv(r)
        results.append(r)
        print(f"{r['name']:<11} F1={r['f1']} well-recovered={r['well_recovered']}/{r['n_gt_present']} "
              f"U={r['ucat']} O={r['ocat']}")
    write_md(results)
    print(f"\nwrote {OUT}/error_table.md + per-case CSVs")


if __name__ == "__main__":
    main()
