"""Reconnect troubleshooting utilities collected into one CLI.

Subcommands cover:
  - compare-followups: compare whether old postprocess ID pairs merged in a new run
  - diagnose-missed: explain likely gates for postprocess ID pairs that stayed split
  - check-coords: sample two coordinates and report whether they share a final ID
  - trace-evolution: compare old postprocess IDs against a newer run
  - visualize-pairs: render pair crops over the SEM image
  - trace-component: trace one run's IDs back to preprocess/stringart branches
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, deque
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "output" / "full_pipeline"
DEFAULT_REJECTION_LOG = ROOT / "3.reconnect" / "output" / "rejection_log_straight.csv"
DEFAULT_TRACE_RUN = PIPELINE_ROOT / "sem_full_00008_mask255_crop_20260507_105149"
PAIR_COLORS = {
    "a": (0, 200, 255),
    "b": (0, 80, 255),
}


def resolve_run_dir(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (path, ROOT / path, PIPELINE_ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (PIPELINE_ROOT / path).resolve()


def load_post_labels(run_dir: Path) -> np.ndarray:
    return tifffile.imread(run_dir / "4.postprocess" / "mask_post_labels.tif").astype(np.int32)


def parse_pairs(values: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in value.split(",")) for value in values]


def majority_label(lbl_img: np.ndarray, mask: np.ndarray) -> int:
    vals = lbl_img[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(Counter(vals.tolist()).most_common(1)[0][0])


def label_at(lbl: np.ndarray, r: int, c: int, radius: int = 2) -> int:
    """Return the dominant nonzero label in a small window around a coordinate."""
    height, width = lbl.shape
    r0, r1 = max(0, r - radius), min(height, r + radius + 1)
    c0, c1 = max(0, c - radius), min(width, c + radius + 1)
    win = lbl[r0:r1, c0:c1]
    vals = win[win > 0]
    if vals.size == 0:
        return 0
    counts = np.bincount(vals.astype(np.int64))
    return int(counts.argmax())


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return nb * skel.astype(np.uint8)


def endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere((skel > 0) & (neighbor_count(skel) == 1))
    return [tuple(int(v) for v in p) for p in pts]


def order_path(skel: np.ndarray) -> np.ndarray:
    """Return endpoint-to-endpoint ordered path for an open-curve skeleton."""
    eps = endpoints(skel)
    if len(eps) < 2:
        return np.argwhere(skel > 0)

    def bfs(start: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
        seen = {start: None}
        last = start
        queue = deque([start])
        while queue:
            p = queue.popleft()
            last = p
            r, c = p
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < skel.shape[0]
                        and 0 <= cc < skel.shape[1]
                        and skel[rr, cc]
                        and (rr, cc) not in seen
                    ):
                        seen[(rr, cc)] = p
                        queue.append((rr, cc))
        return last, seen

    a, _ = bfs(eps[0])
    b, parent = bfs(a)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = parent[p]
    return np.asarray(path[::-1], dtype=np.int32)


def tip_tangent(path: np.ndarray, idx: int, steps: int) -> np.ndarray:
    """Return a unit tangent at one path tip, looking inward by `steps` pixels."""
    if idx == 0:
        j = min(len(path) - 1, steps)
        delta = path[j] - path[0]
    else:
        j = max(0, len(path) - 1 - steps)
        delta = path[-1] - path[j]
    norm = float(np.hypot(delta[0], delta[1]))
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return delta / norm


def bundle_info(lbl_img: np.ndarray, cid: int) -> dict | None:
    mask = lbl_img == cid
    if not mask.any():
        return None
    skel = skeletonize(mask)
    if not skel.any():
        return None
    path = order_path(skel.astype(bool))
    if len(path) < 4:
        return None
    tip0 = path[0]
    tip1 = path[-1]
    tan0 = tip_tangent(path, 0, min(10, len(path) - 1))
    tan1 = tip_tangent(path, -1, min(10, len(path) - 1))
    return {
        "id": int(cid),
        "n_px": int(mask.sum()),
        "path_len_px": int(len(path)),
        "tip0": tuple(int(v) for v in tip0),
        "tip1": tuple(int(v) for v in tip1),
        "tan0_inward": tan0,
        "tan1_inward": -tan1,
    }


def gap_metrics(bundle_a: dict, bundle_b: dict) -> list[dict]:
    """Compute gap geometry for every cross-bundle tip pair."""
    results = []
    tip_pairs = [
        ("t0", bundle_a["tip0"], bundle_a["tan0_inward"], "t0", bundle_b["tip0"], bundle_b["tan0_inward"]),
        ("t0", bundle_a["tip0"], bundle_a["tan0_inward"], "t1", bundle_b["tip1"], bundle_b["tan1_inward"]),
        ("t1", bundle_a["tip1"], bundle_a["tan1_inward"], "t0", bundle_b["tip0"], bundle_b["tan0_inward"]),
        ("t1", bundle_a["tip1"], bundle_a["tan1_inward"], "t1", bundle_b["tip1"], bundle_b["tan1_inward"]),
    ]
    for label_a, tip_a, tan_a, label_b, tip_b, tan_b in tip_pairs:
        gap = np.array(tip_b) - np.array(tip_a)
        dist = float(np.hypot(gap[0], gap[1]))
        if dist < 1e-6:
            continue
        outward_a = -tan_a
        outward_b = -tan_b
        dir_a_to_b = gap / dist
        dir_b_to_a = -dir_a_to_b
        forward_a = float(np.dot(outward_a, dir_a_to_b))
        forward_b = float(np.dot(outward_b, dir_b_to_a))
        opposition = float(np.dot(tan_a, -tan_b))
        mid = (np.array(tip_a) + np.array(tip_b)) / 2
        va = mid - np.array(tip_a)
        vb = mid - np.array(tip_b)
        perp_a = float(np.linalg.norm(va - np.dot(va, outward_a) * outward_a))
        perp_b = float(np.linalg.norm(vb - np.dot(vb, outward_b) * outward_b))
        results.append(
            {
                "tip_a": label_a,
                "tip_b": label_b,
                "ta_rc": tuple(int(v) for v in tip_a),
                "tb_rc": tuple(int(v) for v in tip_b),
                "dist": dist,
                "forward_a": forward_a,
                "forward_b": forward_b,
                "opposition": opposition,
                "line_resid_px": (perp_a + perp_b) / 2.0,
            }
        )
    return results


def best_pair(metrics: list[dict]) -> dict:
    forward_ok = [m for m in metrics if m["forward_a"] > 0 and m["forward_b"] > 0]
    return min(forward_ok or metrics, key=lambda m: m["dist"])


GATES = {
    "stage_clear": {
        "clear_merge_max_dist_px": 16.0,
        "clear_merge_max_line_resid_px": 6.0,
        "clear_merge_min_opposition": 0.6,
    },
    "stage_strict": {
        "max_tip_distance_px": 15.0,
        "min_forward_cos": 0.85,
        "min_inward_opposition": 0.55,
        "max_line_residual_px": 6.0,
    },
    "stage_relaxed": {
        "max_tip_distance_px": 50.0,
        "min_forward_cos": 0.72,
        "min_inward_opposition": 0.4,
        "max_line_residual_px": 16.0,
    },
}


def diagnose_gate(metric: dict) -> list[dict]:
    out = []
    for stage in ("stage_clear", "stage_strict", "stage_relaxed"):
        gate = GATES[stage]
        fails = []
        if stage == "stage_clear":
            if metric["dist"] > gate["clear_merge_max_dist_px"]:
                fails.append(("clear_merge_max_dist_px", round(metric["dist"], 2), gate["clear_merge_max_dist_px"]))
            if metric["line_resid_px"] > gate["clear_merge_max_line_resid_px"]:
                fails.append(
                    (
                        "clear_merge_max_line_resid_px",
                        round(metric["line_resid_px"], 2),
                        gate["clear_merge_max_line_resid_px"],
                    )
                )
            if metric["opposition"] < gate["clear_merge_min_opposition"]:
                fails.append(
                    (
                        "clear_merge_min_opposition",
                        round(metric["opposition"], 3),
                        gate["clear_merge_min_opposition"],
                    )
                )
        else:
            min_forward = min(metric["forward_a"], metric["forward_b"])
            if metric["dist"] > gate["max_tip_distance_px"]:
                fails.append(("max_tip_distance_px", round(metric["dist"], 2), gate["max_tip_distance_px"]))
            if min_forward < gate["min_forward_cos"]:
                fails.append(("min_forward_cos", round(min_forward, 3), gate["min_forward_cos"]))
            if metric["opposition"] < gate["min_inward_opposition"]:
                fails.append(
                    (
                        "min_inward_opposition",
                        round(metric["opposition"], 3),
                        gate["min_inward_opposition"],
                    )
                )
            if metric["line_resid_px"] > gate["max_line_residual_px"]:
                fails.append(
                    (
                        "max_line_residual_px",
                        round(metric["line_resid_px"], 2),
                        gate["max_line_residual_px"],
                    )
                )
        out.append({"stage": stage, "fails": fails, "would_pass": len(fails) == 0})
    return out


def find_in_rejection_log(log_path: Path, ids_a: list[int], ids_b: list[int]) -> list[dict[str, str]]:
    if not log_path.exists():
        return []
    rows = []
    set_a, set_b = set(ids_a), set(ids_b)
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                base_id = int(row["base_id"])
                target_id = int(row["tar_id"])
            except (KeyError, ValueError):
                continue
            if (base_id in set_a and target_id in set_b) or (base_id in set_b and target_id in set_a):
                rows.append(row)
    return rows


def load_branch_masks(run_dir: Path, stage: str) -> dict[int, np.ndarray]:
    folder = run_dir / stage / "branches"
    out = {}
    for path in sorted(folder.glob("mask_branch_*.png")):
        try:
            idx = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        out[idx] = iio.imread(path) > 127
    return out


def print_branch_contributions(title: str, branches: dict[int, np.ndarray], region: np.ndarray) -> None:
    print(title)
    for idx, mask in branches.items():
        overlap = int((mask & region).sum())
        if overlap > 50:
            print(f"       branch {idx:2d}: {overlap:5d}px")


def crop_around(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int = 20) -> tuple[int, int, int, int, np.ndarray]:
    height, width = arr.shape[:2]
    r0 = max(0, r0 - pad)
    r1 = min(height, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(width, c1 + pad)
    return r0, r1, c0, c1, arr[r0:r1, c0:c1]


def draw_bundle(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    overlay = canvas.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def render_pair_crop(sem_rgb: np.ndarray, labels: np.ndarray, label_a: int, label_b: int, pad: int) -> np.ndarray | None:
    mask_a = labels == label_a
    mask_b = labels == label_b
    if not mask_a.any() or not mask_b.any():
        return None
    pts_a = np.argwhere(mask_a)
    pts_b = np.argwhere(mask_b)
    r0 = min(pts_a[:, 0].min(), pts_b[:, 0].min())
    r1 = max(pts_a[:, 0].max(), pts_b[:, 0].max())
    c0 = min(pts_a[:, 1].min(), pts_b[:, 1].min())
    c1 = max(pts_a[:, 1].max(), pts_b[:, 1].max())
    r0, r1, c0, c1, crop = crop_around(sem_rgb, r0, r1, c0, c1, pad=pad)
    out = draw_bundle(crop.copy(), mask_a[r0:r1, c0:c1].astype(np.uint8) * 255, PAIR_COLORS["a"])
    out = draw_bundle(out, mask_b[r0:r1, c0:c1].astype(np.uint8) * 255, PAIR_COLORS["b"])
    cv2.putText(out, f"A={label_a}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAIR_COLORS["a"], 1, cv2.LINE_AA)
    cv2.putText(out, f"B={label_b}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAIR_COLORS["b"], 1, cv2.LINE_AA)
    return out


def branch_index(path: Path) -> int:
    match = re.search(r"_branch_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def find_label_tif(step_path: Path) -> Path:
    for pattern in ("*_post_labels.tif", "*_reconnect_labels.tif", "*_labels.tif"):
        hits = [path for path in step_path.glob(pattern) if "dilated" not in path.stem]
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"No label TIFF found in {step_path}")


def trace_step_dir(run_dir: Path, step: str) -> Path:
    mapping = {
        "final": run_dir / "final",
        "reconnect": run_dir / "3.reconnect",
        "postprocess": run_dir / "4.postprocess",
        "preprocess": run_dir / "2.preprocess" / "branches",
    }
    try:
        return mapping[step.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown step '{step}'. Choose: final, reconnect, postprocess, preprocess.") from exc


def branch_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "2.preprocess" / "branches").glob("*_branch_*.png"), key=branch_index)


def raw_branch_files(run_dir: Path) -> list[Path]:
    raw_dir = run_dir / "2.preprocess" / "raw_branches"
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*_branch_*.png"), key=branch_index)


def load_bin(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img > 0 if img is not None else np.zeros((1, 1), dtype=bool)


def trace_single_id(labels: np.ndarray, label_id: int, branches: list[Path], raw_branches: list[Path]) -> dict:
    mask = labels == label_id
    if not mask.any():
        return {"id": label_id, "found": False}

    pts = np.argwhere(mask)
    r0, c0 = pts.min(0).tolist()
    r1, c1 = pts.max(0).tolist()
    raw_map = {path.name: path for path in raw_branches}
    branch_hits = []
    for branch_path in branches:
        branch_mask = load_bin(branch_path)
        if branch_mask.shape != mask.shape:
            continue
        overlap = int(np.count_nonzero(mask & branch_mask))
        if overlap == 0:
            continue
        raw_overlap = 0
        raw_path = raw_map.get(branch_path.name)
        if raw_path:
            raw_mask = load_bin(raw_path)
            if raw_mask.shape == mask.shape:
                raw_overlap = int(np.count_nonzero(mask & raw_mask))
        branch_hits.append(
            {
                "branch": branch_index(branch_path),
                "file": branch_path.name,
                "preprocess_px": overlap,
                "stringart_px": raw_overlap,
            }
        )
    branch_hits.sort(key=lambda item: -item["preprocess_px"])
    return {
        "id": label_id,
        "found": True,
        "area_px": int(mask.sum()),
        "bbox_rc": [r0, c0, r1, c1],
        "dominant_branch": branch_hits[0]["branch"] if branch_hits else None,
        "branches": branch_hits,
    }


def compare_followups(args: argparse.Namespace) -> None:
    old_dir = resolve_run_dir(args.old_run)
    new_dir = resolve_run_dir(args.new_run)
    old_lbl = load_post_labels(old_dir)
    new_lbl = load_post_labels(new_dir)

    print(f"OLD: {old_dir.name}  ids={len(np.unique(old_lbl)) - 1}")
    print(f"NEW: {new_dir.name}  ids={len(np.unique(new_lbl)) - 1}\n")

    merged = 0
    split = 0
    for old_a, old_b in parse_pairs(args.pairs):
        mask_a = old_lbl == old_a
        mask_b = old_lbl == old_b
        if not mask_a.any() or not mask_b.any():
            print(f"  pair {old_a},{old_b}: missing in OLD")
            continue
        new_a = majority_label(new_lbl, mask_a)
        new_b = majority_label(new_lbl, mask_b)
        cov_a = int((new_lbl == new_a)[mask_a].sum()) if new_a else 0
        cov_b = int((new_lbl == new_b)[mask_b].sum()) if new_b else 0
        same = new_a == new_b and new_a > 0
        outcome = "MERGED" if same else "split"
        merged += int(same)
        split += int(not same)
        print(
            f"  pair {old_a:>3}<->{old_b:<3}  OLD: {old_a}({mask_a.sum()}px) {old_b}({mask_b.sum()}px)"
            f"  -> NEW: {old_a}->{new_a}({cov_a}/{mask_a.sum()}px)"
            f"  {old_b}->{new_b}({cov_b}/{mask_b.sum()}px)  {outcome}"
        )

    total = merged + split
    print(f"\nSummary: {merged}/{total} pairs merged, {split}/{total} still split")


def diagnose_missed(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run)
    labels_path = run_dir / "4.postprocess" / "mask_post_labels.tif"
    rejection_log = Path(args.rejection_log).resolve() if args.rejection_log else DEFAULT_REJECTION_LOG
    lbl = tifffile.imread(labels_path).astype(np.int32)
    pairs = parse_pairs(args.pairs)

    print(f"loaded {labels_path} shape={lbl.shape}  unique={len(np.unique(lbl)) - 1}")
    print(f"diagnosing {len(pairs)} pairs against {rejection_log}")

    for label_a, label_b in pairs:
        print(f"\n====== Pair  {label_a}  <-> {label_b}  ======")
        info_a = bundle_info(lbl, label_a)
        info_b = bundle_info(lbl, label_b)
        if info_a is None or info_b is None:
            print(f"  one of the ids not found / too small: {label_a}={info_a is not None}  {label_b}={info_b is not None}")
            continue
        print(
            f"  {label_a}: {info_a['n_px']}px  skel_len={info_a['path_len_px']}"
            f"  tips={info_a['tip0']} / {info_a['tip1']}"
        )
        print(
            f"  {label_b}: {info_b['n_px']}px  skel_len={info_b['path_len_px']}"
            f"  tips={info_b['tip0']} / {info_b['tip1']}"
        )

        best = best_pair(gap_metrics(info_a, info_b))
        print(
            f"  best tip pair: {best['tip_a']}<->{best['tip_b']}"
            f"  dist={best['dist']:.2f}  fwd=({best['forward_a']:.2f},{best['forward_b']:.2f})"
            f"  opp={best['opposition']:.2f}  resid={best['line_resid_px']:.2f}"
        )

        for diagnosis in diagnose_gate(best):
            verdict = "PASS" if diagnosis["would_pass"] else "fail"
            failures = ", ".join(
                f"{key}={value}>{threshold}" if "max_" in key else f"{key}={value}<{threshold}"
                for key, value, threshold in diagnosis["fails"]
            )
            print(f"    {diagnosis['stage']:14s} {verdict}  {failures}")

        rows = find_in_rejection_log(rejection_log, [label_a], [label_b])
        if rows:
            print(f"  rejection log direct-match: {len(rows)} rows  (showing 3)")
            for row in rows[:3]:
                print(
                    f"    stage={row.get('stage'):14s}  reason={row.get('reason'):16s}"
                    f"  dist={row.get('dist', '')}  fwd_b={row.get('forward_base', '')}"
                    f"  opp={row.get('inward_opposition', '')}  resid={row.get('line_resid', '')}"
                )
        else:
            print("  (no direct rec-id rejection rows under these post IDs; post IDs may differ from reconnect IDs)")


def check_coords(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run)
    lbl = load_post_labels(run_dir)
    print(f"Run: {run_dir.name}  ids={len(np.unique(lbl)) - 1}\n")

    merged = 0
    split = 0
    missing = 0
    for spec in args.coords:
        name, coord_a, coord_b = spec.split("|")
        r1, c1 = (int(x) for x in coord_a.split(","))
        r2, c2 = (int(x) for x in coord_b.split(","))
        label_a = label_at(lbl, r1, c1, radius=args.radius)
        label_b = label_at(lbl, r2, c2, radius=args.radius)
        if label_a == 0 or label_b == 0:
            outcome = "MISSING"
            missing += 1
        elif label_a == label_b:
            outcome = "MERGED"
            merged += 1
        else:
            outcome = "still split"
            split += 1
        print(f"  {name:>14s}: ({r1},{c1})=id{label_a}   ({r2},{c2})=id{label_b}   {outcome}")
    print(f"\nSummary: merged={merged}, still-split={split}, missing={missing}")


def trace_evolution(args: argparse.Namespace) -> None:
    new_dir = resolve_run_dir(args.new_run)
    old_dir = resolve_run_dir(args.old_run)
    post_new = load_post_labels(new_dir)
    post_old = load_post_labels(old_dir)
    rec_new = tifffile.imread(new_dir / "3.reconnect" / "mask_reconnect_labels_dilated.tif")
    rec_old = tifffile.imread(old_dir / "3.reconnect" / "mask_reconnect_labels_dilated.tif")
    stringart_new = load_branch_masks(new_dir, "1.stringart")
    stringart_old = load_branch_masks(old_dir, "1.stringart")
    preprocess_new = load_branch_masks(new_dir, "2.preprocess")

    print(f"NEW = {new_dir}")
    print(f"OLD = {old_dir}\n")
    print(f"post_new ids: {sorted(int(v) for v in np.unique(post_new) if v)}")
    print(f"post_old ids: {sorted(int(v) for v in np.unique(post_old) if v)}\n")
    print(f"reconnect_new ids: n={len(np.unique(rec_new)) - 1}")
    print(f"reconnect_old ids: n={len(np.unique(rec_old)) - 1}\n")

    union_new = np.zeros_like(post_new, dtype=bool)
    for mask in stringart_new.values():
        union_new |= mask
    union_old = np.zeros_like(post_old, dtype=bool)
    for mask in stringart_old.values():
        union_old |= mask

    for old_id in args.ids:
        old_mask = post_old == old_id
        n_old = int(old_mask.sum())
        if n_old == 0:
            print(f"\n=== old id {old_id}: NOT FOUND in post_old ===")
            continue
        print(f"\n=== old id {old_id}: {n_old} px ===")

        new_ids, counts = np.unique(post_new[old_mask], return_counts=True)
        order = np.argsort(counts)[::-1]
        print("  new post IDs that occupy these pixels:")
        for new_id, count in zip(new_ids[order], counts[order]):
            if new_id == 0:
                print(f"    background({new_id}): {count}px ({count / n_old:.0%})")
            else:
                print(f"    id {int(new_id):3d}: {int(count):5d}px ({count / n_old:.0%})")

        rec_old_ids, rec_old_counts = np.unique(rec_old[old_mask], return_counts=True)
        print("  reconnect_old labels at these pixels:")
        for rec_id, count in zip(rec_old_ids, rec_old_counts):
            if rec_id:
                print(f"    rec_old {int(rec_id):3d}: {int(count):5d}px")

        for top_new in new_ids[order][:4]:
            if top_new == 0:
                continue
            new_mask = post_new == int(top_new)
            rec_ids, rec_counts = np.unique(rec_new[new_mask], return_counts=True)
            print(f"  -> new post id {int(top_new)} ({int(new_mask.sum())}px): reconnect_new sources:")
            for rec_id, count in zip(rec_ids, rec_counts):
                if rec_id:
                    print(f"       rec_new {int(rec_id):3d}: {int(count):5d}px")
            print_branch_contributions("     preprocess-new branch contribution:", preprocess_new, new_mask)
            print_branch_contributions("     stringart-new branch contribution:", stringart_new, new_mask)

        lost = old_mask & ~union_new
        kept = old_mask & union_new
        was_in_old = old_mask & union_old
        print("  pixel survival from stringart (in any new branch):")
        print(f"    kept by new stringart: {int(kept.sum()):5d}px")
        print(f"    lost by new stringart: {int(lost.sum()):5d}px")
        print(f"    (was present in old stringart: {int(was_in_old.sum()):5d}px)")


def visualize_pairs(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run)
    labels = load_post_labels(run_dir)
    sem_path = Path(args.sem).resolve() if args.sem else ROOT / "input" / "sem_full_00000_1p66_crop512" / "sem.png"
    sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
    if sem is None:
        raise FileNotFoundError(f"Could not read SEM image: {sem_path}")
    sem_rgb = cv2.cvtColor(sem, cv2.COLOR_GRAY2BGR)
    out_dir = Path(args.output).resolve() if args.output else run_dir / "missed_pairs_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_a, label_b in parse_pairs(args.pairs):
        img = render_pair_crop(sem_rgb, labels, label_a, label_b, pad=args.pad)
        if img is None:
            print(f"  skipping {label_a},{label_b}: not found")
            continue
        scale = max(1, int(np.ceil(args.min_size / min(img.shape[:2]))))
        if scale > 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        path = out_dir / f"pair_{label_a}_{label_b}.png"
        cv2.imwrite(str(path), img)
        print(f"  saved {path}  ({img.shape[1]}x{img.shape[0]})")


def trace_component(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run)
    labels_dir = trace_step_dir(run_dir, args.step)
    labels_path = find_label_tif(labels_dir)
    labels = iio.imread(labels_path).astype(np.int32)
    branches = branch_files(run_dir)
    raw_branches = raw_branch_files(run_dir)

    print(f"Run     : {run_dir.name}")
    print(f"Step    : {args.step} ({labels_path.name})")
    print(f"Branches: {len(branches)} preprocess | {len(raw_branches)} raw stringart")
    print()

    for label_id in args.ids:
        info = trace_single_id(labels, label_id, branches, raw_branches)
        if not info["found"]:
            print(f"  ID {label_id}: NOT FOUND in {labels_path.name}")
            continue
        r0, c0, r1, c1 = info["bbox_rc"]
        print(
            f"ID {label_id:4d}  area={info['area_px']:5d}px"
            f"  bbox=r[{r0}:{r1}] c[{c0}:{c1}]  dominant_branch={info['dominant_branch']}"
        )
        if not args.quiet:
            for hit in info["branches"]:
                bar = "#" * max(1, hit["preprocess_px"] // 5)
                raw_note = f"  (stringart: {hit['stringart_px']}px)" if hit["stringart_px"] else "  (not in raw)"
                print(f"    branch_{hit['branch']:02d}  pre={hit['preprocess_px']:4d}px {bar}{raw_note}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconnect troubleshooting utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare-followups", help="Compare old pair IDs against a newer run.")
    compare.add_argument("--old-run", required=True)
    compare.add_argument("--new-run", required=True)
    compare.add_argument("--pairs", nargs="+", required=True, help='Pair list like "31,14 13,43".')
    compare.set_defaults(func=compare_followups)

    diagnose = subparsers.add_parser("diagnose-missed", help="Explain likely gates for pairs that stayed split.")
    diagnose.add_argument("--run", default="sem_full_00000_1p66_crop512_SKEL")
    diagnose.add_argument("--pairs", nargs="+", required=True, help='Pair list like "31,14 13,43".')
    diagnose.add_argument("--rejection-log", default=None, help="Optional explicit reconnect rejection-log CSV path.")
    diagnose.set_defaults(func=diagnose_missed)

    coords = subparsers.add_parser("check-coords", help="Check whether coordinate pairs share the same final ID.")
    coords.add_argument("--run", required=True)
    coords.add_argument(
        "--coords",
        nargs="+",
        required=True,
        help='Specs like "pairA|120,240|130,248"; merged iff both points resolve to one label.',
    )
    coords.add_argument("--radius", type=int, default=2, help="Sampling radius around each coordinate.")
    coords.set_defaults(func=check_coords)

    trace = subparsers.add_parser("trace-evolution", help="Trace old postprocess IDs into a newer run.")
    trace.add_argument("--new-run", default="sem_full_00000_1p66_crop512_SKEL")
    trace.add_argument("--old-run", default="sem_full_00000_1p66_crop512_b")
    trace.add_argument("--ids", type=int, nargs="+", default=[8, 9, 11])
    trace.set_defaults(func=trace_evolution)

    visualize = subparsers.add_parser("visualize-pairs", help="Render crops for pair review over the SEM image.")
    visualize.add_argument("--run", default="sem_full_00000_1p66_crop512_SKEL")
    visualize.add_argument("--sem", default=None, help="Optional SEM image override.")
    visualize.add_argument("--pairs", nargs="+", required=True, help='Pair list like "31,14 13,43".')
    visualize.add_argument("--output", default=None, help="Optional output folder override.")
    visualize.add_argument("--pad", type=int, default=20, help="Padding around each pair crop.")
    visualize.add_argument("--min-size", type=int, default=300, help="Upscale crops smaller than this size.")
    visualize.set_defaults(func=visualize_pairs)

    component = subparsers.add_parser("trace-component", help="Trace one run's IDs back to source branches.")
    component.add_argument("--run", default=str(DEFAULT_TRACE_RUN), help="Pipeline run root folder.")
    component.add_argument(
        "--step",
        default="final",
        choices=["final", "reconnect", "postprocess", "preprocess"],
        help="Which step's label TIFF to start from.",
    )
    component.add_argument("--ids", type=int, nargs="+", default=[25, 38, 39, 42, 68], help="Label IDs to trace.")
    component.add_argument("--quiet", action="store_true", help="Only print dominant branch per ID.")
    component.set_defaults(func=trace_component)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
