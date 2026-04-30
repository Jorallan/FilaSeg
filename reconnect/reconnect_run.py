from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage import io as skio
from skimage.morphology import skeletonize

ALLOWED_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
EXT_PREFERENCE = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "reconnect_config.yaml"),
    )
    ap.add_argument(
        "--version",
        type=str,
        choices=["v5", "v6", "v7"],
        default="v6",
        help="Which utils version to use (v5 or v6).",
    )
    ap.add_argument("--input", type=str, default=None, help="override input_dir")
    ap.add_argument("--output", type=str, default=None, help="override output_dir")
    ap.add_argument("--background", type=str, default=None, help="optional image used for final overlay background")
    ap.add_argument("--pattern", type=str, default=None, help="process only bases matching this substring")
    ap.add_argument("--compare", action="store_true", help="If set, compare extracted stats to GT json and plot.")
    ap.add_argument("--no_show", action="store_true", help="If set, do not show plots (still saves + prints).")
    return ap.parse_args()


def _pick_existing_by_preference(folder: Path, stem: str) -> Path | None:
    for ext in EXT_PREFERENCE:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_basenames(input_dir: Path) -> list[str]:
    bases = set()

    for p in sorted(input_dir.glob("*_branches_merge.*")):
        name = p.stem
        base = name[: -len("_branches_merge")] if name.endswith("_branches_merge") else name.split("_branches_merge")[0]
        bases.add(base.rstrip("_"))

    rx = re.compile(r"^(.*)_branch_(\d+)$")
    for p in input_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in ALLOWED_EXTS):
            continue
        m = rx.match(p.stem)
        if m:
            bases.add(m.group(1).rstrip("_"))

    return sorted(bases)


def find_branch_paths(input_dir: Path, base: str) -> list[Path]:
    rx = re.compile(rf"^{re.escape(base)}_branch_(\d+)$")
    rank = {ext: i for i, ext in enumerate(EXT_PREFERENCE)}

    best_for_k: dict[int, Path] = {}
    for p in input_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in ALLOWED_EXTS):
            continue
        m = rx.match(p.stem)
        if not m:
            continue
        k = int(m.group(1))
        cur = best_for_k.get(k)
        if cur is None or rank.get(p.suffix.lower(), 999) < rank.get(cur.suffix.lower(), 999):
            best_for_k[k] = p

    return [best_for_k[k] for k in sorted(best_for_k)]


def _load_gt_json_for_base(input_dir: Path, base: str) -> dict | None:
    for p in (
        input_dir / f"{base}_merge.json",
        input_dir / f"{base}_merge_stats.json",
        input_dir / f"{base}.json",
    ):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d["full"] if isinstance(d, dict) and "full" in d else d
    return None


def _read_background_any(path: Path) -> np.ndarray:
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 1.0
    if mx > 1.5:
        img /= 255.0 if mx <= 255.0 else mx
    return np.clip(img, 0.0, 1.0)


def _component_endpoints_from_skeleton(sk: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(sk)
    if ys.size == 0:
        return []
    sk_u8 = sk.astype(np.uint8)
    H, W = sk_u8.shape
    endpoints = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        nb = int(sk_u8[y0:y1, x0:x1].sum()) - 1
        if nb == 1:
            endpoints.append((y, x))
    return endpoints


def _mean_angle_from_skeleton(sk: np.ndarray) -> float:
    ys, xs = np.where(sk)
    if ys.size < 2:
        return 0.0
    X = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    X -= X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    w, v = np.linalg.eigh(C)
    vec = v[:, int(np.argmax(w))]
    ang = float(np.arctan2(vec[1], vec[0]))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    return ang


def _compute_extracted_stats_from_labels(lbl: np.ndarray, dilate_px_for_contacts: int = 2) -> dict:
    if lbl is None or lbl.size == 0:
        return {}

    ids = np.unique(lbl)
    ids = ids[ids != 0]
    n_fil = int(ids.size)
    H, W = lbl.shape
    total_pixels = int((lbl > 0).sum())

    arc_list, ang_list, wid_list, tor_list = [], [], [], []

    for cid in ids:
        m = (lbl == cid)
        area = int(m.sum())
        if area < 2:
            continue

        sk = skeletonize(m)
        arc = float(np.count_nonzero(sk))
        if arc < 2:
            continue

        mean_w = float(area / (arc + 1e-9))
        eps = _component_endpoints_from_skeleton(sk)

        if len(eps) >= 2:
            pts = np.asarray(eps, dtype=np.float32)
            dmax = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dy = float(pts[i, 0] - pts[j, 0])
                    dx = float(pts[i, 1] - pts[j, 1])
                    d = (dx * dx + dy * dy) ** 0.5
                    if d > dmax:
                        dmax = d
            e2e = float(dmax)
        else:
            ys, xs = np.where(sk)
            e2e = float(((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2) ** 0.5) if ys.size else 0.0

        arc_list.append(arc)
        wid_list.append(mean_w)
        tor_list.append(float(arc / (e2e + 1e-9)))
        ang_list.append(_mean_angle_from_skeleton(sk))

    arc = np.asarray(arc_list, dtype=np.float32)
    wid = np.asarray(wid_list, dtype=np.float32)
    tor = np.asarray(tor_list, dtype=np.float32)
    ang = np.asarray(ang_list, dtype=np.float32)

    unique_contacts = 0
    if n_fil > 1:
        k = int(max(1, dilate_px_for_contacts))
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        votes = np.zeros((H, W), np.uint16)
        for cid in ids:
            dm = cv2.dilate((lbl == cid).astype(np.uint8), kernel, iterations=1)
            votes += dm.astype(np.uint16)

        contact_mask = (votes >= 2).astype(np.uint8) * 255
        if contact_mask.max() > 0:
            ncc, _ = cv2.connectedComponents(contact_mask, connectivity=8)
            unique_contacts = int(max(0, ncc - 1))

    return {
        "total_filaments": n_fil,
        "total_pixels": total_pixels,
        "unique_contacts": unique_contacts,
        "arc_values": arc,
        "angle_values": ang,
        "tortuosity_values": tor,
        "width_values": wid,
        "overlap_proxy": {
            "overlap_pixels": 0,
            "covered_pixels": total_pixels,
            "overlap_fraction_of_covered": 0.0,
            "sum_excess_coverage": 0,
            "max_coverage": 1,
        },
    }


def _smooth_counts(counts: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    if counts.size == 0:
        return counts
    r = int(max(1, round(3.0 * float(sigma_bins))))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * (sigma_bins ** 2)))
    k /= (k.sum() + 1e-12)
    cpad = np.pad(counts, (r, r), mode="edge")
    return np.convolve(cpad, k, mode="same")[r:-r].astype(np.float32)


def _total_length_from_gt_hist(gt_hist: dict | None) -> float:
    if not gt_hist:
        return 0.0
    edges, counts = gt_hist.get("bin_edges", []), gt_hist.get("hist", [])
    if not edges or not counts:
        return 0.0
    edges = np.asarray(edges, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float((centers * counts).sum())


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) + 1e-9)


def _try_get_gt_scalar(gt: dict | None, keys: list[str]) -> float | int | None:
    if not isinstance(gt, dict):
        return None

    containers = [
        gt,
        gt.get("summary", {}) if isinstance(gt.get("summary", None), dict) else {},
        gt.get("stats", {}) if isinstance(gt.get("stats", None), dict) else {},
        gt.get("counts", {}) if isinstance(gt.get("counts", None), dict) else {},
        gt.get("metadata", {}) if isinstance(gt.get("metadata", None), dict) else {},
    ]

    for c in containers:
        for k in keys:
            if k in c and isinstance(c[k], (int, float)):
                return c[k]

    return None


def _plot_compare_curve(
    ax,
    gt_hist: dict | None,
    ex_values: np.ndarray,
    title: str,
    xlabel: str,
    *,
    to_degrees: bool = False,
    xlim: tuple[float, float] | None = None,
    bins_if_no_gt: int = 50,
    smooth_sigma_bins: float = 1.5,
):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")

    ex = np.asarray(ex_values, dtype=np.float32)
    ex = ex[np.isfinite(ex)]
    if to_degrees and ex.size:
        ex = ex * (180.0 / np.pi)

    has_gt = bool(gt_hist and gt_hist.get("bin_edges") and gt_hist.get("hist"))
    if not has_gt:
        if ex.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        hist_ex, edges = np.histogram(ex, bins=bins_if_no_gt, range=xlim)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, _smooth_counts(hist_ex, sigma_bins=smooth_sigma_bins), linewidth=2.0, label="Extracted")
        ax.legend(loc="best")
        if xlim:
            ax.set_xlim(*xlim)
        return

    gt_edges = np.asarray(gt_hist["bin_edges"], dtype=np.float32)
    gt_counts = np.asarray(gt_hist["hist"], dtype=np.float32)
    if to_degrees:
        gt_edges = gt_edges * (180.0 / np.pi)

    centers = 0.5 * (gt_edges[:-1] + gt_edges[1:])
    y_gt = _smooth_counts(gt_counts, sigma_bins=smooth_sigma_bins)

    def _mask(x, y):
        if xlim is None:
            return x, y
        m = (x >= xlim[0]) & (x <= xlim[1])
        return x[m], y[m]

    x_gt, y_gt = _mask(centers, y_gt)
    ax.plot(x_gt, y_gt, linewidth=2.0, label="GT")

    if ex.size:
        ex_counts, _ = np.histogram(ex, bins=gt_edges)
        y_ex = _smooth_counts(ex_counts.astype(np.float32), sigma_bins=smooth_sigma_bins)
        x_ex, y_ex = _mask(centers, y_ex)
        ax.plot(x_ex, y_ex, linewidth=2.0, label="Extracted")

    ax.legend(loc="best")
    if xlim:
        ax.set_xlim(*xlim)


def _save_u16_tif(path: Path, arr: np.ndarray):
    skio.imsave(str(path), arr.astype(np.uint16), check_contrast=False)


def _save_preview_png(path: Path, lbl: np.ndarray):
    mx = max(1, int(lbl.max()))
    vis = (lbl.astype(np.float32) / mx * 255.0).astype(np.uint8)
    vis[lbl == 0] = 0
    skio.imsave(str(path), vis, check_contrast=False)


def _load_width_hint(input_dir: Path) -> float | None:
    """Read filament_width_px written by stringart_tiles.py into the branches folder."""
    for p in (input_dir / "filament_width.json", input_dir.parent / "filament_width.json"):
        if p.exists():
            try:
                w = float(json.loads(p.read_text(encoding="utf-8")).get("filament_width_px", 0))
                return w if w > 0 else None
            except Exception:
                pass
    return None


def _make_stage_cfg(base_cfg: dict, stage_key: str) -> dict:
    """Return a deep copy of base_cfg with stage-specific overrides merged in."""
    stage = base_cfg.get(stage_key, {})
    if not stage:
        return base_cfg
    cfg = copy.deepcopy(base_cfg)
    for section, overrides in stage.items():
        if isinstance(overrides, dict):
            cfg.setdefault(section, {}).update(overrides)
        else:
            cfg[section] = overrides
    cfg.setdefault("debug", {})["stage_label"] = stage_key
    return cfg


def _resolve_config_path(config_arg: str) -> Path:
    p = Path(config_arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    p2 = (Path(__file__).parent / config_arg)
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(
        f"Config not found: {config_arg!r}. "
        f"Looked in current working directory and next to the script: {p2}"
    )


def main():
    args = parse_args()

    utils = importlib.import_module(f"reconnect_utils_{args.version}")
    read_gray_any = utils.read_gray_any
    binarize_mask = utils.binarize_mask
    extract_components = utils.extract_components
    clean_binary_mask = utils.clean_binary_mask
    branchpoints_8 = utils.branchpoints_8
    remove_branchpoint_regions = utils.remove_branchpoint_regions
    reconnect_components = utils.reconnect_components
    relabel_components = utils.relabel_components
    dilate_label_image = utils.dilate_label_image
    make_overlay = utils.make_overlay

    cfg_path = _resolve_config_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    io_cfg = cfg.get("io", {})
    input_dir = Path(args.input or io_cfg.get("input_dir", "./input")).resolve()
    output_dir = Path(args.output or io_cfg.get("output_dir", "./output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Width hint from stringart_tiles — adjusts dilate_px to match filament scale
    width_hint = _load_width_hint(input_dir)
    if width_hint is not None:
        dilate_auto = max(1, round(width_hint * 0.5))
        cfg.setdefault("morphology", {})["dilate_px"] = dilate_auto
        print(f"[reconnect-{args.version}] filament_width={width_hint:.2f}px from tiles -> dilate_px={dilate_auto}")
    elif cfg.get("shared", {}).get("filament_width_px"):
        width_hint = float(cfg["shared"]["filament_width_px"])
        dilate_auto = max(1, round(width_hint * 0.5))
        cfg.setdefault("morphology", {})["dilate_px"] = dilate_auto
        print(f"[reconnect-{args.version}] filament_width={width_hint:.2f}px from config -> dilate_px={dilate_auto}")

    # Two-stage setup
    has_stages = "stage1" in cfg and "stage2" in cfg
    if has_stages:
        cfg1 = _make_stage_cfg(cfg, "stage1")
        cfg2 = _make_stage_cfg(cfg, "stage2")
        print(f"[reconnect-{args.version}] two-stage: "
              f"stage1={cfg1['runtime']['max_passes']} passes (strict), "
              f"stage2={cfg2['runtime']['max_passes']} passes (relaxed)")

    print(f"[reconnect-{args.version}] input_dir:", input_dir)

    bases = find_basenames(input_dir)
    if args.pattern:
        bases = [b for b in bases if args.pattern in b]
    if not bases:
        print(f"[reconnect-{args.version}] No bases found in: {input_dir}")
        return

    thr = cfg["thresholds"]
    morph = cfg["morphology"]
    outcfg = cfg["outputs"]
    pre = cfg.get("preprocess", {})
    adv = cfg.get("advanced", {})
    component_mask_mode = str(adv.get("component_mask_mode", "full")).strip().lower()

    for base in bases:
        merged_path = _pick_existing_by_preference(input_dir, f"{base}_branches_merge")
        branch_paths = find_branch_paths(input_dir, base)
        if not branch_paths:
            print(f"[reconnect-{args.version}] skip '{base}': no branch files found")
            continue

        orig_path = _pick_existing_by_preference(input_dir, f"{base}_original")
        print(f"[reconnect-{args.version}] processing: {base}  (branches found: {len(branch_paths)})")
        branch_shape = read_gray_any(branch_paths[0]).shape[:2]

        if args.background:
            bg = _read_background_any(Path(args.background).resolve())
        elif orig_path is not None:
            bg = read_gray_any(orig_path)
        elif merged_path is not None:
            bg = read_gray_any(merged_path)
        else:
            acc = None
            for bp in branch_paths:
                img = read_gray_any(bp)
                acc = img if acc is None else np.maximum(acc, img)
            bg = acc if acc is not None else np.zeros((1, 1), dtype=np.float32)

        if bg.shape[:2] != branch_shape:
            raise ValueError(
                f"Background shape {bg.shape[:2]} does not match branch shape "
                f"{branch_shape}"
            )

        merged_bp = None
        if bool(pre.get("split_at_branchpoints", False)):
            if merged_path is not None:
                merged01 = read_gray_any(merged_path)
            else:
                merged01 = np.zeros_like(bg)
                for bp in branch_paths:
                    merged01 = np.maximum(merged01, read_gray_any(bp))
            merged_bin = binarize_mask(merged01, float(thr["branch_bin"]))
            merged_sk = skeletonize(merged_bin)
            merged_bp = branchpoints_8(merged_sk)

        components = []
        cid = 1
        for layer, bp in enumerate(branch_paths, start=1):
            img = read_gray_any(bp)
            m = binarize_mask(img, float(thr["branch_bin"]))
            m = clean_binary_mask(m, int(pre.get("clean_min_area", thr.get("min_component_area", 0))))
            if merged_bp is not None:
                m = remove_branchpoint_regions(m, merged_bp, int(pre.get("bp_dilate_px", 0)))

            comps, cid = extract_components(
                m,
                layer=layer,
                min_area=int(thr["min_component_area"]),
                start_id=cid,
                skeletonize_each=bool(pre.get("skeletonize_components", True)),
                component_mask_mode=component_mask_mode,
            )
            components.extend(comps)

        if not components:
            print(f"[reconnect-{args.version}] '{base}': no components after thresholding")
            continue

        if has_stages:
            print(f"[reconnect-{args.version}] stage 1 (strict)...")
            components = reconnect_components(components, cfg1)
            print(f"[reconnect-{args.version}] stage 2 (relaxed)...")
            components = reconnect_components(components, cfg2)
        else:
            components = reconnect_components(components, cfg)
        lbl = relabel_components(components)
        lbl_dil = dilate_label_image(lbl, int(morph["dilate_px"])) if lbl is not None else None
        overlay = make_overlay(bg, lbl_dil if lbl_dil is not None else lbl, float(outcfg["overlay_alpha"]))

        if outcfg.get("save_raw_labels", True) and lbl is not None:
            _save_u16_tif(output_dir / f"{base}_reconnect_labels.tif", lbl)

        if outcfg.get("save_dilated_labels", True) and lbl_dil is not None:
            _save_u16_tif(output_dir / f"{base}_reconnect_labels_dilated.tif", lbl_dil)

        if outcfg.get("save_label_previews", True) and lbl is not None:
            _save_preview_png(output_dir / f"{base}_reconnect_labels_preview.png", lbl)

        if outcfg.get("save_label_previews", True) and lbl_dil is not None:
            _save_preview_png(output_dir / f"{base}_reconnect_labels_dilated_preview.png", lbl_dil)

        if outcfg.get("save_overlay", True):
            skio.imsave(str(output_dir / f"{base}_reconnect_overlay.png"), overlay, check_contrast=False)

        print(f"[reconnect-{args.version}] saved -> {output_dir}")

        if not args.compare:
            continue

        ex = _compute_extracted_stats_from_labels(
            lbl.astype(np.int32) if lbl is not None else lbl,
            dilate_px_for_contacts=max(1, int(morph.get("dilate_px", 2))),
        )
        gt = _load_gt_json_for_base(input_dir, base)
        has_gt = gt is not None

        if has_gt:
            gt_fil = _try_get_gt_scalar(
                gt,
                ["total_filaments", "n_filaments", "filament_count", "num_filaments"],
            )
            gt_contacts = _try_get_gt_scalar(
                gt,
                ["unique_contacts", "n_unique_contacts", "contact_count", "num_contacts"],
            )

            ex_fil = ex.get("total_filaments", None)
            ex_contacts = ex.get("unique_contacts", None)

            print(f"\n[compare] --- {base} ---")

            if gt_fil is not None and ex_fil is not None:
                print("[compare] total filaments (GT)      :", int(gt_fil))
                print("[compare] total filaments (Extracted):", int(ex_fil))
                print("[compare] filament ratio EX / GT     :", f"{_safe_ratio(ex_fil, gt_fil):.3f}")
            else:
                print("[compare] total filaments (GT)      : not found in GT json")
                print("[compare] total filaments (Extracted):", ex_fil)

            if gt_contacts is not None and ex_contacts is not None:
                print("[compare] unique contacts (GT)      :", int(gt_contacts))
                print("[compare] unique contacts (Extracted):", int(ex_contacts))
                print("[compare] contact ratio EX / GT     :", f"{_safe_ratio(ex_contacts, gt_contacts):.3f}")
            else:
                print("[compare] unique contacts (GT)      : not found in GT json")
                print("[compare] unique contacts (Extracted):", ex_contacts)

            gt_total_len = _total_length_from_gt_hist(gt.get("arc_length_distribution", None))
            ex_arc = ex.get("arc_values", np.array([], dtype=np.float32))
            ex_total_len = float(np.sum(ex_arc)) if ex_arc.size else 0.0
            ratio = ex_total_len / (gt_total_len + 1e-9)

            print("[compare] total length (GT)       :", f"{gt_total_len:.1f}")
            print("[compare] total length (Extracted):", f"{ex_total_len:.1f}")
            print("[compare] length ratio EX / GT    :", f"{ratio:.3f}")
        else:
            print(f"[compare] '{base}': GT json not found in input_dir -> plotting Extracted-only distributions")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes = np.asarray(axes).ravel()

        _plot_compare_curve(
            axes[0],
            gt.get("arc_length_distribution", None) if has_gt else None,
            ex.get("arc_values", np.array([], dtype=np.float32)),
            title="Arc length distribution",
            xlabel="px (GT: polyline length, EX: skeleton px count)",
            smooth_sigma_bins=1.2,
        )
        _plot_compare_curve(
            axes[1],
            gt.get("angle_distribution", None) if has_gt else None,
            ex.get("angle_values", np.array([], dtype=np.float32)),
            title="Angle distribution",
            xlabel="degrees",
            to_degrees=True,
            xlim=(-90.0, 90.0),
            smooth_sigma_bins=1.2,
        )
        _plot_compare_curve(
            axes[2],
            gt.get("tortuosity_distribution", None) if has_gt else None,
            ex.get("tortuosity_values", np.array([], dtype=np.float32)),
            title="Tortuosity distribution",
            xlabel="arc / e2e",
            xlim=(0.75, 1.20),
            smooth_sigma_bins=1.2,
        )

        fig.suptitle(f"{'GT vs Extracted' if has_gt else 'Extracted only'}: {base}", y=1.05)
        plt.tight_layout()

        fig_path = output_dir / f"{base}_compare.png"
        fig.savefig(str(fig_path), dpi=200)
        print("[compare] saved plot ->", fig_path)

        if not args.no_show:
            plt.show(block=True)
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
