"""Manual correction GUI for flattened and multilabel filament IDs.

Typical use:
    python Tools/manualID_correct.py --input output/full_pipeline/<run>/final

Controls:
    1/select  click selects an ID, ctrl-click toggles IDs
    2/cut     click cuts the active/clicked ID and splits connected pieces
    3/erase   drag removes pixels; with lock on, only selected IDs are affected
    4/paint   drag adds pixels to active/new ID; with lock on, other IDs are shared
    j         join selected IDs into one cleaned/smoothed ID
    q         smooth selected ID paths
    l         toggle non-selected-ID protection
    wheel     brush size in edit modes; ctrl+wheel zoom; shift/alt+wheel pan
    n/p       step active ID; u undo; s save to ./modified
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from scipy.interpolate import splprep, splev
from scipy.ndimage import convolve
from skimage import io as skio
from skimage.color import gray2rgb
from skimage.measure import label as cc_label
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries

for _key in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_key] = []

LABEL_PATTERNS = ["*_manual_labels.tif", "*_post_labels.tif", "*_reconnect_labels.tif", "*_labels.tif", "*.tif"]
BG_PATTERNS = ["*_original.*", "*_post_overlay.png", "*_reconnect_overlay.png", "*_overlay.png", "*_preview.png"]
DEFAULT_INPUT = Path(r"C:\Repos\filaments_quantification\output\full_pipeline\sem_full_00000_1p66_crop512_20260515_220950\final")
PROMPT_FOR_INPUT = False
SCRIPT_INPUT = DEFAULT_INPUT


def strip_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_manual_labels", "_reconnect_labels", "_post_labels"):
        stem = stem.replace(suffix, "")
    return stem[:-7] if stem.endswith("_labels") else stem


def first_match(folder: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(p for p in folder.glob(pat) if "dilated" not in p.stem and "_labels_preview" not in p.stem)
        if hits:
            return hits[0]
    return None


def prompt_folder(default: Path | None) -> Path:
    val = input(f"Folder to edit [{default}]: " if default else "Folder to edit: ").strip().strip('"')
    if val:
        return Path(val)
    if default is not None:
        return default
    raise ValueError("No input folder provided.")


def display_name(folder: Path) -> str:
    return f"{folder.parent.name}/{folder.name}" if folder.name.lower() in {"final", "modified", "3.reconnect", "4.postprocess"} else folder.name


def load_rgb(path: Path) -> np.ndarray:
    img = skio.imread(str(path)).astype(np.float32)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = img[..., :3]
    if img.max() > 1.5:
        img /= 255.0 if img.max() <= 255 else img.max()
    return np.clip(img, 0.0, 1.0)


def sidecar_candidates(folder: Path, stem: str) -> list[Path]:
    roots = [folder / "modified", folder] if (folder / "modified").exists() else [folder]
    tags = ("manual", "reconnect", "post")
    exact = [root / f"{stem}_{tag}_multilabel{ext}" for root in roots for tag in tags for ext in (".npz", ".tif")]
    loose: list[Path] = []
    for root in roots:
        for tag in tags:
            loose += sorted(root.glob(f"*_{tag}_multilabel.npz"))
            loose += sorted(root.glob(f"*_{tag}_multilabel.tif"))
    return exact + loose


def find_sidecar(folder: Path, stem: str) -> Path | None:
    return next((p for p in sidecar_candidates(folder, stem) if p.exists()), None)


def id_color(cid: int) -> np.ndarray:
    return np.asarray([((cid * 53) % 256) / 255.0, ((cid * 97) % 256) / 255.0, ((cid * 193) % 256) / 255.0], dtype=np.float32)


def colorize_members(members: dict[int, set[int]], shape: tuple[int, int]) -> np.ndarray:
    acc = np.zeros((shape[0] * shape[1], 3), dtype=np.float32)
    counts = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    for cid, indices in members.items():
        if indices:
            idx = np.fromiter(indices, dtype=np.intp, count=len(indices))
            acc[idx] += id_color(cid)
            counts[idx] += 1
    m = counts > 0
    acc[m] /= counts[m, None]
    return acc.reshape((*shape, 3))


def flat_set(mask: np.ndarray) -> set[int]:
    return set(int(v) for v in np.flatnonzero(mask))


def mask_from(indices: set[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    if indices:
        out[np.fromiter(indices, dtype=np.intp, count=len(indices))] = True
    return out.reshape(shape)


def members_from_labels(lbl: np.ndarray) -> dict[int, set[int]]:
    flat = lbl.ravel()
    return {int(cid): set(int(v) for v in np.flatnonzero(flat == cid)) for cid in np.unique(flat) if int(cid) > 0}


def flatten_members(members: dict[int, set[int]], shape: tuple[int, int], rule: str) -> np.ndarray:
    ids = [cid for cid, pix in members.items() if pix]
    if rule == "lowest":
        order = sorted(ids, reverse=True)
    elif rule == "highest":
        order = sorted(ids)
    elif rule == "random":
        order = list(np.random.default_rng(0).permutation(sorted(ids)))
    else:
        order = sorted(ids, key=lambda cid: (len(members[cid]), cid))
    out = np.zeros(shape[0] * shape[1], dtype=np.int32)
    for cid in order:
        out[np.fromiter(members[cid], dtype=np.intp, count=len(members[cid]))] = cid
    return out.reshape(shape)


def overlap_mask(members: dict[int, set[int]], shape: tuple[int, int]) -> np.ndarray:
    counts = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    for pix in members.values():
        if pix:
            counts[np.fromiter(pix, dtype=np.intp, count=len(pix))] += 1
    return (counts > 1).reshape(shape)


def load_npz(path: Path, shape: tuple[int, int]) -> dict[int, set[int]]:
    data = np.load(str(path))
    saved_shape = tuple(int(v) for v in data["shape"])
    if saved_shape != tuple(shape):
        raise ValueError(f"Multilabel shape {saved_shape} does not match {shape}")
    ids = data["ids"].astype(np.int64)
    indptr = data["indptr"].astype(np.int64)
    indices = data["indices"].astype(np.int64)
    return {int(cid): set(int(v) for v in indices[indptr[i]:indptr[i + 1]]) for i, cid in enumerate(ids) if indptr[i + 1] > indptr[i]}


def load_multitiff(path: Path, shape: tuple[int, int]) -> dict[int, set[int]]:
    import tifffile

    arr = tifffile.imread(str(path))
    arr = arr[None, ...] if arr.ndim == 2 else arr
    if tuple(arr.shape[-2:]) != tuple(shape):
        raise ValueError(f"Multilabel TIFF shape {tuple(arr.shape[-2:])} does not match {shape}")
    ids_path = path.with_name(f"{path.stem}_ids.json")
    ids = json.loads(ids_path.read_text(encoding="utf-8")).get("page_to_id") if ids_path.exists() else None
    return {int(ids[i]) if ids and i < len(ids) else int(page.max()): flat_set(page > 0) for i, page in enumerate(arr) if np.any(page)}


def save_npz(path: Path, members: dict[int, set[int]], shape: tuple[int, int]) -> None:
    ids = [cid for cid in sorted(members) if members[cid]]
    indptr, chunks = [0], []
    for cid in ids:
        pix = np.fromiter(sorted(members[cid]), dtype=np.uint64, count=len(members[cid]))
        chunks.append(pix)
        indptr.append(indptr[-1] + len(pix))
    np.savez_compressed(
        str(path),
        shape=np.asarray(shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.uint64),
        indices=np.concatenate(chunks) if chunks else np.array([], dtype=np.uint64),
    )


def save_multitiff(path: Path, ids_path: Path, members: dict[int, set[int]], shape: tuple[int, int]) -> None:
    import tifffile

    ids = [cid for cid in sorted(members) if members[cid]]
    dtype = np.uint16 if (not ids or max(ids) <= np.iinfo(np.uint16).max) else np.uint32
    with tifffile.TiffWriter(str(path), bigtiff=True) as tif:
        for cid in ids:
            page = np.zeros(shape[0] * shape[1], dtype=dtype)
            page[np.fromiter(members[cid], dtype=np.intp, count=len(members[cid]))] = cid
            tif.write(page.reshape(shape), photometric="minisblack")
    ids_path.write_text(json.dumps({"page_to_id": ids}, indent=2), encoding="utf-8")


def disk(shape: tuple[int, int], r: int, c: int, radius: int) -> np.ndarray:
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - r) ** 2 + (cc - c) ** 2 <= radius ** 2


def endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return [tuple(int(v) for v in p) for p in np.argwhere((skel > 0) & (nb == 1))]


def neighbors8(p: tuple[int, int], skel: np.ndarray):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                    yield rr, cc


def shortest_path(skel: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    q, parent = deque([a]), {a: None}
    while q:
        p = q.popleft()
        if p == b:
            break
        for n in neighbors8(p, skel):
            if n not in parent:
                parent[n] = p
                q.append(n)
    if b not in parent:
        return np.argwhere(skel > 0).astype(np.float32)
    out, p = [], b
    while p is not None:
        out.append(p)
        p = parent[p]
    return np.asarray(out[::-1], dtype=np.float32)


def dominant_path(mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(mask)
    pts = np.argwhere(sk > 0)
    if len(pts) < 3:
        return pts.astype(np.float32)
    eps = endpoints(sk)
    if len(eps) >= 2:
        a, b = max(((x, y) for i, x in enumerate(eps) for y in eps[i + 1:]), key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
        return shortest_path(sk, a, b)
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    axis = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, 1]
    return pts[np.argsort((xy - xy.mean(0)) @ axis)].astype(np.float32)


def smooth_path(path: np.ndarray, samples: int = 512, strength: float = 1.8) -> np.ndarray:
    if len(path) < 4:
        return path
    pts = path.astype(np.float32)
    dist = np.r_[0.0, np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))]
    pts = pts[np.r_[True, np.diff(dist) > 0.2]]
    if len(pts) < 4:
        return pts
    length = float(dist[-1])
    n_out = int(np.clip(length * 1.2, 16, samples))
    centered = pts - pts.mean(axis=0)
    if len(pts) >= 8:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        straight_tol = min(4.0, max(1.25, 0.006 * length))
        if np.percentile(np.abs(centered @ vh[1]), 85) <= straight_tol:
            return np.linspace(pts[0], pts[-1], n_out).astype(np.float32)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=max(4.0, len(pts) * strength), k=min(3, len(pts) - 1))
        out = np.stack(splev(np.linspace(0, 1, n_out), tck), axis=1).astype(np.float32)
        out[0], out[-1] = pts[0], pts[-1]
        return out
    except Exception:
        return pts


def stroke_width(mask: np.ndarray, path: np.ndarray, fallback: int) -> int:
    """Estimate the actual stroke width of a mask using area / centerline-length.

    Returns the real width so join/smooth do not inflate the source thickness.
    `fallback` is only used when the path is too short to estimate from.
    """
    if len(path) < 2:
        return max(1, int(fallback))
    length = float(np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))))
    if length < 1.0:
        return max(1, int(fallback))
    width = int(round(float(mask.sum()) / length))
    return max(1, width)


def draw_path(shape: tuple[int, int], path: np.ndarray, thickness: int) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    if len(path) == 1:
        cv2.circle(out, tuple(np.round(path[0, ::-1]).astype(np.int32)), max(1, thickness // 2), 255, -1, cv2.LINE_AA)
    elif len(path) > 1:
        cv2.polylines(out, [np.round(path[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)], False, 255, max(1, int(thickness)), cv2.LINE_AA)
    return out > 32


def components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    lab = cc_label(mask, connectivity=2)
    comps = [lab == k for k in range(1, int(lab.max()) + 1) if int(np.count_nonzero(lab == k)) >= min_area]
    return sorted(comps, key=lambda m: int(m.sum()), reverse=True)


def nearest_points(a: np.ndarray, b: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pa = np.asarray(endpoints(skeletonize(a)) or np.argwhere(a).tolist(), dtype=np.int32)
    pb = np.asarray(endpoints(skeletonize(b)) or np.argwhere(b).tolist(), dtype=np.int32)
    ia, ib = np.unravel_index(int(np.argmin(np.sum((pa[:, None, :] - pb[None, :, :]) ** 2, axis=2))), (len(pa), len(pb)))
    return tuple(int(v) for v in pa[ia]), tuple(int(v) for v in pb[ib])


def bridge_components(mask: np.ndarray) -> np.ndarray:
    comps = components(mask, 1)
    if len(comps) <= 1:
        return mask
    out, linked, rest = np.logical_or.reduce(comps), [comps[0]], comps[1:]
    while rest:
        best = min(((np.sum((np.asarray(a) - np.asarray(b)) ** 2), i, a, b) for i, comp in enumerate(rest) for linked_comp in linked for a, b in [nearest_points(linked_comp, comp)]), key=lambda x: x[0])
        _, idx, a, b = best
        line = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(line, (a[1], a[0]), (b[1], b[0]), 1, 1, cv2.LINE_AA)
        out |= line > 0
        linked.append(rest.pop(idx))
    return out


def cleaned_mask(mask: np.ndarray, thickness: int, min_area: int, connect: bool = True) -> np.ndarray:
    """Re-render a mask via skeleton path + estimated local width.

    Width is derived from the source mask itself (area / centerline length),
    so join/smooth never inflate beyond the natural stroke. The `thickness`
    argument is only used as a fallback when the path is too short to measure.
    """
    work = remove_small_objects(mask.astype(bool), min_size=max(1, min_area))
    work = bridge_components(work) if connect else work
    out = np.zeros(mask.shape, dtype=bool)
    for comp in components(work, min_area):
        path = smooth_path(dominant_path(comp))
        width = stroke_width(comp, path, thickness)
        out |= draw_path(mask.shape, path, width)
    return out if out.any() else work


def render(bg: np.ndarray, lbl: np.ndarray, color: np.ndarray, selected: set[int], hover: int | None,
           alpha: float, dim: float, gain: float, outline: bool, focus_mask: np.ndarray | None) -> np.ndarray:
    out = bg.copy()
    m = lbl > 0
    out[m] = alpha * color[m] + (1.0 - alpha) * out[m]
    if selected or hover:
        fm = focus_mask if focus_mask is not None else np.isin(lbl, list(selected | ({hover} if hover else set())))
        out *= dim
        out[fm] = np.clip(out[fm] * gain + 0.15, 0.0, 1.0)
        if outline:
            out[find_boundaries(fm, mode="outer")] = (1.0, 0.1, 0.1)
    return out


class ManualIDEditor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.folder = args.input.resolve()
        if not self.folder.exists():
            raise FileNotFoundError(self.folder)
        self.lbl_path = args.labels or first_match(self.folder, LABEL_PATTERNS)
        self.bg_path = args.background or first_match(self.folder, BG_PATTERNS)
        if self.bg_path is None and self.folder.name.lower() == "modified":
            self.bg_path = first_match(self.folder.parent, BG_PATTERNS)
        if self.lbl_path is None or self.bg_path is None:
            raise FileNotFoundError(f"Need labels and background in {self.folder}")

        self.lbl = skio.imread(str(self.lbl_path)).astype(np.int32)
        if self.lbl.ndim != 2:
            raise ValueError(f"Labels must be 2D, got {self.lbl.shape}")
        self.bg = load_rgb(self.bg_path)
        if self.bg.shape[:2] != self.lbl.shape:
            raise ValueError(f"Background {self.bg.shape[:2]} != labels {self.lbl.shape}")

        self.stem = strip_stem(self.lbl_path)
        sidecar = find_sidecar(self.folder, self.stem)
        self.members = (load_npz(sidecar, self.lbl.shape) if sidecar and sidecar.suffix.lower() == ".npz"
                        else load_multitiff(sidecar, self.lbl.shape) if sidecar else members_from_labels(self.lbl))
        if sidecar:
            print(f"[manual] loaded multilabel memberships: {sidecar}")

        self.state = {
            "mode": "select", "selected": set(), "hover": None, "brush": int(args.brush),
            "alpha": float(args.overlay_alpha), "dim": float(args.dim_bg), "gain": float(args.brighten_gain),
            "outline": True, "drag": False, "dirty": False, "cursor": None,
            "buttons": set(), "pan_anchor": None, "pan_moved": False, "protect": True,
        }
        self.history: list[dict[str, object]] = []
        self.undo: list[tuple[dict[int, set[int]], int]] = []
        self.edit_start: dict[int, set[int]] | None = None
        self.sync()
        self.build_ui()

    def sync(self) -> None:
        for cid in list(self.members):
            if not self.members[cid]:
                self.members.pop(cid, None)
        self.lbl[:, :] = flatten_members(self.members, self.lbl.shape, self.args.flatten_rule)
        self.overlap = overlap_mask(self.members, self.lbl.shape)
        self.color = colorize_members(self.members, self.lbl.shape)

    def ids(self) -> list[int]:
        return sorted(cid for cid, pix in self.members.items() if pix)

    def active(self) -> int | None:
        return max(self.state["selected"]) if self.state["selected"] else None

    def selected_existing(self) -> set[int]:
        return {cid for cid in self.state["selected"] if self.members.get(cid)}

    def locked(self) -> bool:
        return bool(self.state["protect"] and self.selected_existing())

    def push_undo(self) -> None:
        self.undo.append(({cid: set(pix) for cid, pix in self.members.items() if pix}, len(self.history)))
        del self.undo[:-20]

    def log(self, action: str, ids_before, id_after: int | None, pixels: int, note: str = "") -> None:
        self.history.append({
            "step": len(self.history) + 1,
            "action": action,
            "ids_before": ";".join(map(str, ids_before)) if isinstance(ids_before, (list, tuple, set)) else str(ids_before),
            "id_after": "" if id_after is None else int(id_after),
            "pixels_changed": int(pixels),
            "note": note,
        })

    def next_id(self) -> int:
        ids = self.ids()
        return max(ids) + 1 if ids else 1

    def focus_mask(self) -> np.ndarray | None:
        focus = set(self.state["selected"])
        if self.state["hover"]:
            focus.add(int(self.state["hover"]))
        masks = [mask_from(self.members[cid], self.lbl.shape) for cid in focus if self.members.get(cid)]
        return np.logical_or.reduce(masks) if masks else None

    def pick(self, event) -> tuple[int, int, int | None] | None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        r, c = int(round(event.ydata)), int(round(event.xdata))
        if not (0 <= r < self.lbl.shape[0] and 0 <= c < self.lbl.shape[1]):
            return None
        cid = int(self.lbl[r, c])
        return r, c, cid if cid > 0 else None

    def stats_text(self, cid: int | None) -> str:
        if cid is None or not self.members.get(cid):
            return ""
        idx = np.fromiter(self.members[cid], dtype=np.intp, count=len(self.members[cid]))
        rows, cols = np.divmod(idx, self.lbl.shape[1])
        shared = int(self.overlap.ravel()[idx].sum())
        return f"id {cid}\narea {len(idx)} px\nshared {shared} px\nbbox r[{rows.min()}:{rows.max()}] c[{cols.min()}:{cols.max()}]"

    def build_ui(self) -> None:
        self.fig = plt.figure(figsize=(13, 8), num=f"{display_name(self.folder)} | manual ID correction")
        gs = self.fig.add_gridspec(13, 2, width_ratios=[4.8, 1.7])
        self.ax = self.fig.add_subplot(gs[:10, 0])
        self.ax.set_axis_off()
        self.ax_info = self.fig.add_subplot(gs[:10, 1])
        self.ax_info.set_axis_off()
        self.ax_alpha = self.fig.add_subplot(gs[10, :])
        self.ax_dim = self.fig.add_subplot(gs[11, :])
        self.ax_gain = self.fig.add_subplot(gs[12, :])
        plt.subplots_adjust(hspace=0.45)
        self.full_xlim = (-0.5, self.lbl.shape[1] - 0.5)
        self.full_ylim = (self.lbl.shape[0] - 0.5, -0.5)
        self.im = self.ax.imshow(self.render_current(), interpolation="nearest")
        self.brush_shadow = Circle((0, 0), self.state["brush"], fill=False, edgecolor="black", linewidth=2.6, alpha=0.65, visible=False)
        self.brush_cursor = Circle((0, 0), self.state["brush"], fill=False, edgecolor="white", linewidth=1.2, alpha=0.95, visible=False)
        self.ax.add_patch(self.brush_shadow)
        self.ax.add_patch(self.brush_cursor)
        self.info = self.ax_info.text(0.0, 1.0, "", transform=self.ax_info.transAxes, fontsize=9, color="black",
                                      ha="left", va="top", wrap=True)
        self.s_alpha = Slider(self.ax_alpha, "overlay_alpha", 0.0, 1.0, valinit=self.state["alpha"], valstep=0.01)
        self.s_dim = Slider(self.ax_dim, "dim_bg", 0.1, 1.0, valinit=self.state["dim"], valstep=0.01)
        self.s_gain = Slider(self.ax_gain, "brighten_gain", 1.0, 3.0, valinit=self.state["gain"], valstep=0.01)
        for name, fn in {
            "button_press_event": self.on_click, "button_release_event": self.on_release,
            "motion_notify_event": self.on_move, "scroll_event": self.on_scroll,
            "key_press_event": self.on_key,
        }.items():
            self.fig.canvas.mpl_connect(name, fn)
        self.s_alpha.on_changed(self.on_slider)
        self.s_dim.on_changed(self.on_slider)
        self.s_gain.on_changed(self.on_slider)
        self.redraw()

    def render_current(self) -> np.ndarray:
        return render(self.bg, self.lbl, self.color, self.state["selected"], self.state["hover"],
                      self.state["alpha"], self.state["dim"], self.state["gain"], self.state["outline"], self.focus_mask())

    def redraw(self) -> None:
        self.update_brush()
        self.im.set_data(self.render_current())
        lock = "on" if self.state["protect"] else "off"
        title = (f"{display_name(self.folder)} | mode={self.state['mode']} | lock={lock} | ids={len(self.ids())} "
                 f"| overlaps={int(self.overlap.sum())} | selected={sorted(self.state['selected'])} "
                 f"| brush_diam={self.state['brush'] * 2 + 1}")
        self.ax.set_title(title + (" | unsaved" if self.state["dirty"] else ""))
        help_line = "1 select | 2 cut | 3 erase | 4 paint\nctrl-click multi-select | l lock | j join | q smooth\nwheel brush | ctrl+wheel zoom | shift/alt+wheel pan"
        self.info.set_text((self.stats_text(self.active() or self.state["hover"]) + "\n" + help_line).strip())
        self.fig.canvas.draw_idle()

    def update_brush(self) -> None:
        visible = self.state["cursor"] is not None and self.state["mode"] in {"cut", "erase", "paint"}
        for artist in (self.brush_shadow, self.brush_cursor):
            artist.set_visible(visible)
            artist.set_radius(self.state["brush"])
            if visible:
                r, c = self.state["cursor"]
                artist.center = (c, r)

    def save_all(self) -> None:
        out_dir = self.folder / "modified"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.sync()
        label_dtype = np.uint16 if int(self.lbl.max()) <= np.iinfo(np.uint16).max else np.uint32
        out_lbl = out_dir / f"{self.stem}_manual_labels.tif"
        out_npz = out_dir / f"{self.stem}_manual_multilabel.npz"
        out_color = out_dir / f"{self.stem}_manual_instances_color.png"
        out_overlay = out_dir / f"{self.stem}_manual_overlay.png"
        out_history = out_dir / f"{self.stem}_manual_history.csv"
        skio.imsave(str(out_lbl), self.lbl.astype(label_dtype), check_contrast=False)
        save_npz(out_npz, self.members, self.lbl.shape)
        if self.args.save_multilabel_tiff:
            save_multitiff(out_dir / f"{self.stem}_manual_multilabel.tif", out_dir / f"{self.stem}_manual_multilabel_ids.json", self.members, self.lbl.shape)
        skio.imsave(str(out_color), np.clip(self.color * 255, 0, 255).astype(np.uint8), check_contrast=False)
        skio.imsave(str(out_dir / f"{self.stem}_manual_overlap.png"), (self.overlap.astype(np.uint8) * 255), check_contrast=False)
        skio.imsave(str(out_overlay), np.clip(render(self.bg, self.lbl, self.color, set(), None, self.state["alpha"], 1.0, 1.0, False, None) * 255, 0, 255).astype(np.uint8), check_contrast=False)
        with out_history.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "action", "ids_before", "id_after", "pixels_changed", "note"])
            writer.writeheader()
            writer.writerows(self.history)
        self.state["dirty"] = False
        print(f"[saved] {out_lbl}")
        print(f"[saved] {out_npz}")

    def split_id(self, cid: int) -> None:
        comps = components(mask_from(self.members.get(cid, set()), self.lbl.shape), self.args.min_area)
        if not comps:
            self.members.pop(cid, None)
            self.state["selected"].discard(cid)
            self.sync()
            return
        self.members[cid] = flat_set(comps[0])
        for comp in comps[1:]:
            self.members[self.next_id()] = flat_set(comp)
        self.sync()

    def cut_at(self, r: int, c: int, cid: int | None) -> None:
        cid = (cid if cid in self.selected_existing() else self.active()) if self.locked() else (self.active() or cid)
        if cid is None or not self.members.get(cid):
            return
        self.push_undo()
        self.members[cid].difference_update(flat_set(disk(self.lbl.shape, r, c, self.state["brush"])))
        self.split_id(cid)
        self.state["selected"] = {cid} if self.members.get(cid) else set()
        self.state["dirty"] = True

    def erase_at(self, r: int, c: int) -> None:
        area = flat_set(disk(self.lbl.shape, r, c, self.state["brush"]))
        for cid in (self.selected_existing() if self.locked() else set(self.members)):
            self.members.get(cid, set()).difference_update(area)
        self.sync()
        self.state["selected"] = {cid for cid in self.state["selected"] if self.members.get(cid)}
        self.state["dirty"] = True

    def paint_at(self, r: int, c: int) -> None:
        cid = self.active() or self.next_id()
        self.state["selected"] = {cid}
        area = flat_set(disk(self.lbl.shape, r, c, self.state["brush"]))
        if not self.state["protect"]:
            for other in list(self.members):
                if other != cid:
                    self.members[other].difference_update(area)
        self.members.setdefault(cid, set()).update(area)
        self.sync()
        self.state["dirty"] = True

    def join_selected(self) -> None:
        selected = sorted(cid for cid in self.state["selected"] if self.members.get(cid))
        if len(selected) < 2:
            return
        self.push_undo()
        keep = selected[0]
        before_px = sum(len(self.members[cid]) for cid in selected)
        source = np.logical_or.reduce([mask_from(self.members[cid], self.lbl.shape) for cid in selected])
        merged = flat_set(cleaned_mask(source, self.args.thickness, self.args.min_area, True))
        for cid in selected:
            self.members.pop(cid, None)
        if not self.state["protect"]:
            for cid in list(self.members):
                self.members[cid].difference_update(merged)
        self.members[keep] = merged
        self.state["selected"] = {keep}
        self.sync()
        self.log("join", selected, keep, abs(len(merged) - before_px), f"merged_area={len(merged)}")
        self.state["dirty"] = True

    def smooth_selected(self) -> None:
        selected = sorted(self.selected_existing())
        if not selected:
            return
        self.push_undo()
        changed = 0
        for cid in selected:
            before = set(self.members[cid])
            smoothed = flat_set(cleaned_mask(mask_from(before, self.lbl.shape), self.args.thickness, self.args.min_area, False))
            if not smoothed:
                continue
            changed += len(before ^ smoothed)
            self.members[cid] = smoothed
            if not self.state["protect"]:
                for other in list(self.members):
                    if other != cid:
                        self.members[other].difference_update(smoothed)
        if changed:
            self.sync()
            self.state["selected"] = {cid for cid in selected if self.members.get(cid)}
            self.log("smooth", selected, selected[0] if len(selected) == 1 else None, changed)
            self.state["dirty"] = True
        else:
            self.undo.pop()

    def finish_erase_history(self, before: dict[int, set[int]] | None) -> None:
        if before is None:
            return
        removed = {cid: len(before.get(cid, set()) - self.members.get(cid, set())) for cid in before}
        removed = {cid: n for cid, n in removed.items() if n > 0}
        if removed:
            self.log("erase", sorted(removed), None, sum(removed.values()),
                     ";".join(f"{cid}:{n}" for cid, n in sorted(removed.items())))

    def step_id(self, direction: int) -> None:
        ids = self.ids()
        if ids:
            cur = self.active()
            idx = ids.index(cur) if cur in ids else 0
            self.state["selected"] = {ids[(idx + direction) % len(ids)]}

    @staticmethod
    def has_key(event, name: str) -> bool:
        key = str(getattr(event, "key", "") or "").lower()
        return name in key or (name == "control" and "ctrl" in key)

    @staticmethod
    def button(event):
        b = getattr(event, "button", None)
        return int(b.value) if hasattr(b, "value") else b

    def pan_fraction(self, xfrac: float, yfrac: float) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x0 + xfrac * abs(x1 - x0), x1 + xfrac * abs(x1 - x0))
        self.ax.set_ylim(y0 + yfrac * abs(y1 - y0), y1 + yfrac * abs(y1 - y0))

    def on_click(self, event) -> None:
        button = self.button(event)
        if button in {1, 2, 3}:
            self.state["buttons"].add(button)
        if button in {2, 3} and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.state["pan_anchor"], self.state["pan_moved"] = (float(event.xdata), float(event.ydata)), False
            return
        hit = self.pick(event)
        if hit is None or button != 1:
            return
        r, c, cid = hit
        self.state["cursor"], self.state["drag"] = (r, c), True
        if self.state["mode"] == "select":
            if cid is None:
                if not self.has_key(event, "control"):
                    self.state["selected"].clear()
            elif self.has_key(event, "control"):
                self.state["selected"].symmetric_difference_update({cid})
            else:
                self.state["selected"] = {cid}
        elif self.state["mode"] == "cut":
            self.cut_at(r, c, cid)
        else:
            self.push_undo()
            self.edit_start = {cid: set(pix) for cid, pix in self.members.items() if pix} if self.state["mode"] == "erase" else None
            (self.erase_at if self.state["mode"] == "erase" else self.paint_at)(r, c)
        self.redraw()

    def on_release(self, event) -> None:
        button = self.button(event)
        was_drag = bool(self.state["drag"])
        mode = self.state["mode"]
        if button == 1:
            self.state["drag"] = False
            if was_drag and mode == "erase":
                self.finish_erase_history(self.edit_start)
                self.edit_start = None
        self.state["buttons"].discard(button)
        if button == 3 and not self.state["pan_moved"]:
            self.state["selected"].clear()
            self.redraw()
        if button in {2, 3} and not ({2, 3} & self.state["buttons"]):
            self.state["pan_anchor"], self.state["pan_moved"] = None, False

    def on_move(self, event) -> None:
        if self.state["pan_anchor"] is not None and ({2, 3} & self.state["buttons"]) and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None:
                return
            xa, ya = self.state["pan_anchor"]
            dx, dy = xa - float(event.xdata), ya - float(event.ydata)
            if abs(dx) + abs(dy) > 0.01:
                x0, x1 = self.ax.get_xlim()
                y0, y1 = self.ax.get_ylim()
                self.ax.set_xlim(x0 + dx, x1 + dx)
                self.ax.set_ylim(y0 + dy, y1 + dy)
                self.state["pan_moved"] = True
                self.redraw()
            return
        hit = self.pick(event)
        if hit is None:
            self.state["cursor"] = None
            self.redraw()
            return
        r, c, cid = hit
        self.state["cursor"] = (r, c)
        if self.state["drag"] and self.state["mode"] in {"erase", "paint"}:
            (self.erase_at if self.state["mode"] == "erase" else self.paint_at)(r, c)
        else:
            self.state["hover"] = cid
        self.redraw()

    def on_scroll(self, event) -> None:
        hit = self.pick(event)
        if hit is None:
            return
        r, c, _ = hit
        self.state["cursor"] = (r, c)
        step = int(np.sign(getattr(event, "step", 0) or (1 if event.button == "up" else -1)))
        if self.has_key(event, "control"):
            scale = 0.82 if step > 0 else 1.22
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            x, y = float(event.xdata), float(event.ydata)
            self.ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
            self.ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        elif self.has_key(event, "shift") or ({1, 3} <= self.state["buttons"]):
            self.pan_fraction(0.18 * step, 0.0)
        elif self.has_key(event, "alt") or self.has_key(event, "option"):
            self.pan_fraction(0.0, -0.18 * step)
        elif self.state["mode"] in {"cut", "erase", "paint"}:
            self.state["brush"] = int(np.clip(self.state["brush"] + step, 1, 80))
        self.redraw()

    def on_key(self, event) -> None:
        key = str(event.key).lower() if event.key else ""
        if key == "1":
            self.state["mode"] = "select"
        elif key in {"2", "x"}:
            self.state["mode"] = "cut"
        elif key in {"3", "e"}:
            self.state["mode"] = "erase"
        elif key in {"4", "b"}:
            self.state["mode"] = "paint"
        elif key == "j":
            self.join_selected()
        elif key == "q":
            self.smooth_selected()
        elif key == "n":
            self.step_id(+1)
        elif key == "p":
            self.step_id(-1)
        elif key in {"+", "=", "]"}:
            self.state["brush"] = min(80, self.state["brush"] + 1)
        elif key in {"-", "_", "["}:
            self.state["brush"] = max(1, self.state["brush"] - 1)
        elif key == "o":
            self.state["outline"] = not self.state["outline"]
        elif key == "l":
            self.state["protect"] = not self.state["protect"]
        elif key == "0":
            self.ax.set_xlim(*self.full_xlim)
            self.ax.set_ylim(*self.full_ylim)
        elif key == "u" and self.undo:
            self.members, hlen = self.undo.pop()
            del self.history[hlen:]
            self.sync()
            self.state["dirty"] = True
        elif key == "s":
            self.save_all()
        elif key in {"escape", "c"}:
            self.state["selected"].clear()
        self.redraw()

    def on_slider(self, _) -> None:
        self.state["alpha"], self.state["dim"], self.state["gain"] = float(self.s_alpha.val), float(self.s_dim.val), float(self.s_gain.val)
        self.redraw()

    def run(self) -> None:
        print(f"[manual] labels: {self.lbl_path}")
        print(f"[manual] background: {self.bg_path}")
        plt.show()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Manual ID correction tool for label TIFFs.")
    ap.add_argument("--input", type=Path, default=None, help="Folder containing labels/background.")
    ap.add_argument("--ask-input", action="store_true", help="Prompt for folder at startup.")
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--brush", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=8)
    ap.add_argument("--thickness", type=int, default=8)
    ap.add_argument("--flatten-rule", choices=["longest", "lowest", "highest", "random"], default="longest")
    ap.add_argument("--save-multilabel-tiff", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overlay_alpha", type=float, default=0.72)
    ap.add_argument("--dim_bg", type=float, default=0.55)
    ap.add_argument("--brighten_gain", type=float, default=1.60)
    args = ap.parse_args()
    args.input = prompt_folder(SCRIPT_INPUT) if args.ask_input or (PROMPT_FOR_INPUT and args.input is None) else (args.input or SCRIPT_INPUT)
    return args


def main() -> None:
    ManualIDEditor(parse_args()).run()


if __name__ == "__main__":
    main()
