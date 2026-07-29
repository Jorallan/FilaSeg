"""Instance / ground-truth / fragment loading for FilaSeg evaluation.

This module is the single place that knows how the pipeline serialises its
results, so the metric code can stay format-agnostic. Three things are loaded:

1. Predicted instances  -> from a reconnect/postprocess/final run folder.
2. Ground-truth instances -> from a label image OR a centerline JSON.
3. Fragments (the pre-reconnect pieces) -> from the per-angle preprocess
   branches. These are the atomic units the reconnect stage groups, and they
   are what the fragment-clustering metric scores.

Predicted instances may overlap (the reconnect multilabel keeps one layer per
ID), so the canonical representation is a dict of {id -> boolean mask}, not a
flat label image. A flat label image is derived on demand with an explicit
priority rule (largest skeleton/area first, matching the pipeline's own
first-write-wins flattening).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


Shape = Tuple[int, int]


# ── canonical instance container ──────────────────────────────────────────


@dataclass
class InstanceSet:
    """A set of (possibly overlapping) labelled instances on one image."""

    shape: Shape
    masks: Dict[int, np.ndarray]  # id -> bool array (H, W)
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.masks)

    def areas(self) -> Dict[int, int]:
        return {i: int(m.sum()) for i, m in self.masks.items()}

    def label_image(self, priority: str = "area") -> np.ndarray:
        """Flatten to a single int label image.

        Overlaps are resolved first-write-wins after sorting instances by
        `priority` ("area" = largest first). This mirrors how the pipeline
        flattens its layered output, so a flattened pred is comparable to a
        flat GT label image.
        """
        lab = np.zeros(self.shape, dtype=np.int32)
        if priority == "area":
            order = sorted(self.masks, key=lambda i: -int(self.masks[i].sum()))
        else:
            order = sorted(self.masks)
        for i in order:
            m = self.masks[i] & (lab == 0)
            lab[m] = int(i)
        return lab

    def filter_min_area(self, min_area: int) -> "InstanceSet":
        if min_area <= 0:
            return self
        kept = {i: m for i, m in self.masks.items() if int(m.sum()) >= min_area}
        return InstanceSet(self.shape, kept, self.source)


# ── predicted instances ───────────────────────────────────────────────────


def load_multilabel_npz(path: Path) -> InstanceSet:
    """Load the CSR-style sparse multilabel written by reconnect/postprocess.

    Keys: shape (2,), ids (N,), indptr (N+1,), indices (flat raveled idx).
    """
    d = np.load(str(path), allow_pickle=False)
    shape = tuple(int(v) for v in d["shape"])  # (H, W)
    ids = [int(v) for v in d["ids"]]
    indptr = np.asarray(d["indptr"]).astype(np.int64)
    indices = np.asarray(d["indices"]).astype(np.int64)
    masks: Dict[int, np.ndarray] = {}
    for k, cid in enumerate(ids):
        flat = indices[indptr[k]:indptr[k + 1]]
        m = np.zeros(shape[0] * shape[1], dtype=bool)
        m[flat] = True
        masks[cid] = m.reshape(shape)
    return InstanceSet(shape, masks, source=str(path))


def load_label_image(path: Path) -> np.ndarray:
    """Read a label image (tif/png). 0 = background, >0 = instance id."""
    from skimage import io as skio

    img = skio.imread(str(path))
    if img.ndim == 3:
        # A colour PNG label preview is ambiguous; refuse rather than guess.
        raise ValueError(
            f"{path} is RGB, not a single-channel label image. "
            "Pass a label .tif (e.g. *_reconnect_labels.tif) or a "
            "single-channel GT mask."
        )
    return img.astype(np.int64)


def instances_from_label_image(lab: np.ndarray, source: str = "") -> InstanceSet:
    masks: Dict[int, np.ndarray] = {}
    for i in np.unique(lab):
        if int(i) == 0:
            continue
        masks[int(i)] = lab == i
    return InstanceSet(lab.shape, masks, source=source)


# Candidate predicted-instance artifacts, most-overlap-aware first.
_PRED_NPZ_GLOBS = (
    "*_reconnect_multilabel.npz",
    "*_post_multilabel.npz",
    "*_multilabel.npz",
)
_PRED_TIF_GLOBS = (
    "*_reconnect_labels.tif",
    "*_post_labels.tif",
    "*_labels.tif",
)


def find_pred_artifact(path: Path) -> Path:
    """Resolve a user-given path (file or run/stage folder) to a pred artifact.

    Preference order: explicit file > multilabel npz (overlap-aware) > flat
    label tif. When given a run root, the `final/` then `3.reconnect/` stage
    folders are searched.
    """
    path = Path(path)
    if path.is_file():
        return path
    search_dirs: List[Path] = []
    if path.is_dir():
        # A run root: look in the conventional handoff/stage subfolders too.
        for sub in ("final", "3.reconnect", "4.postprocess"):
            if (path / sub).is_dir():
                search_dirs.append(path / sub)
        search_dirs.append(path)
    for d in search_dirs:
        for g in _PRED_NPZ_GLOBS:
            hits = sorted(d.glob(g))
            if hits:
                return hits[0]
    for d in search_dirs:
        for g in _PRED_TIF_GLOBS:
            hits = sorted(d.glob(g))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No predicted-instance artifact found under {path}")


def load_pred_instances(path: Path) -> InstanceSet:
    art = find_pred_artifact(path)
    if art.suffix.lower() == ".npz":
        return load_multilabel_npz(art)
    return instances_from_label_image(load_label_image(art), source=str(art))


# ── ground truth ──────────────────────────────────────────────────────────


def rasterize_centerlines(
    filaments: List[dict],
    shape: Shape,
    default_width_px: float = 5.0,
) -> np.ndarray:
    """Rasterize centerline polylines into a thick label image.

    Each filament: {"id": int, "centerline_xy": [[x,y],...], "width_px": opt}.
    Coordinates are (x=col, y=row). Lines are drawn then dilated to the given
    width so that downstream fragment-overlap assignment has real area to vote
    on. IDs are taken from the filament dicts (1-based recommended).
    """
    from skimage.draw import line as skline
    from scipy.ndimage import binary_dilation

    lab = np.zeros(shape, dtype=np.int32)
    H, W = shape
    for fi, fil in enumerate(filaments):
        fid = int(fil.get("id", fi + 1))
        pts = np.asarray(fil.get("centerline_xy", []), dtype=float)
        if pts.shape[0] == 0:
            continue
        width = float(fil.get("width_px", default_width_px))
        rad = max(0, int(round(width / 2.0)))
        strokes = np.zeros(shape, dtype=bool)
        if pts.shape[0] == 1:
            r, c = int(round(pts[0, 1])), int(round(pts[0, 0]))
            if 0 <= r < H and 0 <= c < W:
                strokes[r, c] = True
        for i in range(len(pts) - 1):
            r0, c0 = int(round(pts[i, 1])), int(round(pts[i, 0]))
            r1, c1 = int(round(pts[i + 1, 1])), int(round(pts[i + 1, 0]))
            r0 = min(max(r0, 0), H - 1); r1 = min(max(r1, 0), H - 1)
            c0 = min(max(c0, 0), W - 1); c1 = min(max(c1, 0), W - 1)
            rr, cc = skline(r0, c0, r1, c1)
            strokes[rr, cc] = True
        if rad > 0:
            strokes = binary_dilation(strokes, iterations=rad)
        # First-write-wins keeps earlier filaments intact at crossings.
        lab[strokes & (lab == 0)] = fid
    return lab


def load_gt(path: Path, shape: Optional[Shape] = None) -> Tuple[InstanceSet, dict]:
    """Load ground truth from a label image or a centerline JSON.

    Returns (InstanceSet, meta). meta records {"kind": "label"|"centerline",
    "has_area": bool}. For centerline GT the instances are thick rasterizations
    (has_area=True for overlap assignment, but pixel-IoU metrics should be
    interpreted with care).
    """
    path = Path(path)
    # Prefer an overlap-aware GT multilabel (each instance owns its full stroke)
    # when one sits beside the requested label image, or when given directly.
    if path.suffix.lower() == ".npz":
        return load_multilabel_npz(path), {"kind": "label_overlap", "has_area": True}
    sibling = path.with_name("gt_multilabel.npz")
    if path.suffix.lower() in (".tif", ".tiff", ".png") and sibling.exists():
        return load_multilabel_npz(sibling), {"kind": "label_overlap", "has_area": True}
    if path.suffix.lower() == ".json":
        spec = json.loads(path.read_text(encoding="utf-8"))
        gshape = tuple(spec.get("image_shape_rc", shape or (0, 0)))
        if gshape == (0, 0):
            raise ValueError(
                "Centerline GT JSON needs 'image_shape_rc' or pass shape=..."
            )
        lab = rasterize_centerlines(
            spec.get("filaments", []), gshape,
            default_width_px=float(spec.get("default_width_px", 5.0)),
        )
        iset = instances_from_label_image(lab, source=str(path))
        return iset, {"kind": "centerline", "has_area": True}
    lab = load_label_image(path)
    iset = instances_from_label_image(lab, source=str(path))
    return iset, {"kind": "label", "has_area": True}


# ── fragments (pre-reconnect pieces) ──────────────────────────────────────


def _cc(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    from skimage.measure import label as cc_label

    lbl = cc_label(mask, connectivity=2)
    out: List[np.ndarray] = []
    for k in range(1, int(lbl.max()) + 1):
        m = lbl == k
        if int(m.sum()) >= int(min_area):
            out.append(m)
    return out


def load_fragments(
    run_dir: Path,
    min_area: int = 15,
    branch_bin: float = 0.5,
) -> List[np.ndarray]:
    """Load the atomic pieces that enter reconnect, as a list of bool masks.

    Faithful source = per-angle preprocess branches (each connected component
    is one reconnect Component). Fallbacks: a merged branch preview, then the
    stringart reconstruction. Fragments from different angle branches may
    overlap spatially; that is expected and correct (they are distinct pieces).
    """
    from skimage import io as skio

    run_dir = Path(run_dir)
    pre_branches = run_dir / "2.preprocess" / "branches"
    frags: List[np.ndarray] = []

    if pre_branches.is_dir():
        branch_files = sorted(
            p for p in pre_branches.glob("*branch_*.png")
        )
        for bf in branch_files:
            img = skio.imread(str(bf))
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            frags.extend(_cc(img >= branch_bin, min_area))
        if frags:
            return frags

    # Fallback: a single merged mask -> connected components.
    for cand in (
        pre_branches / "mask_branches_merge.png",
        run_dir / "1.stringart" / "reconstructed.png",
    ):
        if cand.is_file():
            img = skio.imread(str(cand))
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            return _cc(img >= branch_bin, min_area)

    return frags


def fragment_label_image(fragments: List[np.ndarray], shape: Shape) -> np.ndarray:
    """Stamp fragments into a label image (1-based) for visualisation/debug.

    Overlapping branch fragments are resolved largest-first; this image is for
    inspection only — metrics operate on the fragment list directly.
    """
    lab = np.zeros(shape, dtype=np.int32)
    for k, m in enumerate(sorted(fragments, key=lambda x: -int(x.sum())), start=1):
        lab[m & (lab == 0)] = k
    return lab
