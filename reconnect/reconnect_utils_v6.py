
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from scipy.ndimage import binary_dilation, convolve
from skimage import io as skio
from skimage.color import gray2rgb
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize

Coord = Tuple[int, int]  # (row, col)


# ============================================================
# I/O and basic masks
# ============================================================

def read_gray_any(path: Path) -> np.ndarray:
    """Read any image format into float32 grayscale in [0,1]."""
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0 if img.max() <= 255.0 else max(1.0, float(img.max()))
    return np.clip(img, 0.0, 1.0)


def binarize_mask(img01: np.ndarray, thr: float) -> np.ndarray:
    return (img01 >= thr).astype(bool)


def clean_binary_mask(m: np.ndarray, min_size: int) -> np.ndarray:
    m = m.astype(bool)
    min_size = int(min_size)
    if min_size <= 0:
        return m
    lbl = cc_label(m, connectivity=2)
    if lbl.max() == 0:
        return m
    counts = np.bincount(lbl.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[lbl]


# ============================================================
# Branchpoints / optional splitting
# ============================================================

def branchpoints_8(skel: np.ndarray) -> np.ndarray:
    sk = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nb = convolve(sk, kernel, mode="constant", cval=0)
    return (sk == 1) & (nb >= 3)


def remove_branchpoint_regions(mask: np.ndarray, merged_branchpoints: np.ndarray, bp_dilate_px: int) -> np.ndarray:
    if merged_branchpoints is None:
        return mask.astype(bool)
    bp = merged_branchpoints.astype(bool)
    if int(bp_dilate_px) > 0:
        bp = binary_dilation(bp, iterations=int(bp_dilate_px))
    return np.logical_and(mask.astype(bool), ~bp)


# ============================================================
# Skeleton helpers
# ============================================================

def endpoints_from_skeleton(skel: np.ndarray) -> np.ndarray:
    sk = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nb = convolve(sk, kernel, mode="constant", cval=0)
    return np.argwhere((sk == 1) & (nb == 1))


def _neighbors8(skel: np.ndarray, p: Coord) -> List[Coord]:
    r, c = int(p[0]), int(p[1])
    out: List[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                out.append((rr, cc))
    out.sort(key=lambda x: (x[1], x[0]))
    return out


def _choose_tip_pair(skel: np.ndarray, fallback_mask: np.ndarray) -> Tuple[Coord, Coord]:
    eps = endpoints_from_skeleton(skel)
    if eps.shape[0] >= 2:
        best_i = 0
        best_j = 1
        best_d2 = -1.0
        for i in range(eps.shape[0]):
            di = eps[i]
            for j in range(i + 1, eps.shape[0]):
                dj = eps[j]
                d2 = float((di[0] - dj[0]) ** 2 + (di[1] - dj[1]) ** 2)
                if d2 > best_d2:
                    best_d2 = d2
                    best_i, best_j = i, j
        return tuple(int(v) for v in eps[best_i]), tuple(int(v) for v in eps[best_j])

    pts = np.argwhere(skel if np.count_nonzero(skel) else fallback_mask)
    if pts.shape[0] == 0:
        return (0, 0), (0, 0)
    if pts.shape[0] == 1:
        p = tuple(int(v) for v in pts[0])
        return p, p

    xy = np.stack([pts[:, 1].astype(np.float32), pts[:, 0].astype(np.float32)], axis=1)
    ctr = xy.mean(axis=0, keepdims=True)
    X = xy - ctr
    cov = (X.T @ X) / max(1, len(xy) - 1)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    proj = X @ axis
    p0 = pts[int(np.argmin(proj))]
    p1 = pts[int(np.argmax(proj))]
    return tuple(int(v) for v in p0), tuple(int(v) for v in p1)


def trace_from_tip(skel: np.ndarray, tip: Coord, max_steps: int) -> np.ndarray:
    """
    Ordered tip trace: [tip, ..., inward]. Stops at branch / dead-end / max_steps.
    This is intentionally local; for gating we only need the first part of the centerline.
    """
    if not (0 <= tip[0] < skel.shape[0] and 0 <= tip[1] < skel.shape[1]):
        return np.asarray([[0.0, 0.0]], dtype=np.float32)

    if not skel[tip]:
        pts = np.argwhere(skel)
        if pts.shape[0] == 0:
            return np.asarray([[float(tip[0]), float(tip[1])]], dtype=np.float32)
        d2 = np.sum((pts - np.asarray(tip)[None, :]) ** 2, axis=1)
        tip = tuple(int(v) for v in pts[int(np.argmin(d2))])

    def _support_after_step(prev0: Coord, cur0: Coord, depth: int = 6) -> float:
        prev_i = prev0
        cur_i = cur0
        acc = 0.0
        for _ in range(max(1, int(depth))):
            nbs_i = [p for p in _neighbors8(skel, cur_i) if p != prev_i]
            if not nbs_i:
                break
            if len(nbs_i) == 1:
                nxt_i = nbs_i[0]
            else:
                ref = np.asarray([cur_i[0] - prev_i[0], cur_i[1] - prev_i[1]], dtype=np.float32)
                best = None
                nxt_i = nbs_i[0]
                for cand_i in nbs_i:
                    step = np.asarray([cand_i[0] - cur_i[0], cand_i[1] - cur_i[1]], dtype=np.float32)
                    score = float(np.dot(ref, step))
                    tie = (score, -abs(cand_i[0] - prev_i[0]) - abs(cand_i[1] - prev_i[1]), -cand_i[1], -cand_i[0])
                    if best is None or tie > best:
                        best = tie
                        nxt_i = cand_i
            acc += float(np.hypot(nxt_i[0] - cur_i[0], nxt_i[1] - cur_i[1]))
            prev_i, cur_i = cur_i, nxt_i
        return acc

    trace: List[Coord] = [tip]
    prev: Optional[Coord] = None
    cur: Coord = tip
    n_steps = max(1, int(max_steps))

    for _ in range(n_steps):
        nbs = _neighbors8(skel, cur)
        if prev is not None:
            nbs = [p for p in nbs if p != prev]
        if not nbs:
            break
        if prev is None:
            if len(nbs) == 1:
                nxt = nbs[0]
            else:
                best = None
                nxt = nbs[0]
                for cand in nbs:
                    score = _support_after_step(cur, cand, depth=6)
                    tie = (score, -cand[1], -cand[0])
                    if best is None or tie > best:
                        best = tie
                        nxt = cand
        else:
            prev_vec = np.asarray([cur[0] - prev[0], cur[1] - prev[1]], dtype=np.float32)
            best_score = None
            nxt = nbs[0]
            for cand in nbs:
                cand_vec = np.asarray([cand[0] - cur[0], cand[1] - cur[1]], dtype=np.float32)
                score = float(np.dot(prev_vec, cand_vec))
                support = _support_after_step(cur, cand, depth=5)
                tie = (score, support, -abs(cand[0] - prev[0]) - abs(cand[1] - prev[1]), -cand[1], -cand[0])
                if best_score is None or tie > best_score:
                    best_score = tie
                    nxt = cand
        trace.append(nxt)
        prev, cur = cur, nxt

    return np.asarray(trace, dtype=np.float32)


def _fit_local_tangent_and_curvature(trace_rc: np.ndarray, fit_points: int) -> Tuple[np.ndarray, float]:
    """
    Robust local geometry from ordered centerline points.
    Tangent is estimated in a local PCA frame and signed tip->inward.
    Curvature is a quadratic residual in the local normal direction.
    """
    if trace_rc is None or len(trace_rc) == 0:
        return np.asarray([0.0, 0.0], dtype=np.float32), 0.0

    pts = np.asarray(trace_rc[:max(2, int(fit_points))], dtype=np.float32)
    if pts.shape[0] < 2:
        return np.asarray([0.0, 0.0], dtype=np.float32), 0.0

    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
    ctr = xy.mean(axis=0, keepdims=True)
    X = xy - ctr
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    axis_xy = v[:, int(np.argmax(w))]
    ordered_delta_xy = xy[-1] - xy[0]
    if float(np.dot(axis_xy, ordered_delta_xy)) < 0.0:
        axis_xy = -axis_xy

    tangent_rc = np.asarray([axis_xy[1], axis_xy[0]], dtype=np.float32)
    nt = float(np.linalg.norm(tangent_rc))
    if nt < 1e-9:
        tangent_rc = np.asarray([pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]], dtype=np.float32)
        nt = float(np.linalg.norm(tangent_rc))
        if nt < 1e-9:
            return np.asarray([0.0, 0.0], dtype=np.float32), 0.0
    tangent_rc /= nt

    # Local polynomial fit in the PCA frame.
    axis_xy = np.asarray([tangent_rc[1], tangent_rc[0]], dtype=np.float32)
    normal_xy = np.asarray([-axis_xy[1], axis_xy[0]], dtype=np.float32)
    rel = xy - xy[0:1]
    s = rel @ axis_xy
    n = rel @ normal_xy

    # Normalize arc coordinate for numerical stability.
    s0 = s - s.min()
    span = float(max(1e-6, s0.max()))
    su = s0 / span

    deg = 2 if len(su) >= 5 and np.unique(np.round(su, 4)).size >= 3 else 1
    try:
        coeff = np.polyfit(su, n, deg=deg)
        pred = np.polyval(coeff, su)
        # For n(s)=a*s^2+b*s+c on normalized coordinate, convert second derivative back using span.
        if deg == 2:
            a = float(coeff[0])
            curvature = abs(2.0 * a) / (span * span + 1e-9)
        else:
            curvature = 0.0
        rms = float(np.sqrt(np.mean((n - pred) ** 2)))
        curvature += 0.10 * rms / (span + 1e-9)
    except Exception:
        curvature = 0.0

    return tangent_rc.astype(np.float32), float(curvature)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def draw_gap(p1: Coord, p2: Coord, shape: Tuple[int, int]) -> np.ndarray:
    r0, c0 = p1
    r1, c1 = p2
    n = int(max(abs(r1 - r0), abs(c1 - c0)) + 1)
    rr = np.round(np.linspace(r0, r1, n)).astype(int)
    cc = np.round(np.linspace(c0, c1, n)).astype(int)
    rr = np.clip(rr, 0, shape[0] - 1)
    cc = np.clip(cc, 0, shape[1] - 1)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m


def _polyline_mask(points_rc: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if points_rc is None or len(points_rc) == 0:
        return out
    pts = np.asarray(points_rc, dtype=np.float32)
    for i in range(len(pts) - 1):
        p0 = (int(round(float(pts[i, 0]))), int(round(float(pts[i, 1]))))
        p1 = (int(round(float(pts[i + 1, 0]))), int(round(float(pts[i + 1, 1]))))
        out |= draw_gap(p0, p1, shape)
    if len(pts) == 1:
        r, c = int(round(float(pts[0, 0]))), int(round(float(pts[0, 1])))
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = True
    return out


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    if pts.shape[0] == 0:
        return 0, -1, 0, -1
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _bbox_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _component_union_label(components: List["Component"]) -> np.ndarray:
    if not components:
        return np.zeros((1, 1), dtype=np.int32)
    shape = components[0].mask.shape
    lab = np.zeros(shape, dtype=np.int32)
    for c in sorted([x for x in components if x.exist], key=lambda z: z.id):
        lab[c.mask] = int(c.id)
    return lab


def _sample_labels_in_neighborhood(label_img: np.ndarray, tip: Coord, radius: int) -> np.ndarray:
    r0, c0 = int(tip[0]), int(tip[1])
    rr, cc = np.mgrid[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
    m = (rr >= 0) & (rr < label_img.shape[0]) & (cc >= 0) & (cc < label_img.shape[1])
    if not np.any(m):
        return np.asarray([0], dtype=label_img.dtype)
    return label_img[rr[m], cc[m]]




def _tip_trace_length(trace_rc: Optional[np.ndarray]) -> float:
    if trace_rc is None or len(trace_rc) < 2:
        return 0.0
    pts = np.asarray(trace_rc, dtype=np.float32)
    dif = np.diff(pts, axis=0)
    seg = np.sqrt(np.sum(dif * dif, axis=1))
    return float(seg.sum())


def _dedupe_tip_points(points: List[Coord], min_sep_px: float = 2.0) -> List[Coord]:
    out: List[Coord] = []
    thr2 = float(min_sep_px) * float(min_sep_px)
    for p in points:
        keep = True
        for q in out:
            d2 = float((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            if d2 <= thr2:
                keep = False
                break
        if keep:
            out.append(p)
    return out


def _enumerate_tip_geoms(
    skel: np.ndarray,
    fallback_mask: np.ndarray,
    trace_steps: int,
    fit_points: int,
    max_tips: int = 8,
    min_tip_trace_len: float = 3.0,
    dedupe_sep_px: float = 2.0,
) -> Dict[str, Dict[str, object]]:
    eps = [tuple(int(v) for v in p) for p in endpoints_from_skeleton(skel)]
    if len(eps) == 0:
        p0, p1 = _choose_tip_pair(skel, fallback_mask)
        eps = [p0] if p0 == p1 else [p0, p1]
    else:
        eps = _dedupe_tip_points(eps, min_sep_px=dedupe_sep_px)

    tip_items = []
    for p in eps:
        trace = trace_from_tip(skel, p, trace_steps)
        dvec, curv = _fit_local_tangent_and_curvature(trace, fit_points)
        support = _tip_trace_length(trace)
        tip_items.append({
            "point": p,
            "trace": trace,
            "dir": dvec,
            "curv": float(curv),
            "support": float(support),
        })

    if not tip_items:
        p0, p1 = _choose_tip_pair(skel, fallback_mask)
        for p in ([p0] if p0 == p1 else [p0, p1]):
            trace = trace_from_tip(skel, p, trace_steps)
            dvec, curv = _fit_local_tangent_and_curvature(trace, fit_points)
            tip_items.append({
                "point": p,
                "trace": trace,
                "dir": dvec,
                "curv": float(curv),
                "support": float(_tip_trace_length(trace)),
            })

    good = [t for t in tip_items if float(t["support"]) >= float(min_tip_trace_len)]
    if good:
        tip_items = good

    # Keep all meaningful endpoints, but cap pathological components.
    tip_items.sort(
        key=lambda t: (
            -float(t["support"]),
            float(np.linalg.norm(np.asarray(t["dir"], dtype=np.float32))),
            -int(t["point"][1]),
            -int(t["point"][0]),
        )
    )
    tip_items = tip_items[: max(2, int(max_tips))]

    out: Dict[str, Dict[str, object]] = {}
    for i, t in enumerate(tip_items):
        name = f"t{i}"
        out[name] = t
    return out

# ============================================================
# Component object
# ============================================================

@dataclass
class Component:
    id: int
    layer: int
    mask: np.ndarray
    exist: bool = True

    skel: Optional[np.ndarray] = None
    lt: Coord = (0, 0)
    rt: Coord = (0, 0)
    ltrace: Optional[np.ndarray] = None
    rtrace: Optional[np.ndarray] = None
    ld: Optional[np.ndarray] = None  # inward tip direction
    rd: Optional[np.ndarray] = None
    lcurv: float = 0.0
    rcurv: float = 0.0
    skel_len: float = 0.0
    mean_width: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, -1, 0, -1)
    tips: Dict[str, Dict[str, object]] = None

    def refresh_geom(self, trace_steps: int, fit_points: int):
        self.skel = skeletonize(self.mask)
        self.tips = _enumerate_tip_geoms(
            self.skel,
            self.mask,
            trace_steps=trace_steps,
            fit_points=fit_points,
            max_tips=8,
            min_tip_trace_len=3.0,
            dedupe_sep_px=2.0,
        )

        names = list(self.tips.keys())
        if len(names) >= 1:
            t0 = self.tips[names[0]]
            self.lt = t0["point"]
            self.ltrace = t0["trace"]
            self.ld = np.asarray(t0["dir"], dtype=np.float32)
            self.lcurv = float(t0["curv"])
        else:
            self.lt = (0, 0)
            self.ltrace = np.asarray([[0.0, 0.0]], dtype=np.float32)
            self.ld = np.asarray([0.0, 0.0], dtype=np.float32)
            self.lcurv = 0.0

        if len(names) >= 2:
            t1 = self.tips[names[1]]
            self.rt = t1["point"]
            self.rtrace = t1["trace"]
            self.rd = np.asarray(t1["dir"], dtype=np.float32)
            self.rcurv = float(t1["curv"])
        else:
            self.rt = self.lt
            self.rtrace = self.ltrace
            self.rd = self.ld.copy()
            self.rcurv = float(self.lcurv)

        self.skel_len = float(max(1, np.count_nonzero(self.skel)))
        self.mean_width = float(max(1.0, float(np.count_nonzero(self.mask)) / self.skel_len))
        self.bbox = _bbox_from_mask(self.mask)


# ============================================================
# Extraction and rendering
# ============================================================

def extract_components(
    branch_bin: np.ndarray,
    layer: int,
    min_area: int,
    start_id: int,
    *,
    skeletonize_each: bool = False
) -> Tuple[List[Component], int]:
    lbl = cc_label(branch_bin, connectivity=2)
    comps: List[Component] = []
    cid = start_id
    for k in range(1, lbl.max() + 1):
        m = (lbl == k)
        if int(m.sum()) < int(min_area):
            continue
        if skeletonize_each and not np.any(skeletonize(m)):
            continue
        comps.append(Component(id=cid, layer=layer, mask=m.astype(bool)))
        cid += 1
    return comps, cid


def relabel_components(components: List[Component]) -> np.ndarray:
    if not components:
        return None
    shape = components[0].mask.shape
    out = np.zeros(shape, dtype=np.int32)
    new_id = 1
    for c in components:
        if c.exist:
            out[c.mask] = new_id
            new_id += 1
    return out


def dilate_label_image(lbl: np.ndarray, dilate_px: int) -> np.ndarray:
    if lbl is None:
        return None
    out = np.zeros_like(lbl)
    for k in range(1, int(lbl.max()) + 1):
        m = (lbl == k)
        d = binary_dilation(m, iterations=int(dilate_px))
        out[d] = k
    return out


def make_overlay(background01: np.ndarray, label_img: np.ndarray, alpha: float) -> np.ndarray:
    bg = (np.clip(background01, 0, 1) * 255.0).astype(np.uint8)
    bg_rgb = gray2rgb(bg) if bg.ndim == 2 else bg[..., :3]
    if label_img is None:
        return bg_rgb
    lab = label_img.astype(np.int64)
    color = np.zeros((*lab.shape, 3), dtype=np.uint8)
    color[..., 0] = (lab * 53) % 256
    color[..., 1] = (lab * 97) % 256
    color[..., 2] = (lab * 193) % 256
    color[lab == 0] = 0
    out = (alpha * color + (1.0 - alpha) * bg_rgb).astype(np.uint8)
    return out


# ============================================================
# Smooth bridge and proposal scoring
# ============================================================

@dataclass
class Proposal:
    base_id: int
    tar_id: int
    base_tip_name: str
    tar_tip_name: str
    bridge_points: np.ndarray
    score: Tuple[float, ...]
    metrics: Dict[str, float]


def _tip_data(comp: Component, name: str) -> Tuple[Coord, np.ndarray, float, np.ndarray]:
    if isinstance(getattr(comp, "tips", None), dict) and name in comp.tips:
        t = comp.tips[name]
        return (
            tuple(int(v) for v in t["point"]),
            np.asarray(t["dir"], dtype=np.float32),
            float(t["curv"]),
            np.asarray(t["trace"], dtype=np.float32),
        )
    if name == "l":
        return comp.lt, comp.ld, float(comp.lcurv), comp.ltrace
    return comp.rt, comp.rd, float(comp.rcurv), comp.rtrace


def _sample_hermite_bridge(
    p0_rc: Coord,
    p1_rc: Coord,
    t0_rc: np.ndarray,
    t1_rc: np.ndarray,
    n_samples: int,
    tangent_scale: float,
) -> np.ndarray:
    p0 = np.asarray([float(p0_rc[0]), float(p0_rc[1])], dtype=np.float32)
    p1 = np.asarray([float(p1_rc[0]), float(p1_rc[1])], dtype=np.float32)
    dist = float(np.linalg.norm(p1 - p0))
    scale = max(1.0, tangent_scale * dist)
    m0 = np.asarray(t0_rc, dtype=np.float32) * scale
    m1 = np.asarray(t1_rc, dtype=np.float32) * scale

    ts = np.linspace(0.0, 1.0, int(max(8, n_samples)), dtype=np.float32)
    pts = []
    for t in ts:
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        p = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        pts.append(p)
    return np.asarray(pts, dtype=np.float32)


def _perp_distance_point_to_line(point_rc: Coord, line_point_rc: Coord, line_dir_rc: np.ndarray) -> float:
    d = np.asarray(line_dir_rc, dtype=np.float32)
    nd = float(np.linalg.norm(d))
    if nd < 1e-9:
        return 1e9
    d /= nd
    p = np.asarray([float(point_rc[0] - line_point_rc[0]), float(point_rc[1] - line_point_rc[1])], dtype=np.float32)
    proj = float(np.dot(p, d))
    q = p - proj * d
    return float(np.linalg.norm(q))


def _local_smoothness_metrics(
    base_trace: np.ndarray,
    bridge_pts: np.ndarray,
    tar_trace: np.ndarray,
    fit_degree: int = 2,
) -> Tuple[float, float, float]:
    """
    Smoothness of the *local merged centerline* around the bridge.
    Returns:
      rms_residual_px, quadratic_curvature_proxy, max_turn_deg
    """
    pts_parts = []

    if base_trace is not None and len(base_trace) > 0:
        bt = np.asarray(base_trace, dtype=np.float32)
        pts_parts.append(bt[::-1])
    if bridge_pts is not None and len(bridge_pts) > 0:
        pts_parts.append(np.asarray(bridge_pts, dtype=np.float32)[1:-1])
    if tar_trace is not None and len(tar_trace) > 0:
        tt = np.asarray(tar_trace, dtype=np.float32)
        pts_parts.append(tt)

    if not pts_parts:
        return 1e9, 1e9, 180.0

    pts = np.concatenate(pts_parts, axis=0)
    if len(pts) >= 3:
        # Light de-staircasing for rasterized diagonals; avoids artificial 90° local turns.
        sm = pts.copy()
        for i in range(1, len(pts) - 1):
            sm[i] = 0.25 * pts[i - 1] + 0.50 * pts[i] + 0.25 * pts[i + 1]
        pts = sm
    if len(pts) < 4:
        return 1e9, 1e9, 180.0

    # Drop exact duplicates.
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 0.25:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 4:
        return 1e9, 1e9, 180.0

    # Polyline turning.
    max_turn_deg = 0.0
    if len(pts) >= 3:
        for i in range(1, len(pts) - 1):
            v0 = pts[i] - pts[i - 1]
            v1 = pts[i + 1] - pts[i]
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 < 1e-6 or n1 < 1e-6:
                continue
            cs = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cs)))
            if ang > max_turn_deg:
                max_turn_deg = ang

    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
    ctr = xy.mean(axis=0, keepdims=True)
    X = xy - ctr
    cov = (X.T @ X) / max(1, len(X) - 1)
    w, v = np.linalg.eigh(cov)
    axis_xy = v[:, int(np.argmax(w))]
    normal_xy = np.asarray([-axis_xy[1], axis_xy[0]], dtype=np.float32)
    s = X @ axis_xy
    n = X @ normal_xy

    # Use actual chain order for fitting via cumulative chord length, not sorted PCA projection.
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    chord = np.concatenate([[0.0], np.cumsum(seg)])
    span = float(max(1e-6, chord[-1] - chord[0]))
    su = (chord - chord[0]) / span

    deg = 2 if fit_degree >= 2 and len(su) >= 5 and np.unique(np.round(su, 4)).size >= 3 else 1
    try:
        coeff = np.polyfit(su, n, deg=deg)
        pred = np.polyval(coeff, su)
        rms = float(np.sqrt(np.mean((n - pred) ** 2)))
        if deg == 2:
            a = float(coeff[0])
            curv = abs(2.0 * a) / (span * span + 1e-9)
        else:
            curv = 0.0
    except Exception:
        rms = 1e9
        curv = 1e9

    return rms, curv, max_turn_deg


def _evaluate_tip_pair(
    base: Component,
    tar: Component,
    base_tip_name: str,
    tar_tip_name: str,
    global_label: np.ndarray,
    cfg: Dict,
) -> Optional[Proposal]:
    thr = cfg.get("thresholds", {})
    adv = cfg.get("advanced", {})
    wgt = cfg.get("weights", {})
    morph = cfg.get("morphology", {})

    bp, bd, bcurv, btrace = _tip_data(base, base_tip_name)
    tp, td, tcurv, ttrace = _tip_data(tar, tar_tip_name)

    bp_arr = np.asarray([float(bp[0]), float(bp[1])], dtype=np.float32)
    tp_arr = np.asarray([float(tp[0]), float(tp[1])], dtype=np.float32)
    min_tip_support = float(adv.get("min_tip_trace_len_px", 3.0))
    if _tip_trace_length(btrace) < min_tip_support or _tip_trace_length(ttrace) < min_tip_support:
        return None
    dvec = tp_arr - bp_arr
    dist = float(np.linalg.norm(dvec))
    max_tip_dist = float(thr.get("max_tip_distance_px", 40))
    if dist <= 1e-6 or dist > max_tip_dist:
        return None
    u = dvec / dist

    # Geometry gates.
    inward_opposition = float(cosine(bd, td))         # good ~ -1
    forward_base = float(cosine(-bd, u))              # good ~ +1
    forward_tar = float(cosine(td, u))                # good ~ +1
    width_ratio = float(max(base.mean_width, tar.mean_width) / max(1e-6, min(base.mean_width, tar.mean_width)))
    line_resid_base = _perp_distance_point_to_line(tp, bp, -bd)
    line_resid_tar  = _perp_distance_point_to_line(bp, tp, td)

    # Use a softer symmetric residual.
    # max(...) is too harsh for genuine curved continuations.
    line_resid = 0.5 * (line_resid_base + line_resid_tar)
    curv_delta = float(abs(bcurv - tcurv))

    min_forward_cos = float(thr.get("min_forward_cos", 0.65))
    min_inward_opposition = float(thr.get("min_inward_opposition", 0.60))
    max_width_ratio = float(thr.get("max_width_ratio", 3.0))
    max_line_residual = float(thr.get("max_line_residual_px", 8.0))
    max_curv_delta = float(thr.get("max_curvature_delta", 0.15))

    if forward_base < min_forward_cos or forward_tar < min_forward_cos:
        return None
    if (-inward_opposition) < min_inward_opposition:
        return None
    if width_ratio > max_width_ratio:
        return None
    if line_resid > max_line_residual:
        return None
    if curv_delta > max_curv_delta:
        return None

    bridge_pts = _sample_hermite_bridge(
        bp,
        tp,
        -bd,   # outward continuation from base tip
        td,    # inward continuation into target
        n_samples=int(adv.get("bridge_samples", 24)),
        tangent_scale=float(adv.get("bridge_tangent_scale", 0.35)),
    )
    bridge_mask = _polyline_mask(bridge_pts, base.mask.shape)

    clearance_px = int(max(0, adv.get("bridge_clearance_px", 1)))
    bridge_eval = binary_dilation(bridge_mask, iterations=clearance_px) if clearance_px > 0 else bridge_mask
    touched = global_label[bridge_eval]
    touched = touched[(touched != 0) & (touched != base.id) & (touched != tar.id)]
    intrusion_frac = float(len(np.unique(touched))) / max(1.0, dist)
    max_intrusion = float(thr.get("max_bridge_intrusion_frac", 0.10))
    if intrusion_frac > max_intrusion:
        return None

    smooth_rms, smooth_curv, max_turn_deg = _local_smoothness_metrics(
        btrace[: int(adv.get("smooth_trace_points", 18))] if btrace is not None else btrace,
        bridge_pts,
        ttrace[: int(adv.get("smooth_trace_points", 18))] if ttrace is not None else ttrace,
        fit_degree=2,
    )
    max_smooth_rms = float(thr.get("max_smooth_rms_px", 3.5))
    max_turn_gate = float(thr.get("max_turn_deg", 45.0))
    if smooth_rms > max_smooth_rms:
        return None
    if max_turn_deg > max_turn_gate:
        return None

    # Lower is better.
    score_scalar = (
        float(wgt.get("distance", 1.0)) * (dist / max(1.0, max_tip_dist))
        + float(wgt.get("forward", 1.2)) * ((1.0 - forward_base) + (1.0 - forward_tar))
        + float(wgt.get("opposition", 1.0)) * (1.0 + inward_opposition)
        + float(wgt.get("line_residual", 0.9)) * (line_resid / max(1.0, max_line_residual))
        + float(wgt.get("width_ratio", 0.5)) * math.log(max(1.0, width_ratio))
        + float(wgt.get("curvature_delta", 0.5)) * (curv_delta / max(1e-6, max_curv_delta))
        + float(wgt.get("smooth_rms", 1.2)) * (smooth_rms / max(1e-6, max_smooth_rms))
        + float(wgt.get("smooth_curvature", 0.5)) * smooth_curv
        + float(wgt.get("bridge_intrusion", 1.5)) * intrusion_frac
        - float(wgt.get("length_reward", 0.02)) * math.log1p(max(base.skel_len, tar.skel_len))
    )

    metrics = {
        "dist": dist,
        "forward_base": forward_base,
        "forward_tar": forward_tar,
        "inward_opposition": inward_opposition,
        "width_ratio": width_ratio,
        "line_residual": line_resid,
        "curv_delta": curv_delta,
        "intrusion_frac": intrusion_frac,
        "smooth_rms": smooth_rms,
        "smooth_curv": smooth_curv,
        "max_turn_deg": max_turn_deg,
    }

    score = (
        float(score_scalar),
        float(smooth_rms),
        float(dist),
        float(line_resid),
        int(min(base.id, tar.id)),
        int(max(base.id, tar.id)),
        0 if base_tip_name == "l" else 1,
        0 if tar_tip_name == "l" else 1,
    )
    return Proposal(
        base_id=base.id,
        tar_id=tar.id,
        base_tip_name=base_tip_name,
        tar_tip_name=tar_tip_name,
        bridge_points=bridge_pts,
        score=score,
        metrics=metrics,
    )


# ============================================================
# Reconnection engine
# ============================================================

def reconnect_components(components: List[Component], cfg: Dict) -> List[Component]:
    thr = cfg.get("thresholds", {})
    runtime = cfg.get("runtime", {})
    morph = cfg.get("morphology", {})
    adv = cfg.get("advanced", {})

    verbose = bool(runtime.get("verbose", True))
    max_passes = int(runtime.get("max_passes", 200))
    max_merges_per_pass = int(runtime.get("max_merges_per_pass", 100))
    print_every_pass = int(runtime.get("print_every_pass", 1))
    print_every_merge = int(runtime.get("print_every_merge", 10))

    trace_steps = int(max(4, adv.get("trace_steps", morph.get("tip_dir_steps", 20))))
    fit_points = int(max(4, adv.get("fit_points", 12)))
    search_size = int(thr.get("search_size_px", 25))
    candidate_layers = str(thr.get("candidate_layers", "all")).lower().strip()
    overlap_kill_thr = float(thr.get("overlap_kill_thr", 0.70))

    for c in components:
        c.refresh_geom(trace_steps, fit_points)

    def _alive() -> List[Component]:
        return [c for c in components if c.exist]

    def _id2comp() -> Dict[int, Component]:
        return {c.id: c for c in components if c.exist}

    def _build_layer_labels() -> Dict[int, np.ndarray]:
        alive = _alive()
        if not alive:
            return {}
        shape = alive[0].mask.shape
        out: Dict[int, np.ndarray] = {}
        for lay in sorted({c.layer for c in alive}):
            lab = np.zeros(shape, dtype=np.int32)
            for c in sorted([x for x in alive if x.layer == lay], key=lambda z: z.id):
                lab[c.mask] = int(c.id)
            out[lay] = lab
        return out

    def _candidate_ids(base: Component, layer_labels: Dict[int, np.ndarray]) -> List[int]:
        cands: set[int] = set()
        lay_ids = [base.layer] if candidate_layers in ("same", "self", "within") else list(layer_labels.keys())
        for lay in lay_ids:
            lab = layer_labels.get(lay, None)
            if lab is None:
                continue
            tip_points = [tuple(int(v) for v in t["point"]) for t in getattr(base, "tips", {}).values()]                if isinstance(getattr(base, "tips", None), dict) and len(base.tips) > 0 else [base.lt, base.rt]
            for tip in tip_points:
                vals = _sample_labels_in_neighborhood(lab, tip, search_size)
                for cid in np.unique(vals):
                    cid = int(cid)
                    if cid != 0 and cid != base.id:
                        cands.add(cid)
        return sorted(cands)

    def _suppress_large_overlaps() -> int:
        alive = sorted(_alive(), key=lambda z: (-int(z.mask.sum()), z.id))
        kills = 0
        for i in range(len(alive)):
            a = alive[i]
            if not a.exist:
                continue
            for j in range(i + 1, len(alive)):
                b = alive[j]
                if not b.exist:
                    continue
                if not _bbox_intersects(a.bbox, b.bbox, pad=0):
                    continue
                overlap = np.logical_and(a.mask, b.mask)
                if not overlap.any():
                    continue
                oa = float(np.count_nonzero(overlap)) / max(1.0, float(np.count_nonzero(a.mask)))
                ob = float(np.count_nonzero(overlap)) / max(1.0, float(np.count_nonzero(b.mask)))
                if max(oa, ob) <= overlap_kill_thr:
                    continue
                # Keep the component with larger skeleton support, then larger area, then lower id.
                key_a = (a.skel_len, float(np.count_nonzero(a.mask)), -a.id)
                key_b = (b.skel_len, float(np.count_nonzero(b.mask)), -b.id)
                dead = b if key_a >= key_b else a
                dead.exist = False
                kills += 1
                if verbose and kills % print_every_merge == 0:
                    keep = a if dead is b else b
                    print(
                        f"  [kill-overlap] keep(id={keep.id},layer={keep.layer}) "
                        f"kill(id={dead.id},layer={dead.layer}) "
                        f"oa={oa:.3f} ob={ob:.3f}"
                    )
                if dead is a:
                    break
        return kills

    pass_idx = 0
    total_merges = 0
    total_kills = 0

    while pass_idx < max_passes:
        pass_idx += 1
        if verbose and pass_idx % print_every_pass == 0:
            print(
                f"[reconnect-smooth] pass {pass_idx} | alive={len(_alive())} "
                f"| merges={total_merges} kills={total_kills}"
            )

        kills = _suppress_large_overlaps()
        total_kills += kills

        alive = _alive()
        if not alive:
            break

        layer_labels = _build_layer_labels()
        global_label = _component_union_label(alive)
        id2comp = _id2comp()

        proposals: List[Proposal] = []

        # Build one best proposal per component, then resolve globally.
        for base in sorted(alive, key=lambda z: (-z.skel_len, z.layer, z.id)):
            if not base.exist:
                continue
            cand_ids = _candidate_ids(base, layer_labels)
            if not cand_ids:
                continue

            best: Optional[Proposal] = None
            for cid in cand_ids:
                tar = id2comp.get(cid, None)
                if tar is None or not tar.exist or tar.id == base.id:
                    continue
                base_tip_names = list(getattr(base, "tips", {}).keys()) if isinstance(getattr(base, "tips", None), dict) and len(base.tips) > 0 else ["l", "r"]
                tar_tip_names = list(getattr(tar, "tips", {}).keys()) if isinstance(getattr(tar, "tips", None), dict) and len(tar.tips) > 0 else ["l", "r"]
                for bt in base_tip_names:
                    for tt in tar_tip_names:
                        prop = _evaluate_tip_pair(base, tar, bt, tt, global_label, cfg)
                        if prop is None:
                            continue
                        if best is None or prop.score < best.score:
                            best = prop
            if best is not None:
                proposals.append(best)

        if not proposals:
            if kills == 0:
                break
            # if only kills happened, continue another pass
            for c in _alive():
                c.refresh_geom(trace_steps, fit_points)
            continue

        # Deduplicate by unordered pair: keep the best proposal for each pair.
        best_pair: Dict[Tuple[int, int], Proposal] = {}
        for p in proposals:
            key = (min(p.base_id, p.tar_id), max(p.base_id, p.tar_id))
            old = best_pair.get(key, None)
            if old is None or p.score < old.score:
                best_pair[key] = p

        proposal_list = sorted(best_pair.values(), key=lambda p: p.score)

        used_ids: set[int] = set()
        merges_this_pass = 0

        for p in proposal_list:
            if merges_this_pass >= max_merges_per_pass:
                break
            if p.base_id in used_ids or p.tar_id in used_ids:
                continue

            base = id2comp.get(p.base_id, None)
            tar = id2comp.get(p.tar_id, None)
            if base is None or tar is None or (not base.exist) or (not tar.exist):
                continue

            # Keep the more stable trunk.
            keep = base
            kill = tar
            if (tar.skel_len, float(np.count_nonzero(tar.mask)), -tar.id) > (
                base.skel_len, float(np.count_nonzero(base.mask)), -base.id
            ):
                keep, kill = tar, base

            bridge_mask = _polyline_mask(p.bridge_points, keep.mask.shape)
            keep.mask = np.logical_or(keep.mask, kill.mask)
            keep.mask = np.logical_or(keep.mask, bridge_mask)
            kill.exist = False

            keep.refresh_geom(trace_steps, fit_points)

            used_ids.add(base.id)
            used_ids.add(tar.id)
            merges_this_pass += 1
            total_merges += 1

            if verbose and total_merges % print_every_merge == 0:
                m = p.metrics
                print(
                    f"  [merge-smooth] keep(id={keep.id},layer={keep.layer}) "
                    f"+ kill(id={kill.id},layer={kill.layer}) "
                    f"| dist={m['dist']:.2f} fwd=({m['forward_base']:.2f},{m['forward_tar']:.2f}) "
                    f"opp={m['inward_opposition']:.2f} smooth={m['smooth_rms']:.2f} "
                    f"turn={m['max_turn_deg']:.1f}"
                )

        if merges_this_pass == 0 and kills == 0:
            break

    return components
