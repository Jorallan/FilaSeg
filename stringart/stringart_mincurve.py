"""
Minimum-Curvature Filament Tracker
Tracks filament centerlines in a binary mask via skeleton graph + min-curvature continuation.

Usage:
    python filament_tracker.py                    # uses CONFIG below
    python filament_tracker.py input.png out/     # CLI override
    python filament_tracker.py input_folder/ out/ # batch mode

pip install opencv-python numpy matplotlib scikit-image scipy pandas
"""

import argparse, csv, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, label, center_of_mass, distance_transform_edt
from skimage.morphology import skeletonize


# ============================================================
# CONFIG
# ============================================================

# Input can be a single image OR a folder
INPUT_PATH = r"C:\Repos\filaments_quantification\2_stringart\input\case00000_0000"
OUTPUT_FOLDER = r"C:\Repos\filaments_quantification\2_stringart\output\case00000_0000MC"

# Preprocessing
ASSUME_BINARY        = True     # skip thresholding if already a binary mask
THRESH_MODE          = "auto"   # "auto" | "fixed" | "otsu" | "adaptive"
FIXED_THRESHOLD      = 127
INVERT               = False
DO_OPEN, OPEN_K      = False, 3
DO_CLOSE, CLOSE_K    = False, 3
REMOVE_SMALL_CC      = False
MIN_CC_AREA_PX       = 25

# Skeleton
PRUNE_SPURS          = True
PRUNE_ITERS          = 8

# Graph
INSERT_LOOP_NODES    = True     # inject pseudonode for closed-loop components
MIN_EDGE_PX          = 2

# Filament tracing
TANGENT_LEN_PX       = 12
MAX_TURN_DEG         = 70.0
MIN_SCORE            = 0.15
LENGTH_BIAS          = 0.10     # small preference for longer edges
DIRECTION_WEIGHT     = 1.00
MIN_FILAMENT_PX      = 8

# Orientation export
NUM_BINS             = 12
JUNCTION_DILATE      = 2        # px to exclude junction zone from orientation seeds

# Output
SAVE_DEBUG_PREPROC   = True
SAVE_SUMMARY_FIG     = True
SAVE_CSV             = True
SHOW_PLOTS           = False
FIG_DPI              = 250
RNG_SEED             = 0
RECURSIVE            = True
SUPPORTED_EXT        = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

NEI8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Edge:
    eid: int
    u: int
    v: int
    pixels: List[Tuple[int,int]]  # ordered skeleton pixels, excluding node interiors

@dataclass
class Filament:
    fid: int
    edges: List[int]
    pixels: List[Tuple[int,int]]
    length_px: float
    mean_angle_deg: float


# ============================================================
# UTILITIES
# ============================================================

def _inb(y, x, h, w): return 0 <= y < h and 0 <= x < w

def neighbors8(y, x, sk):
    h, w = sk.shape
    return [(y+dy, x+dx) for dy,dx in NEI8 if _inb(y+dy,x+dx,h,w) and sk[y+dy,x+dx]]

def path_length(path):
    if len(path) < 2: return float(len(path))
    return sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) for i in range(1,len(path)))

def circular_mean_180(angles_deg):
    if not angles_deg: return 0.0
    a2 = np.deg2rad(np.array(angles_deg) * 2.0)
    m = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean())
    return float(np.rad2deg(m) / 2.0 % 180.0)

def edge_orientation(edge):
    if len(edge.pixels) < 2: return None
    (y0,x0),(y1,x1) = edge.pixels[0], edge.pixels[-1]
    if abs(x1-x0) < 1e-12 and abs(y1-y0) < 1e-12: return None
    return float(np.degrees(np.arctan2(y1-y0, x1-x0)) % 180.0)

def other_node(edge, nid):
    return edge.v if edge.u == nid else edge.u

def is_binary_like(img):
    s = img[::2,::2] if img.size > 2_000_000 else img
    if np.unique(s).size <= 4: return True
    return (np.mean(s <= 10) + np.mean(s >= 245)) > 0.98


# ============================================================
# PREPROCESSING
# ============================================================

def load(filepath):
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot load: {filepath}")
    raw = img.copy()

    if ASSUME_BINARY or (THRESH_MODE == "auto" and is_binary_like(img)):
        binary = ((img > 0).astype(np.uint8) * 255)
        return raw, (255 - binary if INVERT else binary)

    mode = THRESH_MODE.lower()
    if mode == "fixed":
        _, binary = cv2.threshold(img, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY)
    elif mode in ("otsu", "auto"):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)

    if mode == "auto" and np.mean(binary > 0) > 0.5:
        binary = 255 - binary
    if INVERT: binary = 255 - binary

    if DO_OPEN:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(OPEN_K))
    if DO_CLOSE:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(CLOSE_K))

    if REMOVE_SMALL_CC:
        cc, n = label((binary > 0).astype(np.uint8), structure=np.ones((3,3), np.uint8))
        mask = np.zeros_like(binary)
        for i in range(1, n+1):
            if (cc == i).sum() >= MIN_CC_AREA_PX:
                mask[cc == i] = 255
        binary = mask

    return raw, binary

def _ellipse(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))


# ============================================================
# SKELETON
# ============================================================

_DEG_KERNEL = np.ones((3,3), np.uint8); _DEG_KERNEL[1,1] = 0

def prune_spurs(sk, iters):
    sk = sk.copy().astype(np.uint8)
    for _ in range(iters):
        deg = convolve(sk, _DEG_KERNEL, mode="constant", cval=0)
        ends = (sk == 1) & (deg == 1)
        if not ends.any(): break
        sk[ends] = 0
    return sk


# ============================================================
# GRAPH EXTRACTION
# ============================================================

def build_graph(sk_bool):
    sk = sk_bool.astype(np.uint8)
    deg = convolve(sk, _DEG_KERNEL, mode="constant", cval=0)
    node_pix = (sk == 1) & (deg != 2)

    struct8 = np.ones((3,3), np.uint8)
    node_id_map, _ = label(node_pix.astype(np.uint8), structure=struct8)
    node_id_map = node_id_map.astype(np.int32)

    # inject pseudonodes for closed-loop components (no endpoints/junctions)
    if INSERT_LOOP_NODES:
        sk_cc, n_cc = label(sk, structure=struct8)
        nxt = int(node_id_map.max()) + 1
        for c in range(1, n_cc+1):
            m = sk_cc == c
            if not np.any(m & node_pix):
                ys, xs = np.where(m)
                node_id_map[int(ys[0]), int(xs[0])] = nxt
                nxt += 1
        node_id_map, _ = label((node_id_map > 0).astype(np.uint8), structure=struct8)
        node_id_map = node_id_map.astype(np.int32)

    # build node info dict
    nodes = {}
    for nid in range(1, node_id_map.max()+1):
        m = (node_id_map == nid)
        cy, cx = center_of_mass(m.astype(np.uint8))
        nodes[nid] = {
            "pixels": list(map(tuple, np.column_stack(np.where(m)))),
            "centroid": (float(cy), float(cx)),
        }

    # trace edges
    visited_steps: Set[Tuple] = set()
    edges: Dict[int, Edge] = {}
    eid = 1

    def mark(a, b):
        visited_steps.add((a[0],a[1],b[0],b[1]))
        visited_steps.add((b[0],b[1],a[0],a[1]))

    for nid, nd in nodes.items():
        node_set = set(nd["pixels"])
        # find boundary pixels of this node
        boundary = [(y,x) for (y,x) in nd["pixels"]
                    if any((yy,xx) not in node_set for yy,xx in neighbors8(y,x,sk_bool))]

        for (y0,x0) in boundary:
            for (y1,x1) in neighbors8(y0,x0,sk_bool):
                if node_id_map[y1,x1] == nid: continue
                if (y0,x0,y1,x1) in visited_steps: continue

                prev, cur = (y0,x0), (y1,x1)
                mark(prev, cur)
                path = []

                while True:
                    cy,cx = cur
                    cur_nid = int(node_id_map[cy,cx])
                    if cur_nid > 0:
                        end_nid = cur_nid; break

                    path.append(cur)
                    nbs = [p for p in neighbors8(cy,cx,sk_bool) if p != prev]

                    if not nbs: end_nid = None; break
                    elif len(nbs) == 1: nxt = nbs[0]
                    else:
                        # greedy: prefer most-forward direction
                        py,px = prev
                        v0 = np.array([cy-py,cx-px],np.float32)
                        n0 = float(np.linalg.norm(v0))
                        if n0 < 1e-6: nxt = nbs[0]
                        else:
                            v0 /= n0
                            nxt = max(nbs, key=lambda p: np.dot(v0, np.array([p[0]-cy,p[1]-cx],np.float32)))

                    mark(cur, nxt); prev,cur = cur,nxt

                if end_nid is None or len(path) < MIN_EDGE_PX: continue
                edges[eid] = Edge(eid=eid, u=nid, v=end_nid, pixels=path)
                eid += 1

    return nodes, edges, node_id_map


# ============================================================
# MINIMUM-CURVATURE CONTINUATION
# ============================================================

def _tangent(edge, nid, nodes):
    pts = edge.pixels if edge.u == nid else list(reversed(edge.pixels))
    if len(pts) < 2: return None
    use = pts[:min(TANGENT_LEN_PX, len(pts))]
    cy,cx = nodes[nid]["centroid"]
    ys,xs = np.array([p[0] for p in use],np.float32), np.array([p[1] for p in use],np.float32)
    v = np.array([ys.mean()-cy, xs.mean()-cx], np.float32)
    n = float(np.linalg.norm(v))
    return (v/n) if n > 1e-6 else None

def _score(in_tan, out_tan, length):
    turn = float(np.degrees(np.arccos(np.clip(np.dot(-in_tan, out_tan), -1,1))))
    if turn > MAX_TURN_DEG: return turn, -1e9
    angle_term = 1.0 - turn / max(MAX_TURN_DEG, 1e-6)
    score = DIRECTION_WEIGHT * angle_term + LENGTH_BIAS * math.tanh(length / 20.0)
    return turn, score

def _choose_next(cur_eid, at_node, nodes, edges, inc, visited):
    in_tan = _tangent(edges[cur_eid], at_node, nodes)
    if in_tan is None: return None

    best_eid, best_score = None, -1e9
    for eid in inc[at_node]:
        if eid == cur_eid or eid in visited: continue
        out_tan = _tangent(edges[eid], at_node, nodes)
        if out_tan is None: continue
        _, score = _score(in_tan, out_tan, path_length(edges[eid].pixels))
        if score > best_score:
            best_score, best_eid = score, eid

    return best_eid if best_score >= MIN_SCORE else None

def trace_filaments(nodes, edges):
    inc: Dict[int,List[int]] = {nid:[] for nid in nodes}
    for eid,e in edges.items():
        inc[e.u].append(eid); inc[e.v].append(eid)

    visited: Set[int] = set()
    filaments: List[Filament] = []
    fid = 1

    # prioritise starts at low-degree nodes (endpoints first)
    starts = [(eid, e.u) for eid,e in edges.items() if len(inc[e.u]) <= 2] + \
             [(eid, e.v) for eid,e in edges.items() if len(inc[e.v]) <= 2] + \
             [(eid, e.u) for eid,e in edges.items()]  # fallback for interior/loops

    for eid, snode in starts:
        if eid in visited: continue

        seq, cur_eid, cur_node = [], eid, snode
        while cur_eid not in visited:
            visited.add(cur_eid); seq.append(cur_eid)
            nxt_node = other_node(edges[cur_eid], cur_node)
            nxt_eid = _choose_next(cur_eid, nxt_node, nodes, edges, inc, visited)
            if nxt_eid is None: break
            cur_node, cur_eid = nxt_node, nxt_eid

        pixels, angles, total = [], [], 0.0
        for e in seq:
            pixels.extend(edges[e].pixels)
            total += path_length(edges[e].pixels)
            ang = edge_orientation(edges[e])
            if ang is not None: angles.append(ang)

        if total < MIN_FILAMENT_PX: continue
        filaments.append(Filament(fid=fid, edges=seq, pixels=pixels,
                                  length_px=total, mean_angle_deg=circular_mean_180(angles)))
        fid += 1

    return filaments


# ============================================================
# ORIENTATION LAYERS  (exclusive — each pixel assigned to nearest seed bin)
# ============================================================

def export_orientation_layers(mask_u8, sk_bool, node_id_map, edges, out_dir, base):
    h, w = mask_u8.shape
    bin_step = 180.0 / NUM_BINS

    # dilate junction zone to exclude from seeds
    node_mask = (node_id_map > 0) & sk_bool
    if JUNCTION_DILATE > 0:
        k = _ellipse(2*JUNCTION_DILATE+1)
        node_mask = cv2.dilate(node_mask.astype(np.uint8)*255, k) > 0

    seeds = np.zeros((h,w), np.int32)
    for eid,e in edges.items():
        ang = edge_orientation(e)
        if ang is None: continue
        lab = int(ang // bin_step) % NUM_BINS + 1
        for (y,x) in e.pixels:
            if not node_mask[y,x]: seeds[y,x] = lab

    if not seeds.any(): raise RuntimeError("No orientation seeds — check preprocessing.")

    dt_in = (seeds == 0).astype(np.uint8)
    _, inds = distance_transform_edt(dt_in, return_indices=True)
    assigned = seeds[inds[0], inds[1]]
    final = np.where(mask_u8 > 0, assigned, 0).astype(np.int32)

    layer_dir = Path(out_dir) / "orientation_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_BINS):
        layer = ((final == i+1).astype(np.uint8) * 255)
        cv2.imwrite(str(layer_dir / f"{base}_branch_{i+1}.png"), layer)

    # debug colour viz
    rng = np.random.default_rng(RNG_SEED)
    colors = np.vstack([[[0,0,0]], rng.integers(32,255,size=(NUM_BINS,3),dtype=np.uint8)]).astype(np.uint8)
    viz = colors[np.clip(final, 0, NUM_BINS)].astype(np.uint8)
    cv2.imwrite(str(layer_dir / f"{base}_orientation_rgb.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


# ============================================================
# VISUALIZATION & STATS
# ============================================================

def save_outputs(raw, binary, sk, node_id_map, filaments, edges, out_dir, base):
    h, w = binary.shape
    fil_map = np.zeros((h,w), np.int32)
    for f in filaments:
        for (y,x) in f.pixels: fil_map[y,x] = f.fid

    rng = np.random.default_rng(RNG_SEED)
    colors = np.vstack([[[0,0,0]], rng.integers(50,255,size=(len(filaments),3),dtype=np.uint8)]).astype(np.uint8)
    rgb = colors[np.clip(fil_map, 0, len(filaments))].copy().astype(np.uint8)
    rgb[(node_id_map > 0) & sk] = [255,255,255]

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir/"color_coded_filaments.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if SAVE_SUMMARY_FIG:
        fig, axes = plt.subplots(1,5, figsize=(20,4))
        for ax,(im,title,cmap) in zip(axes, [
            (raw,                         "Raw",        "gray"),
            (binary,                      "Binary",     "gray"),
            (sk.astype(np.uint8)*255,     "Skeleton",   "gray"),
            ((node_id_map>0).astype(np.uint8)*255, "Nodes", "gray"),
            (rgb,                         "Filaments",  None),
        ]):
            ax.imshow(im, cmap=cmap); ax.set_title(title); ax.axis("off")
        plt.tight_layout()
        fig.savefig(str(out_dir/"summary_plot.png"), dpi=FIG_DPI, bbox_inches="tight")
        if SHOW_PLOTS: plt.show()
        else: plt.close(fig)

    if SAVE_CSV:
        rows = [{"fid":f.fid,"edges":len(f.edges),"pixels":len(f.pixels),
                 "length_px":f.length_px,"mean_angle_deg":f.mean_angle_deg} for f in filaments]
        pd.DataFrame(rows).to_csv(str(out_dir/f"{base}_stats.csv"), index=False)


# ============================================================
# PIPELINE
# ============================================================

def process(filepath, output_root):
    filepath = Path(filepath)
    base = filepath.stem
    out_dir = Path(output_root) / base
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, binary = load(filepath)

    if SAVE_DEBUG_PREPROC:
        dbg = out_dir / "debug"
        dbg.mkdir(exist_ok=True)
        cv2.imwrite(str(dbg/"raw.png"), raw)
        cv2.imwrite(str(dbg/"binary.png"), binary)

    sk = skeletonize(binary > 0)
    if PRUNE_SPURS:
        sk = prune_spurs(sk.astype(np.uint8), PRUNE_ITERS).astype(bool)

    nodes, edges, node_id_map = build_graph(sk)
    if not edges: raise RuntimeError("No edges found — check your mask/preprocessing settings.")

    filaments = trace_filaments(nodes, edges)

    save_outputs(raw, binary, sk, node_id_map, filaments, edges, out_dir, base)
    export_orientation_layers(binary, sk, node_id_map, edges, out_dir, base)

    print(f"  [{base}] nodes={len(nodes)} edges={len(edges)} filaments={len(filaments)} -> {out_dir}")


def find_inputs(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED_EXT else []
    fn = p.rglob if RECURSIVE else p.glob
    return sorted(f for f in fn("*") if f.suffix.lower() in SUPPORTED_EXT)


def main():
    parser = argparse.ArgumentParser(description="Minimum-curvature filament tracker")
    parser.add_argument("input",  nargs="?", default=INPUT_PATH)
    parser.add_argument("output", nargs="?", default=OUTPUT_FOLDER)
    args = parser.parse_args()

    files = find_inputs(args.input)
    if not files: raise RuntimeError(f"No supported images found at: {args.input}")
    print(f"Found {len(files)} image(s).")

    failures = []
    for fp in files:
        try:
            process(fp, args.output)
        except Exception as e:
            failures.append((fp, e)); print(f"  ERROR [{fp.name}]: {e}")

    print(f"\nDone. {len(files)-len(failures)}/{len(files)} succeeded.")
    for fp,e in failures: print(f"  FAILED: {fp}\n    {e}")


if __name__ == "__main__":
    main()
