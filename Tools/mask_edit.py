#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive binary mask editor (OpenCV GUI) with overlay, brush, polyline, zoom/pan.

Dependencies:
    pip install opencv-python numpy

Run:
    python mask_edit.py

Main controls
-------------
Brush mode (default):
- Left drag      : add mask
- Ctrl + Left drag OR Right drag : erase mask
- Wheel / [ ]    : brush radius
- L              : toggle polyline mode

Polyline mode:
- Left click     : add point
- Enter          : apply polyline to mask (smoothed)
- Backspace      : remove last point
- Esc            : cancel current polyline
- Wheel / - +    : polyline width

Zoom / pan:
- Ctrl + Wheel   : zoom at cursor
- Middle drag    : pan
- Shift + Left drag : pan (optional)

General:
- S              : save mask (0/255)
- Q              : quit (prompts if unsaved)
- 0 or F         : reset zoom/pan
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_io import REPO_ROOT, imread_any, imwrite_any, to_gray_u8 as common_to_gray_u8


# =============================================================================
# USER-EDITABLE SETTINGS
# =============================================================================
GRAY_PATH = str(REPO_ROOT / "Datasets" / "processed" / "unet" / "sem_nnUNet_pred" / "nnunet_infer_2026-04-30_15-45-39" / "nnunet_input" / "case00001_0000.png")
MASK_PATH = str(REPO_ROOT / "Datasets" / "processed" / "unet" / "sem_nnUNet_pred" / "nnunet_infer_2026-04-30_15-45-39" / "preds_vis_mask_255" / "sem_full_00001_mask255.png")
OUTPUT_MASK_PATH = str(REPO_ROOT / "Datasets" / "processed" / "unet" / "sem_nnUNet_pred" / "nnunet_infer_2026-04-30_15-45-39" / "preds_vis_mask_255" / "case00001_0000.png")

OVERLAY_ALPHA = 0.45
MASK_COLOR_BGR = (0, 0, 255)
POLYLINE_PREVIEW_COLOR_BGR = (0, 255, 255)
BRUSH_CURSOR_COLOR_BGR = (0, 255, 0)

DEFAULT_BRUSH_RADIUS = 7
MIN_BRUSH_RADIUS = 1
MAX_BRUSH_RADIUS = 200

DEFAULT_POLYLINE_WIDTH = 1
MIN_POLYLINE_WIDTH = 1
MAX_POLYLINE_WIDTH = 25

# Polyline smoothing (applied before rasterization)
POLYLINE_SMOOTH_ENABLED = True
POLYLINE_SMOOTH_SAMPLES_PER_SEG = 8   # more = smoother preview/application
POLYLINE_SMOOTH_GAUSS_K = 5           # odd integer >=3; set 1/0 to disable smoothing
POLYLINE_SMOOTH_GAUSS_SIGMA = 1.0

DEFAULT_ZOOM = 1.0
MIN_ZOOM = 1.0
MAX_ZOOM = 32.0
ZOOM_STEP_PER_WHEEL = 1.15
PAN_KEY_WITH_LEFT_DRAG = True

WINDOW_NAME = "Mask Editor"
SHOW_HELP_TEXT = True
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1


# =============================================================================
# GLOBAL STATE
# =============================================================================
gray_u8: Optional[np.ndarray] = None
mask_bool: Optional[np.ndarray] = None
dirty = False

editor_mode = "brush"  # "brush" or "polyline"
brush_radius = DEFAULT_BRUSH_RADIUS
polyline_width = DEFAULT_POLYLINE_WIDTH

# mouse/drawing state
mouse_pos_view = (0, 0)          # window coords
mouse_pos_img = (0, 0)           # image coords
is_drawing_left = False
is_drawing_right = False
last_pt_left = None
last_pt_right = None

# pan state
is_panning = False
pan_start_view = None
pan_start_offset = None

# polyline points in image coordinates
poly_points: List[Tuple[int, int]] = []

# viewport transform
zoom = DEFAULT_ZOOM
view_offset_x = 0.0
view_offset_y = 0.0


# =============================================================================
# SMALL HELPERS
# =============================================================================
def fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def clip_pt(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def ctrl_pressed(flags: int) -> bool:
    return (flags & getattr(cv2, "EVENT_FLAG_CTRLKEY", 8)) != 0


def shift_pressed(flags: int) -> bool:
    return (flags & getattr(cv2, "EVENT_FLAG_SHIFTKEY", 16)) != 0


def wheel_delta(flags: int) -> int:
    if hasattr(cv2, "getMouseWheelDelta"):
        return int(cv2.getMouseWheelDelta(flags))
    d = (int(flags) >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d


def load_gray_u8(path: str) -> np.ndarray:
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        fail(f"Cannot read grayscale image: {path}")
    gray = common_to_gray_u8(img)
    if gray is None:
        fail(f"Unsupported grayscale image: shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}")
    return gray


def load_mask_bool(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        fail(f"Cannot read mask image: {path}")
    img = common_to_gray_u8(img)
    if img is None:
        fail(f"Unsupported mask image: shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}")
    if img.shape != shape_hw:
        fail(f"Mask size {img.shape} != grayscale size {shape_hw}")
    return (img != 0)


def save_mask(mask: np.ndarray, out_path: str):
    if not imwrite_any(out_path, (mask.astype(np.uint8) * 255)):
        fail(f"Failed to save: {out_path}")
    print(f"[Saved] {out_path}")


def mark_dirty():
    global dirty
    dirty = True


def save_current():
    global dirty
    save_mask(mask_bool, OUTPUT_MASK_PATH)
    dirty = False


def prompt_save_if_dirty() -> bool:
    if not dirty:
        return True
    print("\nUnsaved changes detected.")
    print("Save before quit? [y]es / [n]o / [c]ancel : ", end="", flush=True)
    try:
        ans = input().strip().lower()
    except EOFError:
        ans = "c"
    if ans in ("y", "yes"):
        save_current()
        return True
    if ans in ("n", "no"):
        return True
    print("Quit canceled.")
    return False


# =============================================================================
# VIEW TRANSFORM
# =============================================================================
def reset_view():
    global zoom, view_offset_x, view_offset_y
    zoom = float(DEFAULT_ZOOM)
    view_offset_x = 0.0
    view_offset_y = 0.0


def clamp_view():
    global view_offset_x, view_offset_y
    h, w = gray_u8.shape
    dw, dh = w * zoom, h * zoom
    view_offset_x = float(np.clip(view_offset_x, -dw + 1, w - 1))
    view_offset_y = float(np.clip(view_offset_y, -dh + 1, h - 1))


def view_to_img(vx: int, vy: int) -> Tuple[int, int]:
    h, w = gray_u8.shape
    x = int(np.floor((vx - view_offset_x) / zoom + 0.5))
    y = int(np.floor((vy - view_offset_y) / zoom + 0.5))
    return clip_pt(x, y, w, h)


def zoom_at(vx: int, vy: int, factor: float):
    global zoom, view_offset_x, view_offset_y
    old = zoom
    new = float(np.clip(old * factor, MIN_ZOOM, MAX_ZOOM))
    if abs(new - old) < 1e-12:
        return
    ix = (vx - view_offset_x) / old
    iy = (vy - view_offset_y) / old
    zoom = new
    view_offset_x = vx - ix * zoom
    view_offset_y = vy - iy * zoom
    clamp_view()


# =============================================================================
# MASK DRAW OPS
# =============================================================================
def draw_line_bool(mask: np.ndarray, p0, p1, value=True, width=1):
    h, w = mask.shape
    x0, y0 = clip_pt(int(p0[0]), int(p0[1]), w, h)
    x1, y1 = clip_pt(int(p1[0]), int(p1[1]), w, h)
    tmp = np.zeros((h, w), np.uint8)
    cv2.line(tmp, (x0, y0), (x1, y1), 1, int(max(1, width)), cv2.LINE_8)
    mask[tmp > 0] = bool(value)


def draw_brush_stroke(mask: np.ndarray, p0, p1, radius: int, value: bool):
    h, w = mask.shape
    x0, y0 = clip_pt(int(p0[0]), int(p0[1]), w, h)
    x1, y1 = clip_pt(int(p1[0]), int(p1[1]), w, h)
    tmp = np.zeros((h, w), np.uint8)
    t = int(max(1, 2 * radius + 1))
    cv2.line(tmp, (x0, y0), (x1, y1), 1, t, cv2.LINE_8)
    cv2.circle(tmp, (x0, y0), int(radius), 1, -1, cv2.LINE_8)
    cv2.circle(tmp, (x1, y1), int(radius), 1, -1, cv2.LINE_8)
    mask[tmp > 0] = bool(value)


# =============================================================================
# POLYLINE SMOOTHING
# =============================================================================
def _resample_polyline(points: np.ndarray, samples_per_seg: int) -> np.ndarray:
    """Linear densification (no geometry change), used before smoothing."""
    if len(points) < 2:
        return points
    out = [points[0]]
    n = max(1, int(samples_per_seg))
    for i in range(1, len(points)):
        a = points[i - 1].astype(np.float32)
        b = points[i].astype(np.float32)
        for t in range(1, n + 1):
            out.append(a + (b - a) * (t / n))
    return np.asarray(out, dtype=np.float32)


def _smooth_polyline_points(points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Returns float Nx2 smoothed points (image coordinates).
    Endpoints are preserved; smoothing only affects interior points.
    """
    if len(points) < 2:
        return np.asarray(points, dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    dense = _resample_polyline(pts, POLYLINE_SMOOTH_SAMPLES_PER_SEG)

    if not POLYLINE_SMOOTH_ENABLED or POLYLINE_SMOOTH_GAUSS_K < 3 or len(dense) < 3:
        return dense

    k = int(POLYLINE_SMOOTH_GAUSS_K)
    if k % 2 == 0:
        k += 1
    sigma = float(POLYLINE_SMOOTH_GAUSS_SIGMA)

    # Smooth x(t), y(t) separately with endpoint padding to reduce edge distortion
    x = dense[:, 0].reshape(-1, 1)
    y = dense[:, 1].reshape(-1, 1)
    x_s = cv2.GaussianBlur(x, (k, 1), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE).reshape(-1)
    y_s = cv2.GaussianBlur(y, (k, 1), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE).reshape(-1)

    sm = np.stack([x_s, y_s], axis=1)
    sm[0] = dense[0]
    sm[-1] = dense[-1]
    return sm


def rasterize_smoothed_polyline_into_mask(points: List[Tuple[int, int]], width: int):
    """Smooth in float coords, then rasterize segment-by-segment into mask."""
    global mask_bool
    if len(points) < 2:
        print("[Info] Polyline needs at least 2 points.")
        return

    sm = _smooth_polyline_points(points)
    h, w = mask_bool.shape
    # Round and clip to image pixels
    pix = np.rint(sm).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)

    # Draw segments
    for a, b in zip(pix[:-1], pix[1:]):
        if a[0] == b[0] and a[1] == b[1]:
            continue
        draw_line_bool(mask_bool, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), True, width)

    mark_dirty()


# =============================================================================
# DISPLAY COMPOSITION
# =============================================================================
def compose_display() -> np.ndarray:
    # base grayscale -> BGR
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    out = base.copy()

    # mask overlay in image coordinates
    if np.any(mask_bool):
        m = mask_bool
        color = np.zeros_like(out)
        color[:, :] = MASK_COLOR_BGR
        out[m] = ((1.0 - OVERLAY_ALPHA) * out[m].astype(np.float32) +
                  OVERLAY_ALPHA * color[m].astype(np.float32)).astype(np.uint8)

    # polyline preview (smoothed preview)
    if editor_mode == "polyline" and len(poly_points) >= 1:
        if len(poly_points) >= 2:
            sm = _smooth_polyline_points(poly_points)
            p = np.rint(sm).astype(np.int32)
            for i in range(1, len(p)):
                cv2.line(out, tuple(p[i - 1]), tuple(p[i]), POLYLINE_PREVIEW_COLOR_BGR,
                         int(max(1, polyline_width)), cv2.LINE_8)
        # draw vertices and rubber-band
        for pt in poly_points:
            cv2.circle(out, pt, 2, POLYLINE_PREVIEW_COLOR_BGR, -1, cv2.LINE_8)

        # rubber-band from last point to current mouse (raw preview)
        cv2.line(out, poly_points[-1], mouse_pos_img, POLYLINE_PREVIEW_COLOR_BGR,
                 int(max(1, polyline_width)), cv2.LINE_8)

    # brush cursor
    if editor_mode == "brush":
        cv2.circle(out, mouse_pos_img, int(brush_radius), BRUSH_CURSOR_COLOR_BGR, 1, cv2.LINE_8)

    # apply zoom/pan transform for display
    if abs(zoom - 1.0) > 1e-12 or abs(view_offset_x) > 1e-12 or abs(view_offset_y) > 1e-12:
        h, w = out.shape[:2]
        M = np.array([[zoom, 0.0, view_offset_x], [0.0, zoom, view_offset_y]], dtype=np.float32)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25))

    # HUD (constant screen size)
    if SHOW_HELP_TEXT:
        lines = [
            f"Mode:{editor_mode.upper()} | Brush:{brush_radius}px | PolyW:{polyline_width}px | Zoom:{zoom:.2f}x | Dirty:{'YES' if dirty else 'NO'}",
            "Brush: LMB add | Ctrl+LMB or RMB erase | Wheel/[ ] brush size | Ctrl+Wheel zoom",
            "Polyline (L): click pts, Enter apply (smoothed), Backspace undo, Esc cancel | Wheel or -/+ width",
            "Pan: middle-drag" + (" or Shift+LMB drag" if PAN_KEY_WITH_LEFT_DRAG else "") + " | 0/F reset view | S save | Q quit",
        ]
        y = 18
        for line in lines:
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS + 2, cv2.LINE_AA)
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)
            y += 20

    if is_panning:
        cv2.putText(out, "PANNING", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, "PANNING", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 1, cv2.LINE_AA)

    return out


# =============================================================================
# INPUT HANDLERS
# =============================================================================
def reset_draw_flags():
    global is_drawing_left, is_drawing_right, last_pt_left, last_pt_right
    is_drawing_left = is_drawing_right = False
    last_pt_left = last_pt_right = None


def on_mouse(event, x, y, flags, param):
    global mouse_pos_view, mouse_pos_img
    global is_drawing_left, is_drawing_right, last_pt_left, last_pt_right
    global is_panning, pan_start_view, pan_start_offset
    global view_offset_x, view_offset_y
    global brush_radius, polyline_width, poly_points

    mouse_pos_view = (int(x), int(y))
    mouse_pos_img = view_to_img(int(x), int(y))

    ctrl = ctrl_pressed(flags)
    shift = shift_pressed(flags)

    # Wheel: Ctrl+wheel zoom, otherwise mode-dependent size
    if event == cv2.EVENT_MOUSEWHEEL:
        d = wheel_delta(flags)
        if d == 0:
            return
        if ctrl:
            zoom_at(x, y, ZOOM_STEP_PER_WHEEL if d > 0 else 1.0 / ZOOM_STEP_PER_WHEEL)
        else:
            if editor_mode == "brush":
                brush_radius = int(np.clip(brush_radius + (1 if d > 0 else -1), MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS))
            else:
                polyline_width = int(np.clip(polyline_width + (1 if d > 0 else -1), MIN_POLYLINE_WIDTH, MAX_POLYLINE_WIDTH))
        return

    # Start pan
    if event == cv2.EVENT_MBUTTONDOWN or (PAN_KEY_WITH_LEFT_DRAG and event == cv2.EVENT_LBUTTONDOWN and shift):
        is_panning = True
        pan_start_view = (int(x), int(y))
        pan_start_offset = (float(view_offset_x), float(view_offset_y))
        reset_draw_flags()
        return

    # Pan move
    if event == cv2.EVENT_MOUSEMOVE and is_panning:
        if pan_start_view is not None and pan_start_offset is not None:
            dx = int(x) - pan_start_view[0]
            dy = int(y) - pan_start_view[1]
            view_offset_x = pan_start_offset[0] + dx
            view_offset_y = pan_start_offset[1] + dy
            clamp_view()
        return

    # End pan
    if event == cv2.EVENT_MBUTTONUP or (PAN_KEY_WITH_LEFT_DRAG and event == cv2.EVENT_LBUTTONUP and is_panning):
        is_panning = False
        pan_start_view = None
        pan_start_offset = None
        return

    # Polyline mode
    if editor_mode == "polyline":
        if event == cv2.EVENT_LBUTTONDOWN and not ctrl:  # ctrl reserved for zoom/erase semantics elsewhere
            poly_points.append(mouse_pos_img)
        return

    # Brush mode: LMB add unless Ctrl held => erase; RMB always erase
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing_left = True
        last_pt_left = mouse_pos_img
        erase = ctrl  # requested behavior
        draw_brush_stroke(mask_bool, mouse_pos_img, mouse_pos_img, brush_radius, value=(not erase))
        mark_dirty()
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        is_drawing_right = True
        last_pt_right = mouse_pos_img
        draw_brush_stroke(mask_bool, mouse_pos_img, mouse_pos_img, brush_radius, value=False)
        mark_dirty()
        return

    if event == cv2.EVENT_MOUSEMOVE:
        if is_drawing_left and last_pt_left is not None:
            erase = ctrl
            draw_brush_stroke(mask_bool, last_pt_left, mouse_pos_img, brush_radius, value=(not erase))
            last_pt_left = mouse_pos_img
            mark_dirty()
        if is_drawing_right and last_pt_right is not None:
            draw_brush_stroke(mask_bool, last_pt_right, mouse_pos_img, brush_radius, value=False)
            last_pt_right = mouse_pos_img
            mark_dirty()
        return

    if event == cv2.EVENT_LBUTTONUP:
        is_drawing_left = False
        last_pt_left = None
        return

    if event == cv2.EVENT_RBUTTONUP:
        is_drawing_right = False
        last_pt_right = None
        return


def handle_key(key: int) -> str:
    global editor_mode, poly_points, brush_radius, polyline_width

    if key < 0:
        return "continue"
    k = key & 0xFF

    if k in (ord('s'), ord('S')):
        save_current()
        return "continue"

    if k in (ord('q'), ord('Q')):
        return "quit" if prompt_save_if_dirty() else "cancel"

    if k in (ord('l'), ord('L')):
        editor_mode = "polyline" if editor_mode == "brush" else "brush"
        reset_draw_flags()
        print(f"[Mode] {editor_mode}")
        return "continue"

    if k == ord('['):
        brush_radius = max(MIN_BRUSH_RADIUS, brush_radius - 1)
        return "continue"
    if k == ord(']'):
        brush_radius = min(MAX_BRUSH_RADIUS, brush_radius + 1)
        return "continue"

    if k in (ord('-'), ord('_')):
        polyline_width = max(MIN_POLYLINE_WIDTH, polyline_width - 1)
        return "continue"
    if k in (ord('='), ord('+')):
        polyline_width = min(MAX_POLYLINE_WIDTH, polyline_width + 1)
        return "continue"

    if k in (ord('0'), ord('f'), ord('F')):
        reset_view()
        print("[View] reset")
        return "continue"

    # Backspace
    if key in (8, 127):
        if editor_mode == "polyline" and poly_points:
            poly_points.pop()
        return "continue"

    # Esc
    if key == 27:
        if editor_mode == "polyline" and poly_points:
            poly_points = []
            print("[Polyline] canceled")
        return "continue"

    # Enter
    if key in (10, 13):
        if editor_mode == "polyline":
            rasterize_smoothed_polyline_into_mask(poly_points, polyline_width)
            if len(poly_points) >= 2:
                print(f"[Polyline] applied ({len(poly_points)} pts, width={polyline_width}px, smooth={'on' if POLYLINE_SMOOTH_ENABLED else 'off'})")
            poly_points = []
        return "continue"

    return "continue"


# =============================================================================
# MAIN
# =============================================================================
def main():
    global gray_u8, mask_bool

    gray_u8 = load_gray_u8(GRAY_PATH)
    mask_bool = load_mask_bool(MASK_PATH, gray_u8.shape)
    reset_view()

    print(f"[Loaded] Grayscale: {GRAY_PATH}  shape={gray_u8.shape} dtype={gray_u8.dtype}")
    print(f"[Loaded] Mask     : {MASK_PATH}  shape={mask_bool.shape} dtype=bool")
    print(f"[Output] {OUTPUT_MASK_PATH}\n")
    print("Brush: LMB add | Ctrl+LMB or RMB erase | Wheel/[ ] brush radius")
    print("Polyline (L): click pts, Enter apply (smoothed), Backspace undo, Esc cancel | Wheel or -/+ width")
    print("Zoom/Pan: Ctrl+Wheel zoom at cursor | Middle-drag pan" + (" | Shift+LMB pan" if PAN_KEY_WITH_LEFT_DRAG else ""))
    print("General: S save | Q quit | 0/F reset view\n")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        raise RuntimeError(
            "OpenCV GUI unavailable. Install 'opencv-python' (GUI build), not 'opencv-python-headless'."
        ) from e

    h, w = gray_u8.shape
    cv2.resizeWindow(WINDOW_NAME, min(1400, w), min(900, h))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        try:
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1

        if visible < 1:
            if prompt_save_if_dirty():
                break
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, min(1400, w), min(900, h))
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        cv2.imshow(WINDOW_NAME, compose_display())
        action = handle_key(cv2.waitKeyEx(16))
        if action == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
