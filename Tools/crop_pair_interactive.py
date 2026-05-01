from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


DEFAULT_MASK    = Path(r"C:\Repos\filaments_quantification\input\sem_full_00000_1p66\sem_full_00000_1p66_mask255.png")
DEFAULT_OVERLAY = Path(r"C:\Repos\filaments_quantification\input\sem_full_00000_1p66\sem_full_00000_1p66_overlay.png")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactively crop an aligned mask/overlay image pair.")
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    ap.add_argument("--output", type=Path, default=None, help="Output folder; defaults beside the mask.")
    ap.add_argument("--suffix", default="_crop", help="Suffix added before each output extension.")
    ap.add_argument("--max-display", type=int, default=1200, help="Largest display dimension in pixels.")
    return ap.parse_args()


def read_image(path: Path, mode: int):
    img = cv2.imread(str(path), mode)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def select_roi(mask, overlay, max_display: int) -> tuple[int, int, int, int]:
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    mask_show = mask if mask.ndim == 2 else cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
    state = {"roi": None, "accepted": False}
    h, w = overlay.shape[:2]
    fig_w = min(14.0, max(6.0, max_display / 100.0))
    fig_h = max(4.0, fig_w * h / max(1, 2 * w))
    fig, axs = plt.subplots(1, 2, num="Crop pair", figsize=(fig_w, fig_h), constrained_layout=True)
    axs[0].imshow(mask_show, cmap="gray")
    axs[1].imshow(overlay_rgb)
    for ax, title in zip(axs, ("mask", "overlay")):
        ax.set_title(title)
        ax.set_axis_off()
    patches = [Rectangle((0, 0), 0, 0, fill=False, ec="yellow", lw=2) for _ in axs]
    for ax, patch in zip(axs, patches):
        ax.add_patch(patch)
    fig.suptitle("Drag ROI on either image. Enter/Space saves, R resets, C/Esc cancels.")

    def update_rect(x, y, w, h):
        for patch in patches:
            patch.set_xy((x, y))
            patch.set_width(w)
            patch.set_height(h)
            patch.set_visible(w > 0 and h > 0)
        fig.canvas.draw_idle()

    def on_select(start, end):
        if start.xdata is None or end.xdata is None:
            return
        x0, x1 = sorted((start.xdata, end.xdata))
        y0, y1 = sorted((start.ydata, end.ydata))
        state["roi"] = (round(x0), round(y0), round(x1 - x0), round(y1 - y0))
        update_rect(*state["roi"])

    selectors = [
        RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=True)
        for ax in axs
    ]

    def on_key(event):
        if event.key in ("enter", " ", "space"):
            if state["roi"] is not None:
                state["accepted"] = True
                plt.close(fig)
        elif event.key in ("r", "R"):
            state["roi"] = None
            update_rect(0, 0, 0, 0)
        elif event.key in ("escape", "c", "C"):
            state["roi"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Drag ROI on either image; press Enter/Space to save, R reset, C/Esc cancel.")
    plt.show()
    _ = selectors
    if state["roi"] is None or not state["accepted"]:
        raise SystemExit("No crop selected.")
    x, y, w, h = state["roi"]
    if w <= 0 or h <= 0:
        raise SystemExit("No crop selected.")
    return x, y, w, h


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def out_path(src: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{src.stem}{suffix}{src.suffix}"


def main() -> None:
    args = parse_args()
    mask = read_image(args.mask, cv2.IMREAD_UNCHANGED)
    overlay = read_image(args.overlay, cv2.IMREAD_COLOR)

    if mask.shape[:2] != overlay.shape[:2]:
        raise ValueError(f"Shape mismatch: mask {mask.shape[:2]} vs overlay {overlay.shape[:2]}")

    x, y, w, h = clamp_roi(select_roi(mask, overlay, args.max_display), overlay.shape[1], overlay.shape[0])

    out_dir = args.output or (args.mask.parent / "crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_out = out_path(args.mask, out_dir, args.suffix)
    overlay_out = out_path(args.overlay, out_dir, args.suffix)

    cv2.imwrite(str(mask_out), mask[y : y + h, x : x + w])
    cv2.imwrite(str(overlay_out), overlay[y : y + h, x : x + w])

    meta = {
        "mask": str(args.mask),
        "overlay": str(args.overlay),
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "mask_crop": str(mask_out),
        "overlay_crop": str(overlay_out),
    }
    meta_out = out_dir / f"{args.mask.stem}{args.suffix}_roi.json"
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved:\n  {mask_out}\n  {overlay_out}\n  {meta_out}")


if __name__ == "__main__":
    main()
