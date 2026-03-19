from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from skimage import io as skio
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries

ALLOWED_EXTS = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

DEFAULT_INPUT_DIR = r"C:\Repos\filaments_quantification\reconnect\analysis\cnt_orient_0003_merge\v5"
DEFAULT_BASE = None  # e.g. "cnt_orient_0001_merge"

def read_gray_any(path: Path) -> np.ndarray:
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0 if img.max() <= 255 else max(1.0, float(img.max()))
        return np.clip(img, 0.0, 1.0)

    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0 if img.max() <= 255 else max(1.0, float(img.max()))
    return np.clip(img, 0.0, 1.0)


def to_rgb01(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return gray2rgb(np.clip(img, 0.0, 1.0))
    if img.ndim == 3 and img.shape[2] >= 3:
        return np.clip(img[..., :3], 0.0, 1.0)
    raise ValueError("Unsupported image shape for RGB conversion.")


def _pick_existing_by_preference(folder: Path, stem: str) -> Path | None:
    for ext in ALLOWED_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_basenames(folder: Path) -> list[str]:
    bases = set()

    for p in folder.glob("*_reconnect_labels.tif"):
        bases.add(p.stem[: -len("_reconnect_labels")])

    for p in folder.glob("*_reconnect_labels_dilated.tif"):
        bases.add(p.stem[: -len("_reconnect_labels_dilated")])

    for p in folder.glob("*_branches_merge.*"):
        name = p.stem
        if name.endswith("_branches_merge"):
            bases.add(name[: -len("_branches_merge")])
        else:
            bases.add(name.split("_branches_merge")[0])

    rx = re.compile(r"^(.*)_branch_(\d+)$")
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTS:
            continue
        m = rx.match(p.stem)
        if m:
            bases.add(m.group(1).rstrip("_"))

    return sorted(bases)


def find_best_base(folder: Path, pattern: str | None) -> str | None:
    bases = find_basenames(folder)
    if not bases:
        return None
    if pattern is None:
        return bases[0]
    for b in bases:
        if pattern in b:
            return b
    return None


def load_background(folder: Path, base: str) -> np.ndarray:
    candidates = [
        folder / f"{base}_reconnect_overlay.png",
        folder / f"{base}_original.png",
        folder / f"{base}_original.tif",
        folder / f"{base}_original.tiff",
        folder / f"{base}_branches_merge.png",
        folder / f"{base}_branches_merge.tif",
        folder / f"{base}_branches_merge.tiff",
    ]
    for p in candidates:
        if p.exists():
            return to_rgb01(read_gray_any(p))

    acc = None
    rx = re.compile(rf"^{re.escape(base)}_branch_(\d+)$")
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTS:
            continue
        if not rx.match(p.stem):
            continue
        img = read_gray_any(p)
        if img.ndim == 3:
            img = img[..., 0]
        acc = img if acc is None else np.maximum(acc, img)

    if acc is None:
        raise FileNotFoundError(f"Could not find any background candidate for base '{base}'.")
    return to_rgb01(acc)


def load_labels(folder: Path, base: str, use_dilated: bool = False) -> np.ndarray:
    names = [
        f"{base}_reconnect_labels_dilated.tif" if use_dilated else f"{base}_reconnect_labels.tif",
        f"{base}_reconnect_labels.tif" if use_dilated else f"{base}_reconnect_labels_dilated.tif",
    ]
    for name in names:
        p = folder / name
        if p.exists():
            lbl = skio.imread(str(p))
            if lbl.ndim != 2:
                raise ValueError(f"Label image must be 2D, got shape {lbl.shape}.")
            return lbl.astype(np.int32)
    raise FileNotFoundError(f"Could not find reconnect label tif for base '{base}'.")


def brighten_selection(
    base_rgb: np.ndarray,
    lbl: np.ndarray,
    selected_id: int | None,
    *,
    dim_bg: float = 0.55,
    brighten_gain: float = 1.60,
    outline_on: bool = True,
) -> np.ndarray:
    out = np.clip(base_rgb.copy(), 0.0, 1.0)

    if selected_id is None or selected_id <= 0:
        return out

    mask = (lbl == selected_id)
    if not np.any(mask):
        return out

    out *= dim_bg

    # brighten selected instance
    sel = out[mask]
    sel = np.clip(sel * brighten_gain + 0.15, 0.0, 1.0)
    out[mask] = sel

    if outline_on:
        bd = find_boundaries(mask, mode="outer")
        out[bd] = np.array([1.0, 0.1, 0.1], dtype=np.float32)

    return out


def compute_instance_table(lbl: np.ndarray) -> dict[int, dict]:
    ids = np.unique(lbl)
    ids = ids[ids > 0]

    table: dict[int, dict] = {}
    for cid in ids:
        pts = np.argwhere(lbl == cid)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        ctr = pts.mean(axis=0)
        table[int(cid)] = {
            "area": int(pts.shape[0]),
            "bbox": (int(r0), int(r1), int(c0), int(c1)),
            "center_rc": (float(ctr[0]), float(ctr[1])),
        }
    return table


def save_current_view(path: Path, img01: np.ndarray):
    arr = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    skio.imsave(str(path), arr, check_contrast=False)


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive CNT reconnect label viewer")
    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing reconnect label outputs and background images.",
    )
    ap.add_argument(
        "--base",
        type=str,
        default=DEFAULT_BASE,
        help="Base name to view, e.g. 'cnt_orient_0001_merge'. If omitted, first detected base is used.",
    )
    ap.add_argument(
        "--use_dilated",
        action="store_true",
        help="Prefer *_reconnect_labels_dilated.tif instead of raw labels.",
    )
    ap.add_argument(
        "--dim_bg",
        type=float,
        default=0.55,
        help="Background dim factor when something is selected.",
    )
    ap.add_argument(
        "--brighten_gain",
        type=float,
        default=1.60,
        help="Brightness multiplier for selected instance.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    folder = Path(args.input).resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    base = find_best_base(folder, args.base)
    if args.base is not None and base is None:
        raise FileNotFoundError(f"Could not find a matching base for pattern: {args.base}")
    if base is None:
        raise FileNotFoundError(f"No valid reconnect outputs found in: {folder}")

    bg = load_background(folder, base)
    lbl = load_labels(folder, base, use_dilated=args.use_dilated)

    if bg.shape[:2] != lbl.shape:
        raise ValueError(
            f"Background shape {bg.shape[:2]} does not match label shape {lbl.shape}."
        )

    instance_table = compute_instance_table(lbl)
    ids_sorted = sorted(instance_table.keys())

    selected_id: int | None = None
    hover_id: int | None = None
    locked = False
    show_outline = True
    show_info = True

    current_dim_bg = float(args.dim_bg)
    current_brighten_gain = float(args.brighten_gain)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(12, 1)
    ax = fig.add_subplot(gs[:10, 0])
    ax_dim = fig.add_subplot(gs[10, 0])
    ax_gain = fig.add_subplot(gs[11, 0])

    plt.subplots_adjust(hspace=0.45)

    shown = brighten_selection(
        bg,
        lbl,
        None,
        dim_bg=current_dim_bg,
        brighten_gain=current_brighten_gain,
        outline_on=show_outline,
    )
    im = ax.imshow(shown, interpolation="nearest")
    ax.set_title(f"{base}  |  hover: preview, left click: lock, right click: clear")
    ax.set_axis_off()

    info_text = ax.text(
        0.01,
        0.01,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.6, pad=4),
    )

    slider_dim = Slider(ax_dim, "dim_bg", 0.10, 1.00, valinit=current_dim_bg, valstep=0.01)
    slider_gain = Slider(ax_gain, "brighten_gain", 1.00, 3.00, valinit=current_brighten_gain, valstep=0.01)

    def get_current_id() -> int | None:
        return selected_id if locked else hover_id

    def update_text(cid: int | None):
        if not show_info or cid is None or cid not in instance_table:
            info_text.set_text("")
            return

        meta = instance_table[cid]
        r0, r1, c0, c1 = meta["bbox"]
        cr, cc = meta["center_rc"]
        text = (
            f"id: {cid}\n"
            f"area: {meta['area']} px\n"
            f"bbox: r[{r0}:{r1}] c[{c0}:{c1}]\n"
            f"center: ({cr:.1f}, {cc:.1f})"
        )
        info_text.set_text(text)

    def redraw():
        cid = get_current_id()
        img = brighten_selection(
            bg,
            lbl,
            cid,
            dim_bg=current_dim_bg,
            brighten_gain=current_brighten_gain,
            outline_on=show_outline,
        )
        im.set_data(img)
        update_text(cid)
        fig.canvas.draw_idle()

    def pick_id_from_event(event) -> int | None:
        if event.inaxes != ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if r < 0 or r >= lbl.shape[0] or c < 0 or c >= lbl.shape[1]:
            return None
        cid = int(lbl[r, c])
        return cid if cid > 0 else None

    def on_move(event):
        nonlocal hover_id
        if locked:
            return
        cid = pick_id_from_event(event)
        if cid != hover_id:
            hover_id = cid
            redraw()

    def on_click(event):
        nonlocal locked, selected_id, hover_id
        if event.inaxes != ax:
            return

        if event.button == 1:
            cid = pick_id_from_event(event)
            if cid is None:
                locked = False
                selected_id = None
                hover_id = None
            else:
                locked = True
                selected_id = cid
                hover_id = cid
            redraw()

        elif event.button == 3:
            locked = False
            selected_id = None
            hover_id = None
            redraw()

    def step_selection(direction: int):
        nonlocal locked, selected_id, hover_id
        if not ids_sorted:
            return
        cid = get_current_id()
        if cid is None or cid not in instance_table:
            cid = ids_sorted[0]
        else:
            idx = ids_sorted.index(cid)
            cid = ids_sorted[(idx + direction) % len(ids_sorted)]
        locked = True
        selected_id = cid
        hover_id = cid
        redraw()

    def on_key(event):
        nonlocal locked, selected_id, hover_id, show_outline, show_info

        if event.key in ("escape", "c"):
            locked = False
            selected_id = None
            hover_id = None
            redraw()

        elif event.key == "n":
            step_selection(+1)

        elif event.key == "p":
            step_selection(-1)

        elif event.key == "o":
            show_outline = not show_outline
            redraw()

        elif event.key == "i":
            show_info = not show_info
            redraw()

        elif event.key == "s":
            cid = get_current_id()
            out = brighten_selection(
                bg,
                lbl,
                cid,
                dim_bg=current_dim_bg,
                brighten_gain=current_brighten_gain,
                outline_on=show_outline,
            )
            suffix = f"_id{cid}" if cid is not None else "_full"
            save_path = folder / f"{base}_interactive{suffix}.png"
            save_current_view(save_path, out)
            print(f"[interactive] saved -> {save_path}")

    def on_slider(_):
        nonlocal current_dim_bg, current_brighten_gain
        current_dim_bg = float(slider_dim.val)
        current_brighten_gain = float(slider_gain.val)
        redraw()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    slider_dim.on_changed(on_slider)
    slider_gain.on_changed(on_slider)

    redraw()
    print(f"[interactive] base: {base}")
    print(f"[interactive] instances: {len(ids_sorted)}")
    print("[interactive] controls: hover preview | left click lock | right click clear | n/p next/prev | o outline | i info | s save | c/esc clear")
    plt.show()


if __name__ == "__main__":
    main()