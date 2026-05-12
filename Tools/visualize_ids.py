"""Interactive label viewer for reconnect / postprocess / final outputs.

Auto-detects the labels TIFF and a matching background image in --input.
Hover to preview, left-click to lock, n/p to step, esc/right-click to clear,
o toggles outline, i toggles info, s saves the current view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from skimage import io as skio
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries

LABEL_PRIORITY  = ["*_post_labels.tif", "*_reconnect_labels.tif", "*_labels.tif", "*.tif"]
BG_PRIORITY     = ["*_original.*", "*_post_overlay.png", "*_reconnect_overlay.png", "*_overlay.png", "*_preview.png"]
DEFAULT_INPUT = Path(r"C:\Repos\filaments_quantification\output\full_pipeline\sem_full_00000_1p66_crop512_20260512_153551\final")


def find_first(folder: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(p for p in folder.glob(pat) if "dilated" not in p.stem and "_labels_preview" not in p.stem)
        if hits:
            return hits[0]
    return None


def load_rgb(path: Path) -> np.ndarray:
    img = skio.imread(str(path)).astype(np.float32)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = img[..., :3]
    if img.max() > 1.5:
        img /= 255.0 if img.max() <= 255 else img.max()
    return np.clip(img, 0.0, 1.0)


def compute_table(lbl: np.ndarray) -> dict[int, dict]:
    out = {}
    for cid in [int(v) for v in np.unique(lbl) if v]:
        pts = np.argwhere(lbl == cid)
        r0, c0 = pts.min(0); r1, c1 = pts.max(0); cr, cc = pts.mean(0)
        out[cid] = {"area": int(pts.shape[0]), "bbox": (int(r0), int(r1), int(c0), int(c1)),
                    "center": (float(cr), float(cc))}
    return out


def colorize(lbl: np.ndarray) -> np.ndarray:
    out = np.zeros((*lbl.shape, 3), dtype=np.float32)
    out[..., 0] = ((lbl.astype(np.int64) *  53) % 256) / 255.0
    out[..., 1] = ((lbl.astype(np.int64) *  97) % 256) / 255.0
    out[..., 2] = ((lbl.astype(np.int64) * 193) % 256) / 255.0
    out[lbl == 0] = 0
    return out


def render(bg: np.ndarray, lbl: np.ndarray, color: np.ndarray, sel: int | None,
           alpha: float, dim: float, gain: float, outline: bool) -> np.ndarray:
    out = bg.copy()
    m = lbl > 0
    out[m] = alpha * color[m] + (1.0 - alpha) * out[m]
    if sel is None or sel <= 0:
        return out
    sm = lbl == sel
    if not sm.any():
        return out
    out *= dim
    out[sm] = np.clip(out[sm] * gain + 0.15, 0.0, 1.0)
    if outline:
        out[find_boundaries(sm, mode="outer")] = (1.0, 0.1, 0.1)
    return out



# For IDE / direct script runs:
# - Set PROMPT_FOR_INPUT = True to type the folder path when the script starts.
# - Or set SCRIPT_INPUT to a Path and leave PROMPT_FOR_INPUT = False.
# CLI usage still works, e.g. python Tools/visualize_ids.py --input "path\to\final"
PROMPT_FOR_INPUT = False
SCRIPT_INPUT = DEFAULT_INPUT


def prompt_for_folder(default: Path | None = None) -> Path:
    prompt = "Folder to analyze"
    if default is not None:
        prompt += f" [{default}]"
    prompt += ": "
    value = input(prompt).strip().strip('"')
    if value:
        return Path(value)
    if default is not None:
        return default
    raise ValueError("No input folder provided.")


def display_name_for_folder(folder: Path) -> str:
    if folder.name.lower() in {"final", "3.reconnect", "4.postprocess"} and folder.parent.name:
        return f"{folder.parent.name}/{folder.name}"
    return folder.name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive label viewer (reconnect / postprocess / final).")
    ap.add_argument("--input", type=Path, default=None, help="Folder containing a labels TIFF and a background image.")
    ap.add_argument("--ask-input", action="store_true", help="Prompt for the input folder when the script starts.")
    ap.add_argument("--labels", type=Path, default=None, help="Optional explicit path to labels TIFF.")
    ap.add_argument("--background", type=Path, default=None, help="Optional explicit path to background image.")
    ap.add_argument("--dim_bg", type=float, default=0.55)
    ap.add_argument("--brighten_gain", type=float, default=1.60)
    ap.add_argument("--overlay_alpha", type=float, default=0.72)
    args = ap.parse_args()
    if args.ask_input or (PROMPT_FOR_INPUT and args.input is None):
        args.input = prompt_for_folder(SCRIPT_INPUT)
    elif args.input is None:
        args.input = SCRIPT_INPUT
    return args


def main() -> None:
    args = parse_args()
    folder = args.input.resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)

    lbl_path = args.labels or find_first(folder, LABEL_PRIORITY)
    bg_path  = args.background or find_first(folder, BG_PRIORITY)
    if lbl_path is None:
        raise FileNotFoundError(f"No labels TIFF found in {folder}")
    if bg_path is None:
        raise FileNotFoundError(f"No background image found in {folder}")

    lbl = skio.imread(str(lbl_path))
    if lbl.ndim != 2:
        raise ValueError(f"Labels must be 2D, got {lbl.shape}")
    lbl = lbl.astype(np.int32)
    bg = load_rgb(bg_path)
    if bg.shape[:2] != lbl.shape:
        raise ValueError(f"Background {bg.shape[:2]} != labels {lbl.shape}")

    table = compute_table(lbl)
    ids = sorted(table)
    color = colorize(lbl)
    state = {"hover": None, "sel": None, "locked": False, "outline": True, "info": True,
             "dim": float(args.dim_bg), "gain": float(args.brighten_gain), "alpha": float(args.overlay_alpha)}

    display_name = display_name_for_folder(folder)
    fig = plt.figure(figsize=(13, 8), num=f"{display_name} | {lbl_path.name}")
    gs = fig.add_gridspec(13, 1)
    ax = fig.add_subplot(gs[:10, 0]); ax.set_axis_off()
    ax_alpha = fig.add_subplot(gs[10, 0])
    ax_dim   = fig.add_subplot(gs[11, 0])
    ax_gain  = fig.add_subplot(gs[12, 0])
    plt.subplots_adjust(hspace=0.45)

    im = ax.imshow(render(bg, lbl, color, None, state["alpha"], state["dim"], state["gain"], state["outline"]), interpolation="nearest")
    ax.set_title(f"{display_name} | {len(ids)} ids | hover preview | click lock | rclick/esc clear | n/p step | o outline | i info | s save")
    info_text = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=10, color="white",
                        ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.6, pad=4))

    s_alpha = Slider(ax_alpha, "overlay_alpha", 0.00, 1.00, valinit=state["alpha"], valstep=0.01)
    s_dim   = Slider(ax_dim,   "dim_bg",        0.10, 1.00, valinit=state["dim"],   valstep=0.01)
    s_gain  = Slider(ax_gain,  "brighten_gain", 1.00, 3.00, valinit=state["gain"],  valstep=0.01)

    def current() -> int | None:
        return state["sel"] if state["locked"] else state["hover"]

    def update_text(cid: int | None):
        if not state["info"] or cid is None or cid not in table:
            info_text.set_text(""); return
        m = table[cid]; r0, r1, c0, c1 = m["bbox"]; cr, cc = m["center"]
        info_text.set_text(f"id: {cid}\narea: {m['area']} px\nbbox: r[{r0}:{r1}] c[{c0}:{c1}]\ncenter: ({cr:.1f}, {cc:.1f})")

    def redraw():
        cid = current()
        im.set_data(render(bg, lbl, color, cid, state["alpha"], state["dim"], state["gain"], state["outline"]))
        update_text(cid)
        fig.canvas.draw_idle()

    def pick(event) -> int | None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return None
        r, c = int(round(event.ydata)), int(round(event.xdata))
        if not (0 <= r < lbl.shape[0] and 0 <= c < lbl.shape[1]):
            return None
        cid = int(lbl[r, c])
        return cid if cid > 0 else None

    def on_move(event):
        if state["locked"]: return
        cid = pick(event)
        if cid != state["hover"]:
            state["hover"] = cid; redraw()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button == 1:
            cid = pick(event)
            if cid is None:
                state["locked"] = False; state["sel"] = None; state["hover"] = None
            else:
                state["locked"] = True; state["sel"] = cid; state["hover"] = cid
            redraw()
        elif event.button == 3:
            state["locked"] = False; state["sel"] = None; state["hover"] = None
            redraw()

    def step(direction: int):
        if not ids: return
        cid = current()
        if cid is None or cid not in table:
            cid = ids[0]
        else:
            cid = ids[(ids.index(cid) + direction) % len(ids)]
        state["locked"] = True; state["sel"] = cid; state["hover"] = cid
        redraw()

    def on_key(event):
        if event.key in ("escape", "c"):
            state["locked"] = False; state["sel"] = None; state["hover"] = None; redraw()
        elif event.key == "n": step(+1)
        elif event.key == "p": step(-1)
        elif event.key == "o": state["outline"] = not state["outline"]; redraw()
        elif event.key == "i": state["info"]    = not state["info"];    redraw()
        elif event.key == "s":
            cid = current()
            img = render(bg, lbl, color, cid, state["alpha"], state["dim"], state["gain"], state["outline"])
            suffix = f"_id{cid}" if cid is not None else "_full"
            out_path = folder / f"{lbl_path.stem}_view{suffix}.png"
            skio.imsave(str(out_path), np.clip(img * 255.0, 0, 255).astype(np.uint8), check_contrast=False)
            print(f"[saved] {out_path}")

    def on_slider(_):
        state["alpha"] = float(s_alpha.val)
        state["dim"]   = float(s_dim.val)
        state["gain"]  = float(s_gain.val)
        redraw()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)
    s_alpha.on_changed(on_slider)
    s_dim.on_changed(on_slider)
    s_gain.on_changed(on_slider)

    print(f"[viewer] labels: {lbl_path}")
    print(f"[viewer] background: {bg_path}")
    print(f"[viewer] ids: {len(ids)}")
    plt.show()


if __name__ == "__main__":
    main()
