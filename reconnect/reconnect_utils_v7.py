
from __future__ import annotations

"""
reconnect_utils_v7 — curvy-filament-aware reconnection.

Changes vs v6
-------------
1. Arc-first gate order: arc_miss is computed BEFORE the strict straight-line
   and angular gates, so it can be used to relax them rather than being a
   final sanity check that never runs for curvy pairs.

2. Curvature-adaptive angular relaxation: local tip curvature drives a
   continuous relaxation of min_forward_cos and min_inward_opposition.
   Curved-segment tips that are naturally misaligned from the gap direction
   are no longer rejected before the arc check runs.

3. OR-gate for line_residual vs arc_miss: a pair passes if EITHER the
   straight-line perpendicular residual is within tolerance OR the mutual
   arc-prediction miss is within max_arc_miss_px.  This is the critical
   change for slightly curved filaments.

4. Curvature-delta gate conditioned on arc agreement: curv_delta is only
   enforced when the arc predictions also disagree.  Two fragments with
   locally different curvature estimates still connect if their arcs land
   near each other.

5. max_turn_deg tolerance is relaxed proportionally to mean curvature,
   since a smooth Hermite bridge over a curved gap will always exhibit a
   local turning angle.

6. arc_miss is added as a score term so the ranker prefers pairs whose
   extrapolated arcs agree best.

7. Implementation uses a temporary monkey-patch of v6's _evaluate_tip_pair
   inside reconnect_components so the ~230-line engine is not duplicated.

New config keys (with suggested defaults — see reconnect_config_v7.yaml):
  thresholds:
    max_arc_miss_px: 15.0      # absolute arc-miss threshold in pixels
  advanced:
    curv_relax_factor: 30.0    # curvature × factor → cosine slack
    max_curv_relax: 0.30       # cap on curvature-driven slack
  weights:
    arc_miss: 0.8              # score penalty per unit of arc_miss_norm
"""

import csv
import math
import os
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import binary_dilation

# ── public API re-exported unchanged ─────────────────────────────────────
from reconnect_utils_v6 import (                             # noqa: F401
    Coord,
    read_gray_any, binarize_mask, clean_binary_mask,
    branchpoints_8, remove_branchpoint_regions,
    endpoints_from_skeleton, skeletonize,
    extract_components, relabel_components, dilate_label_image, make_overlay,
    Component, Proposal, cosine, draw_gap,
)

# ── private helpers needed inside the overridden functions ────────────────
from reconnect_utils_v6 import (                             # noqa: F401
    _tip_data, _tip_trace_length,
    _perp_distance_point_to_line,
    _sample_hermite_bridge, _polyline_mask, _arc_predicted_position,
    _local_smoothness_metrics,
    _bbox_from_mask, _bbox_intersects,
    _component_union_label, _sample_labels_in_neighborhood,
    _enumerate_tip_geoms, _dedupe_tip_points, trace_from_tip,
    _choose_tip_pair,
)

import reconnect_utils_v6 as _v6


# ─────────────────────────────────────────────────────────────────────────
# Rejection logger — writes every pair evaluation to CSV
# ─────────────────────────────────────────────────────────────────────────

_LOG_FH = None
_LOG_WRITER = None
_LOG_PATHS_WRITTEN: set = set()
_CURRENT_STAGE: str = ""
_LOG_COLUMNS = [
    "stage",
    "base_id", "tar_id", "base_tip", "tar_tip",
    "base_tip_r", "base_tip_c", "tar_tip_r", "tar_tip_c",
    "reason",
    "dist", "forward_base", "forward_tar", "inward_opposition",
    "width_ratio", "line_resid", "arc_miss_px", "curv_delta",
    "intrusion_frac", "smooth_rms", "max_turn_deg",
    "eff_forward_cos", "eff_opposition", "eff_max_turn",
    "mean_curv", "curv_relax", "arc_ok", "line_ok",
    "clear_merge",
]


def _open_rejection_log(path: str) -> None:
    global _LOG_FH, _LOG_WRITER
    _close_rejection_log()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    abs_path = os.path.abspath(path)
    write_header = abs_path not in _LOG_PATHS_WRITTEN
    mode = "w" if write_header else "a"
    _LOG_FH = open(path, mode, newline="", encoding="utf-8")
    _LOG_WRITER = csv.DictWriter(_LOG_FH, fieldnames=_LOG_COLUMNS)
    if write_header:
        _LOG_WRITER.writeheader()
        _LOG_PATHS_WRITTEN.add(abs_path)


def _close_rejection_log() -> None:
    global _LOG_FH, _LOG_WRITER
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except Exception:
            pass
    _LOG_FH = None
    _LOG_WRITER = None


def _log_row(rec: Dict) -> None:
    if _LOG_WRITER is None:
        return
    rec.setdefault("stage", _CURRENT_STAGE)
    row = {k: rec.get(k, "") for k in _LOG_COLUMNS}
    _LOG_WRITER.writerow(row)
    try:
        _LOG_FH.flush()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Curvy-aware pair evaluator
# ─────────────────────────────────────────────────────────────────────────

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

    bp, bd, bcurv, btrace = _tip_data(base, base_tip_name)
    tp, td, tcurv, ttrace = _tip_data(tar,  tar_tip_name)

    rec: Dict = {
        "base_id": int(base.id),
        "tar_id": int(tar.id),
        "base_tip": base_tip_name,
        "tar_tip": tar_tip_name,
        "base_tip_r": float(bp[0]),
        "base_tip_c": float(bp[1]),
        "tar_tip_r": float(tp[0]),
        "tar_tip_c": float(tp[1]),
    }

    def reject(reason: str):
        rec["reason"] = reason
        _log_row(rec)
        return None

    bp_arr = np.asarray([float(bp[0]), float(bp[1])], dtype=np.float32)
    tp_arr = np.asarray([float(tp[0]), float(tp[1])], dtype=np.float32)
    min_tip_support = float(adv.get("min_tip_trace_len_px", 3.0))
    if _tip_trace_length(btrace) < min_tip_support or _tip_trace_length(ttrace) < min_tip_support:
        return reject("tip_trace_too_short")

    dvec = tp_arr - bp_arr
    dist = float(np.linalg.norm(dvec))
    rec["dist"] = dist
    max_tip_dist = float(thr.get("max_tip_distance_px", 40))
    if dist <= 1e-6 or dist > max_tip_dist:
        return reject("distance")
    u = dvec / dist

    # Standard geometry metrics
    inward_opposition = float(cosine(bd, td))
    forward_base = float(cosine(-bd, u))
    forward_tar  = float(cosine( td, u))
    width_ratio  = float(
        max(base.mean_width, tar.mean_width) / max(1e-6, min(base.mean_width, tar.mean_width))
    )
    line_resid_base = _perp_distance_point_to_line(tp, bp, -bd)
    line_resid_tar  = _perp_distance_point_to_line(bp, tp,  td)
    line_resid  = 0.5 * (line_resid_base + line_resid_tar)
    curv_delta  = float(abs(bcurv - tcurv))

    rec.update({
        "forward_base": forward_base,
        "forward_tar": forward_tar,
        "inward_opposition": inward_opposition,
        "width_ratio": width_ratio,
        "line_resid": line_resid,
        "curv_delta": curv_delta,
    })

    # ── v7 ① arc miss computed BEFORE the angular gates ──────────────────
    arc_miss_px: Optional[float] = None
    b_extra = (base.tips or {}).get(base_tip_name, {}).get("extra", None)
    t_extra = (tar.tips  or {}).get(tar_tip_name,  {}).get("extra", None)
    if b_extra is not None and t_extra is not None and dist > 2.0:
        bp_xy = np.asarray([float(bp[1]), float(bp[0])], dtype=np.float32)
        tp_xy = np.asarray([float(tp[1]), float(tp[0])], dtype=np.float32)
        pred_from_base = _arc_predicted_position(b_extra, dist)
        pred_from_tar  = _arc_predicted_position(t_extra, dist)
        arc_miss_px = 0.5 * (
            float(np.linalg.norm(pred_from_base - tp_xy))
            + float(np.linalg.norm(pred_from_tar  - bp_xy))
        )

    max_arc_miss_px = float(thr.get("max_arc_miss_px", 15.0))
    arc_ok = arc_miss_px is not None and arc_miss_px <= max_arc_miss_px

    # ── v7 ② curvature-adaptive angular relaxation ───────────────────────
    mean_curv         = 0.5 * (float(bcurv) + float(tcurv))
    curv_relax_factor = float(adv.get("curv_relax_factor", 30.0))
    max_curv_relax    = float(adv.get("max_curv_relax",    0.30))
    curv_relax        = min(max_curv_relax, mean_curv * curv_relax_factor)
    # Extra slack when arc prediction already confirms alignment
    arc_boost = 0.06 if arc_ok else 0.0

    min_forward_cos       = float(thr.get("min_forward_cos",        0.65))
    min_inward_opposition = float(thr.get("min_inward_opposition",  0.60))
    max_width_ratio       = float(thr.get("max_width_ratio",        3.0))
    max_line_residual     = float(thr.get("max_line_residual_px",   8.0))
    max_curv_delta        = float(thr.get("max_curvature_delta",    0.15))

    eff_forward_cos = max(0.08, min_forward_cos       - curv_relax - arc_boost)
    eff_opposition  = max(0.03, min_inward_opposition - curv_relax - arc_boost)

    # ── v7 ⑦ clear-merge override ────────────────────────────────────────
    # When tips are close, collinear, and face each other, bypass the gates
    # that are noise-sensitive at short distances (forward_cos from jittery
    # tip directions, max_turn from trace↔bridge seam kinks, intrusion_frac
    # which blows up as 1-label/small-dist). Line + arc + opposition are all
    # measured from stable quantities, so they remain the source of truth.
    clear_dist = float(thr.get("clear_merge_max_dist_px",       15.0))
    clear_line = float(thr.get("clear_merge_max_line_resid_px",  4.0))
    clear_arc  = float(thr.get("clear_merge_max_arc_miss_px",    5.0))
    clear_opp  = float(thr.get("clear_merge_min_opposition",     0.85))
    arc_tight  = (dist <= 2.0) or (arc_miss_px is not None and arc_miss_px <= clear_arc)
    clear_merge = (
        dist <= clear_dist
        and line_resid <= clear_line
        and arc_tight
        and (-inward_opposition) >= clear_opp
    )

    rec.update({
        "arc_miss_px": arc_miss_px if arc_miss_px is not None else -1.0,
        "arc_ok": int(bool(arc_ok)),
        "mean_curv": mean_curv,
        "curv_relax": curv_relax,
        "eff_forward_cos": eff_forward_cos,
        "eff_opposition": eff_opposition,
        "clear_merge": int(bool(clear_merge)),
    })

    if not clear_merge and (forward_base < eff_forward_cos or forward_tar < eff_forward_cos):
        return reject("forward_cos")
    if (-inward_opposition) < eff_opposition:
        return reject("opposition")
    if width_ratio > max_width_ratio:
        return reject("width_ratio")

    # ── v7 ③ line residual: OR-gate with arc miss ─────────────────────────
    line_ok = line_resid <= max_line_residual
    rec["line_ok"] = int(bool(line_ok))
    if not (line_ok or arc_ok):
        return reject("line_residual_and_arc")

    # ── v7 ④ curvature delta only gated when arc also disagrees ──────────
    if curv_delta > max_curv_delta and not arc_ok:
        return reject("curv_delta")

    # Hermite bridge + intrusion check (unchanged from v6)
    bridge_pts = _sample_hermite_bridge(
        bp, tp,
        -bd, td,
        n_samples=int(adv.get("bridge_samples", 24)),
        tangent_scale=float(adv.get("bridge_tangent_scale", 0.35)),
    )
    bridge_mask = _polyline_mask(bridge_pts, base.mask.shape)

    clearance_px = int(max(0, adv.get("bridge_clearance_px", 1)))
    bridge_eval  = binary_dilation(bridge_mask, iterations=clearance_px) if clearance_px > 0 else bridge_mask
    touched = global_label[bridge_eval]
    touched = touched[(touched != 0) & (touched != base.id) & (touched != tar.id)]
    intrusion_frac = float(len(np.unique(touched))) / max(1.0, dist)
    rec["intrusion_frac"] = intrusion_frac
    max_intrusion  = float(thr.get("max_bridge_intrusion_frac", 0.10))
    if not clear_merge and intrusion_frac > max_intrusion:
        return reject("bridge_intrusion")

    smooth_rms, smooth_curv, max_turn_deg = _local_smoothness_metrics(
        btrace[: int(adv.get("smooth_trace_points", 18))] if btrace is not None else btrace,
        bridge_pts,
        ttrace[: int(adv.get("smooth_trace_points", 18))] if ttrace is not None else ttrace,
        fit_degree=2,
    )
    max_smooth_rms = float(thr.get("max_smooth_rms_px",  3.5))
    max_turn_gate  = float(thr.get("max_turn_deg",       45.0))

    # ── v7 ⑤ relaxed turn tolerance for curved bridges ───────────────────
    # Hard cap prevents the curvature relaxation from opening the gate to
    # near-right-angle turns (stage2 was reaching ~96° before this cap).
    turn_hard_cap = float(thr.get("max_turn_deg_hard_cap", 75.0))
    eff_max_turn  = min(turn_hard_cap, max_turn_gate + curv_relax * 20.0)

    # Pairs rescued only by arc agreement (failed straight-line residual)
    # must still be geometrically smooth — if the turn is sharp the arc
    # prediction happened to agree by coincidence, not genuine curvature.
    if arc_ok and not line_ok:
        arc_rescue_cap = float(thr.get("max_turn_deg_arc_rescue", 50.0))
        eff_max_turn   = min(eff_max_turn, arc_rescue_cap)

    rec.update({
        "smooth_rms": float(smooth_rms),
        "max_turn_deg": float(max_turn_deg),
        "eff_max_turn": float(eff_max_turn),
    })

    if smooth_rms > max_smooth_rms:
        return reject("smooth_rms")
    if not clear_merge and max_turn_deg > eff_max_turn:
        return reject("max_turn")

    # ── v7 ⑥ score includes arc_miss term ────────────────────────────────
    arc_miss_norm = (arc_miss_px / max(1.0, max_arc_miss_px)) if arc_miss_px is not None else 1.0

    score_scalar = (
        float(wgt.get("distance",          1.0)) * (dist / max(1.0, max_tip_dist))
        + float(wgt.get("forward",         1.2)) * ((1.0 - forward_base) + (1.0 - forward_tar))
        + float(wgt.get("opposition",      1.0)) * (1.0 + inward_opposition)
        + float(wgt.get("line_residual",   0.9)) * (line_resid / max(1.0, max_line_residual))
        + float(wgt.get("width_ratio",     0.5)) * math.log(max(1.0, width_ratio))
        + float(wgt.get("curvature_delta", 0.5)) * (curv_delta / max(1e-6, max_curv_delta))
        + float(wgt.get("smooth_rms",      1.2)) * (smooth_rms / max(1e-6, max_smooth_rms))
        + float(wgt.get("smooth_curvature",0.5)) * smooth_curv
        + float(wgt.get("bridge_intrusion",1.5)) * intrusion_frac
        + float(wgt.get("arc_miss",        0.8)) * arc_miss_norm
        - float(wgt.get("length_reward",   0.02)) * math.log1p(max(base.skel_len, tar.skel_len))
    )

    metrics = {
        "dist":              dist,
        "forward_base":      forward_base,
        "forward_tar":       forward_tar,
        "inward_opposition": inward_opposition,
        "width_ratio":       width_ratio,
        "line_residual":     line_resid,
        "curv_delta":        curv_delta,
        "arc_miss_px":       arc_miss_px if arc_miss_px is not None else -1.0,
        "intrusion_frac":    intrusion_frac,
        "smooth_rms":        smooth_rms,
        "smooth_curv":       smooth_curv,
        "max_turn_deg":      max_turn_deg,
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
    rec["reason"] = "accepted"
    _log_row(rec)
    return Proposal(
        base_id=base.id,
        tar_id=tar.id,
        base_tip_name=base_tip_name,
        tar_tip_name=tar_tip_name,
        bridge_points=bridge_pts,
        score=score,
        metrics=metrics,
    )


# ─────────────────────────────────────────────────────────────────────────
# reconnect_components — injects v7 evaluator into v6's engine
# ─────────────────────────────────────────────────────────────────────────

def reconnect_components(components, cfg):
    """
    Identical logic to v6.reconnect_components, but uses v7's curvy-aware
    _evaluate_tip_pair.  Implemented via a temporary monkey-patch so the
    ~230-line greedy engine is not duplicated.
    """
    global _CURRENT_STAGE
    debug_cfg = cfg.get("debug", {}) or {}
    log_path = debug_cfg.get("rejection_log_path")
    stage_label = str(debug_cfg.get("stage_label", ""))
    orig = _v6._evaluate_tip_pair
    prev_stage = _CURRENT_STAGE
    try:
        if log_path:
            _open_rejection_log(str(log_path))
        _CURRENT_STAGE = stage_label
        _v6._evaluate_tip_pair = _evaluate_tip_pair
        result = _v6.reconnect_components(components, cfg)
    finally:
        _v6._evaluate_tip_pair = orig
        _CURRENT_STAGE = prev_stage
        _close_rejection_log()
    return result
