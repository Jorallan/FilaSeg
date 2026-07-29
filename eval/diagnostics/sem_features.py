"""SEM-bridge feature sampling — the evidence geometry lacks.

Given the SEM image and two endpoints, sample intensity along the straight
bridge between them (plus normal offsets) and return features that should
distinguish a *real* gap (intensity continues across it — the UNet dropped the
pixels, the SEM didn't) from a *wrong* bridge (collinear but actually two
different filaments — the bridge crosses dark background).

Shared by the skeleton gap-bridge and the reconnect candidate evaluator.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


def _sample(sem: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    return map_coordinates(sem, np.vstack([rr, cc]), order=1, mode="nearest")


def sample_bridge_sem(
    sem: np.ndarray,
    p0_rc: Sequence[float],
    p1_rc: Sequence[float],
    n: int = 0,
    normal_offsets: Tuple[float, ...] = (2.0, 4.0),
) -> Dict[str, float]:
    """SEM features for the straight bridge p0->p1 (both (row, col)).

    Features (all higher = more like a real continuation):
      bridge_mean        mean intensity along the bridge centerline
      bridge_min         darkest point on the centerline (a real filament has no
                         fully-dark point; a wrong bridge crosses background)
      bridge_contrast    centerline mean minus the far off-bridge background
      bridge_min_contrast weakest centerline-vs-background contrast along bridge
      ridge_frac         fraction of samples where the centerline is brighter
                         than its near normal neighbours (an actual bright ridge)
    """
    p0 = np.asarray(p0_rc, dtype=np.float64)
    p1 = np.asarray(p1_rc, dtype=np.float64)
    d = float(np.hypot(*(p1 - p0)))
    npts = int(max(5, round(d))) if n <= 0 else int(n)
    t = np.linspace(0.0, 1.0, npts)
    rr = p0[0] + t * (p1[0] - p0[0])
    cc = p0[1] + t * (p1[1] - p0[1])
    center = _sample(sem, rr, cc)

    dirv = (p1 - p0) / max(1e-6, d)
    normal = np.asarray([-dirv[1], dirv[0]], dtype=np.float64)

    far = max(normal_offsets)
    near = min(normal_offsets)
    bg_far, ridge_hits = [], np.ones(npts, dtype=bool)
    for off in normal_offsets:
        for sgn in (+1.0, -1.0):
            s = _sample(sem, rr + sgn * off * normal[0], cc + sgn * off * normal[1])
            if off == far:
                bg_far.append(s)
            if off == near:
                ridge_hits &= (center >= s)
    bg = np.mean(bg_far, axis=0) if bg_far else np.zeros(npts)

    return {
        "bridge_mean": float(center.mean()),
        "bridge_min": float(center.min()),
        "bridge_contrast": float((center - bg).mean()),
        "bridge_min_contrast": float((center - bg).min()),
        "ridge_frac": float(ridge_hits.mean()),
    }


SEM_FEATURE_NAMES = (
    "bridge_mean", "bridge_min", "bridge_contrast",
    "bridge_min_contrast", "ridge_frac",
)
