"""Uncertainty reporting for the FilaSeg locked synthetic evaluation.

Everything here operates on the *scene* as the unit of replication. Fragments
within one image are not independent replicates and are never bootstrapped.

Three quantities are provided:

    summarize(values)                 mean, sd, and a 95% bootstrap CI
    paired_summarize(a, b)            the same for the paired difference a - b
    ci_halfwidth(...)                 the achieved 95% CI half-width

The bootstrap is the ordinary non-parametric percentile bootstrap over scenes,
with a fixed seed so the reported interval is reproducible. The percentile
bootstrap is used rather than a t interval because the per-scene F1 values are
bounded in [0, 1] and visibly skewed at the densest tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Mapping, Sequence

import numpy as np

BOOTSTRAP_SEED = 20260729
N_BOOTSTRAP = 10000


@dataclass
class Summary:
    n: int
    mean: float
    sd: float
    ci_lo: float
    ci_hi: float
    ci_halfwidth: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                  alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI of the mean, over the scene axis."""
    n = values.size
    if n < 2:
        v = float(values[0]) if n == 1 else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def summarize(values: Sequence[float], seed: int = BOOTSTRAP_SEED,
              n_boot: int = N_BOOTSTRAP) -> Summary:
    """Mean, sample standard deviation and 95% bootstrap CI over scenes."""
    v = np.asarray([float(x) for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Summary(0, float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    lo, hi = _bootstrap_ci(v, n_boot, seed)
    return Summary(int(v.size), mean, sd, lo, hi, float((hi - lo) / 2.0))


def paired_summarize(a: Sequence[float], b: Sequence[float],
                     seed: int = BOOTSTRAP_SEED,
                     n_boot: int = N_BOOTSTRAP) -> Summary:
    """Summary of the paired difference a - b.

    `a` and `b` must be aligned scene by scene: element i of both must come
    from the SAME geometry seed. Pairing is what removes scene-to-scene
    geometry variance from the comparison, which is why the locked evaluation
    reuses one geometry for every compared method.
    """
    av = np.asarray([float(x) for x in a], dtype=float)
    bv = np.asarray([float(x) for x in b], dtype=float)
    if av.shape != bv.shape:
        raise ValueError(f"paired inputs must align: {av.shape} vs {bv.shape}")
    ok = np.isfinite(av) & np.isfinite(bv)
    return summarize(av[ok] - bv[ok], seed=seed, n_boot=n_boot)


def paired_summarize_by_scene(a: Mapping[str, float], b: Mapping[str, float],
                              seed: int = BOOTSTRAP_SEED,
                              n_boot: int = N_BOOTSTRAP) -> Summary:
    """Bootstrap paired method differences after exact scene-ID validation.

    The mapping keys are part of the statistical contract: a scene omitted by
    either method is an error, rather than an accidental unpaired comparison.
    IDs are sorted before conversion so dictionary insertion order cannot affect
    a seeded bootstrap result.
    """
    a_keys, b_keys = set(a), set(b)
    if a_keys != b_keys:
        missing_from_a = sorted(b_keys - a_keys)
        missing_from_b = sorted(a_keys - b_keys)
        raise ValueError(
            "scene IDs must match exactly; "
            f"missing from a={missing_from_a}, missing from b={missing_from_b}"
        )
    scene_ids = sorted(a_keys)
    return paired_summarize([a[k] for k in scene_ids], [b[k] for k in scene_ids],
                            seed=seed, n_boot=n_boot)


def needs_more_scenes(s: Summary, target_halfwidth: float = 0.05) -> bool:
    """Stopping rule declared before the locked evaluation was executed.

    If the achieved 95% CI half-width for the headline F1 of any density
    exceeds `target_halfwidth`, that density is extended to 30 or more scenes.
    """
    return bool(np.isfinite(s.ci_halfwidth) and s.ci_halfwidth > target_halfwidth)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Pearson correlation and the linear-fit R^2, with a bootstrap CI on r."""
    xv = np.asarray([float(v) for v in x], dtype=float)
    yv = np.asarray([float(v) for v in y], dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if xv.size < 3 or np.ptp(xv) == 0 or np.ptp(yv) == 0:
        return {"n": int(xv.size), "r": float("nan"), "r2": float("nan"),
                "r_ci_lo": float("nan"), "r_ci_hi": float("nan")}
    r = float(np.corrcoef(xv, yv)[0, 1])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, xv.size, size=(N_BOOTSTRAP, xv.size))
    # Degenerate resamples (all-identical x or y) give a 0/0 correlation; they
    # are dropped below, so silence the divide warning rather than the result.
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = np.array([np.corrcoef(xv[i], yv[i])[0, 1] for i in idx])
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return {"n": int(xv.size), "r": r, "r2": float(r * r),
                "r_ci_lo": float("nan"), "r_ci_hi": float("nan")}
    return {
        "n": int(xv.size),
        "r": r,
        "r2": float(r * r),
        "r_ci_lo": float(np.percentile(rs, 2.5)),
        "r_ci_hi": float(np.percentile(rs, 97.5)),
    }


def signed_bias_and_mape(measured: Sequence[float],
                         truth: Sequence[float]) -> Dict[str, float]:
    """Signed percentage bias and mean absolute percentage error.

    Reported separately and explicitly, rather than as a vague 'within X%',
    because a reproducible one-sided bias and a symmetric random error have
    different consequences for a downstream user.
    """
    m = np.asarray([float(v) for v in measured], dtype=float)
    t = np.asarray([float(v) for v in truth], dtype=float)
    ok = np.isfinite(m) & np.isfinite(t) & (np.abs(t) > 1e-12)
    pct = 100.0 * (m[ok] - t[ok]) / t[ok]
    s = summarize(pct)
    return {
        "n": int(pct.size),
        "bias_pct": s.mean,
        "bias_ci_lo": s.ci_lo,
        "bias_ci_hi": s.ci_hi,
        "mape_pct": float(np.mean(np.abs(pct))) if pct.size else float("nan"),
        "frac_overestimated": float(np.mean(pct > 0.0)) if pct.size else float("nan"),
    }
