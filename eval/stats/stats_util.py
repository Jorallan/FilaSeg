"""Uncertainty reporting for the FilaSeg locked synthetic evaluation.

Everything here operates on the *scene* as the unit of replication. Fragments
within one image are not independent replicates and are never bootstrapped.

Three quantities are provided:

    summarize(values)                 mean, sd, and a 95% bootstrap CI
    paired_summarize(a, b)            the same for the paired difference a - b
    stratified_summarize(values)      a within-stratum bootstrap CI
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


def _stratified_bootstrap_ci(values_by_stratum: Mapping[str, np.ndarray],
                             n_boot: int, seed: int,
                             alpha: float = 0.05) -> tuple:
    """Percentile CI after resampling scenes independently within strata.

    Every replicate contains the original number of finite scenes from every
    declared stratum.  Pooling those resamples therefore targets the ordinary
    scene-weighted mean, while respecting the density-stratified design.
    """
    groups = [(name, values_by_stratum[name])
              for name in sorted(values_by_stratum)]
    n = sum(values.size for _, values in groups)
    if n < 2:
        v = float(groups[0][1][0]) if n == 1 else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot, dtype=float)
    for _, values in groups:
        if values.size == 0:
            continue
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        totals += values[idx].sum(axis=1)
    means = totals / n
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


def stratified_summarize(values_by_stratum: Mapping[str, Sequence[float]],
                         seed: int = BOOTSTRAP_SEED,
                         n_boot: int = N_BOOTSTRAP) -> Summary:
    """Scene-weighted summary with a bootstrap stratified by design group.

    The point estimate and standard deviation are calculated after pooling all
    finite scene values.  Only the uncertainty interval is stratified: a
    replicate resamples each stratum independently, retaining that stratum's
    original scene count before pooling.  Stratum names are sorted so a seeded
    result does not depend on mapping insertion order.
    """
    finite_by_stratum = {
        name: np.asarray([float(x) for x in values], dtype=float)
        for name, values in values_by_stratum.items()
    }
    finite_by_stratum = {
        name: values[np.isfinite(values)]
        for name, values in finite_by_stratum.items()
    }
    pooled = np.concatenate([finite_by_stratum[name]
                             for name in sorted(finite_by_stratum)]) \
        if finite_by_stratum else np.array([], dtype=float)
    if pooled.size == 0:
        return Summary(0, float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))
    mean = float(pooled.mean())
    sd = float(pooled.std(ddof=1)) if pooled.size > 1 else float("nan")
    lo, hi = _stratified_bootstrap_ci(finite_by_stratum, n_boot, seed)
    return Summary(int(pooled.size), mean, sd, lo, hi, float((hi - lo) / 2.0))


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


def paired_stratified_summarize(
        a_by_stratum: Mapping[str, Sequence[float]],
        b_by_stratum: Mapping[str, Sequence[float]],
        seed: int = BOOTSTRAP_SEED,
        n_boot: int = N_BOOTSTRAP) -> Summary:
    """Stratified summary of paired differences, ``a - b``.

    Within every stratum, the two method vectors must have identical shape and
    the bootstrap samples the resulting differences by one shared index.  This
    preserves the locked scene pairing while resampling within density.
    """
    a_keys, b_keys = set(a_by_stratum), set(b_by_stratum)
    if a_keys != b_keys:
        missing_from_a = sorted(b_keys - a_keys)
        missing_from_b = sorted(a_keys - b_keys)
        raise ValueError(
            "stratum keys must match exactly; "
            f"missing from a={missing_from_a}, missing from b={missing_from_b}"
        )
    differences = {}
    for stratum in sorted(a_keys):
        av = np.asarray([float(x) for x in a_by_stratum[stratum]], dtype=float)
        bv = np.asarray([float(x) for x in b_by_stratum[stratum]], dtype=float)
        if av.shape != bv.shape:
            raise ValueError(
                f"paired inputs must align within stratum {stratum!r}: "
                f"{av.shape} vs {bv.shape}"
            )
        ok = np.isfinite(av) & np.isfinite(bv)
        differences[stratum] = av[ok] - bv[ok]
    return stratified_summarize(differences, seed=seed, n_boot=n_boot)


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


def paired_stratified_summarize_by_scene(
        a: Mapping[str, float], b: Mapping[str, float],
        stratum_by_scene: Mapping[str, str], seed: int = BOOTSTRAP_SEED,
        n_boot: int = N_BOOTSTRAP) -> Summary:
    """Stratified paired summary after exact scene and stratum-ID validation.

    ``a``, ``b``, and ``stratum_by_scene`` must describe precisely the same
    scenes.  Scene IDs are sorted within each stratum before conversion, making
    seeded results independent of dictionary insertion order.
    """
    a_keys, b_keys, stratum_keys = set(a), set(b), set(stratum_by_scene)
    if a_keys != b_keys:
        missing_from_a = sorted(b_keys - a_keys)
        missing_from_b = sorted(a_keys - b_keys)
        raise ValueError(
            "scene IDs must match exactly; "
            f"missing from a={missing_from_a}, missing from b={missing_from_b}"
        )
    if a_keys != stratum_keys:
        missing_from_strata = sorted(a_keys - stratum_keys)
        extra_in_strata = sorted(stratum_keys - a_keys)
        raise ValueError(
            "stratum scene IDs must match method scene IDs exactly; "
            f"missing from strata={missing_from_strata}, "
            f"extra in strata={extra_in_strata}"
        )
    a_by_stratum = {}
    b_by_stratum = {}
    for scene_id in sorted(a_keys):
        stratum = stratum_by_scene[scene_id]
        a_by_stratum.setdefault(stratum, []).append(a[scene_id])
        b_by_stratum.setdefault(stratum, []).append(b[scene_id])
    return paired_stratified_summarize(a_by_stratum, b_by_stratum,
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
