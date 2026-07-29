"""Unit tests for the dimensionally-corrected tip-curvature estimator.

Run with:
    C:\\Repos\\venv_cnt\\Scripts\\python.exe -m unittest tests.test_curvature -v

These tests load `_fit_local_tangent_and_curvature` directly out of
`3.reconnect/reconnect_utils_straight.py` via importlib (the directory name
starts with a digit, so it cannot be imported as a normal package/module).

The correction under test:
    kappa = ( |2a| / Lambda^2 ) / ( 1 + (b_eff/Lambda)^2 )^{3/2}
    q_fit = rms / Lambda
where the quadratic fit n(su) = a*su^2 + b*su + c is over the NORMALIZED arc
length su = s/Lambda, and b_eff = 2*a*su_tip + b is the slope at the tip
(dn/d(su) there). kappa is an inverse length everywhere; q_fit is a separate,
dimensionless roughness measure that must never be added into kappa.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "3.reconnect" / "reconnect_utils_straight.py"

_spec = importlib.util.spec_from_file_location("reconnect_utils_straight", str(_MODULE_PATH))
_rus = importlib.util.module_from_spec(_spec)
# Must be registered in sys.modules BEFORE exec: the module defines a
# @dataclass (Component) whose type-resolution machinery looks the module
# up by name in sys.modules while it is being executed.
sys.modules[_spec.name] = _rus
_spec.loader.exec_module(_rus)

_fit_local_tangent_and_curvature = _rus._fit_local_tangent_and_curvature


# ── synthetic trace builders ──────────────────────────────────────────────

def make_circular_arc(radius: float, arc_length: float = 28.0, step: float = 1.0) -> np.ndarray:
    """Ordered (row, col) trace, tip first, along a circle of the given radius.

    Sampled at `step` px of arc length, matching production's 1 px/step trace
    and the default trace_steps: 28 (see reconnect_config.yaml).
    """
    n = int(round(arc_length / step)) + 1
    s = np.arange(n, dtype=np.float64) * step
    theta = s / radius
    col = radius * np.sin(theta)
    row = radius * (1.0 - np.cos(theta))
    return np.stack([row, col], axis=1)


def make_straight_line(
    length: float = 28.0, step: float = 1.0, angle: float = 0.37,
    noise_sigma: float = 0.0, rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Ordered (row, col) trace, tip first, along a straight line at `angle`
    (an arbitrary non-axis-aligned direction), optionally with additive
    Gaussian normal noise of std `noise_sigma` px.
    """
    n = int(round(length / step)) + 1
    t = np.arange(n, dtype=np.float64) * step
    row = t * np.sin(angle)
    col = t * np.cos(angle)
    pts = np.stack([row, col], axis=1)
    if noise_sigma > 0.0:
        assert rng is not None
        pts = pts + rng.normal(0.0, noise_sigma, size=pts.shape)
    return pts


def old_formula_curvature(trace_rc: np.ndarray, fit_points: int) -> float:
    """Re-implementation of the OLD (defective) curvature formula, kept
    local to this test so the regression guard does not depend on the
    production code having the bug (it no longer does).

        curvature = abs(2a)/Lambda^2 + 0.10 * rms/Lambda

    This mirrors the same PCA-tangent + normalized-su quadratic fit used by
    `_fit_local_tangent_and_curvature`, but applies the old, dimensionally
    inconsistent combination rule.
    """
    pts = np.asarray(trace_rc[:fit_points], dtype=np.float64)
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
    ctr = xy.mean(axis=0, keepdims=True)
    X = xy - ctr
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    axis_xy = v[:, int(np.argmax(w))]
    if float(np.dot(axis_xy, xy[-1] - xy[0])) < 0.0:
        axis_xy = -axis_xy
    normal_xy = np.asarray([-axis_xy[1], axis_xy[0]])
    rel = xy - xy[0:1]
    s0 = (rel @ axis_xy) - (rel @ axis_xy).min()
    span = max(1e-6, s0.max())
    su = s0 / span
    n = rel @ normal_xy
    coeff = np.polyfit(su, n, deg=2)
    pred = np.polyval(coeff, su)
    a = float(coeff[0])
    rms = float(np.sqrt(np.mean((n - pred) ** 2)))
    return abs(2.0 * a) / (span * span + 1e-9) + 0.10 * rms / (span + 1e-9)


class TestTipCurvature(unittest.TestCase):

    def test_production_fit_window(self):
        """The active 12-point fit window retains inverse-pixel behaviour."""
        fit_points = 12
        for radius in (60.0, 120.0, 240.0):
            trace = make_circular_arc(radius, arc_length=28.0, step=1.0)
            _dvec, kappa, extra = _fit_local_tangent_and_curvature(
                trace, fit_points
            )
            expected = 1.0 / radius
            rel_err = abs(kappa - expected) / expected
            self.assertLess(
                rel_err, 0.12,
                f"production fit_points={fit_points}, R={radius}: "
                f"kappa={kappa}, expected={expected}",
            )
            self.assertGreaterEqual(extra["q_fit"], 0.0)

    def test_circular_arcs_of_known_radius(self):
        # Measured relative errors at arc_length=28px, 1px sampling (this
        # test's own run): R=30 -> 22.3%, R=60 -> 6.5%, R=120 -> 1.7%,
        # R=240 -> 0.4%. All comfortably clear a uniform 25% bound, so the
        # tighter 25% tolerance is used for every radius including R=30; the
        # 35%-for-R=30 fallback described in the spec is not needed here but
        # is documented below in case discretization behaves differently
        # under a future change to trace_steps or sampling.
        radii = (30, 60, 120, 240)
        for R in radii:
            trace = make_circular_arc(R, arc_length=28.0, step=1.0)
            _dvec, kappa, extra = _fit_local_tangent_and_curvature(trace, len(trace))
            expected = 1.0 / R
            rel_err = abs(kappa - expected) / expected
            tol = 0.25  # uniform 25% bound; see comment above for R=30 measurement
            print(f"[circular arc] R={R:>4d}px  kappa={kappa:.6f}  1/R={expected:.6f}  "
                  f"rel_err={rel_err:.4f}  tol={tol}  q_fit={extra['q_fit']:.6f}")
            self.assertLess(rel_err, tol,
                             f"R={R}: kappa={kappa} vs 1/R={expected}, rel_err={rel_err} > {tol}")

    def test_straight_line(self):
        trace = make_straight_line(length=28.0, step=1.0, angle=0.37)
        _dvec, kappa, extra = _fit_local_tangent_and_curvature(trace, len(trace))
        print(f"[straight line] kappa={kappa:.3e}  q_fit={extra['q_fit']:.3e}")
        self.assertLess(kappa, 1e-3)
        self.assertLess(extra["q_fit"], 1e-6)

    def test_noisy_straight_line(self):
        rng = np.random.default_rng(0)
        trace = make_straight_line(length=28.0, step=1.0, angle=0.37, noise_sigma=0.4, rng=rng)
        _dvec, kappa, extra = _fit_local_tangent_and_curvature(trace, len(trace))
        print(f"[noisy straight] kappa={kappa:.6f}  q_fit={extra['q_fit']:.6f}")
        # Roughness (q_fit) must be clearly non-zero, while curvature must
        # NOT be contaminated by it (this is the core dimensional fix).
        self.assertLess(kappa, 0.02)
        self.assertGreater(extra["q_fit"], 0.005)

    def test_inverse_length_scaling_under_resampling(self):
        # Same physical arc, sampled at 1x (R=100, 28px) and 2x (R=200, 56px)
        # pixel resolution. kappa is an inverse length, so it must halve.
        trace_1x = make_circular_arc(100.0, arc_length=28.0, step=1.0)
        trace_2x = make_circular_arc(200.0, arc_length=56.0, step=1.0)
        _d1, kappa_1x, _e1 = _fit_local_tangent_and_curvature(trace_1x, len(trace_1x))
        _d2, kappa_2x, _e2 = _fit_local_tangent_and_curvature(trace_2x, len(trace_2x))
        ratio = kappa_2x / kappa_1x
        print(f"[scaling] kappa_1x={kappa_1x:.6f}  kappa_2x={kappa_2x:.6f}  ratio={ratio:.4f}")
        self.assertGreaterEqual(ratio, 0.4)
        self.assertLessEqual(ratio, 0.6)

    def test_q_fit_scale_invariance(self):
        # Same physical noisy arc, sampled at 1x and 2x resolution, with the
        # noise amplitude scaled along with the pixels (2x pixels -> 2x px
        # noise sigma for the same physical roughness). q_fit = rms/Lambda
        # is dimensionless and must be scale-invariant.
        rng_1x = np.random.default_rng(1)
        rng_2x = np.random.default_rng(1)
        trace_1x = make_circular_arc(100.0, arc_length=28.0, step=1.0) \
            + rng_1x.normal(0.0, 0.4, size=(29, 2))
        trace_2x = make_circular_arc(200.0, arc_length=56.0, step=1.0) \
            + rng_2x.normal(0.0, 0.8, size=(57, 2))
        _d1, _k1, e1 = _fit_local_tangent_and_curvature(trace_1x, len(trace_1x))
        _d2, _k2, e2 = _fit_local_tangent_and_curvature(trace_2x, len(trace_2x))
        q1, q2 = e1["q_fit"], e2["q_fit"]
        ratio = q2 / q1
        print(f"[q_fit scale invariance] q_fit_1x={q1:.6f}  q_fit_2x={q2:.6f}  ratio={ratio:.4f}")
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_regression_guard_against_old_formula(self):
        # This test documents the behavioural change from the defective old
        # formula (curvature = |2a|/Lambda^2 + 0.10*rms/Lambda, which adds a
        # dimensionless roughness term directly into an inverse-pixel
        # curvature) to the corrected one, which keeps q_fit fully separate.
        #
        # A straight-but-rough trace like test_noisy_straight_line's (28 px,
        # sigma=0.4) only overstates the old formula by ~1.5-2x, because at
        # that short length the added 0.10*rms/span term is not yet much
        # larger than the (already tiny) true-curvature term shared by both
        # formulas. Since the old term scales as O(1/Lambda) while the
        # shared/legitimate leading-order term scales as O(1/Lambda^2), the
        # relative overstatement of the old formula GROWS with trace length.
        # A 140 px straight-but-rough trace (5x the production 28 px
        # trace_steps) makes this dominance obvious and gives a robust >=5x
        # margin (checked across seeds 0-14: min observed ratio ~5.5x).
        rng = np.random.default_rng(0)
        trace = make_straight_line(length=140.0, step=1.0, angle=0.37, noise_sigma=0.4, rng=rng)
        _dvec, kappa_new, _extra = _fit_local_tangent_and_curvature(trace, len(trace))
        kappa_old = old_formula_curvature(trace, len(trace))
        ratio = kappa_old / max(kappa_new, 1e-12)
        print(f"[regression guard] kappa_new={kappa_new:.6e}  kappa_old={kappa_old:.6e}  "
              f"old/new={ratio:.2f}")
        self.assertGreaterEqual(ratio, 5.0,
                                 f"expected new kappa to be >=5x smaller than old; "
                                 f"old={kappa_old}, new={kappa_new}, ratio={ratio}")

    def test_legacy_value_is_exposed_only_for_development_comparison(self):
        rng = np.random.default_rng(7)
        trace = make_straight_line(
            length=28.0, step=1.0, angle=0.37,
            noise_sigma=0.4, rng=rng,
        )
        _dvec, geometric, extra = _fit_local_tangent_and_curvature(trace, 12)
        expected_legacy = old_formula_curvature(trace, 12)
        self.assertAlmostEqual(
            extra["legacy_mixed_curvature"], expected_legacy, places=7
        )
        self.assertNotAlmostEqual(geometric, expected_legacy, places=6)


if __name__ == "__main__":
    unittest.main()
