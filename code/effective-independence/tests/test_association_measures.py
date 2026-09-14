"""Tests for the pairwise association measures in src/analysis/neff.py.

These exist because the paper's central methodological claim rests on them:
that phi is bounded by the two configs' error rates and therefore overstates
ensemble independence, while Yule's Q is not so bounded. A silent bug in
either function would propagate straight into the reported N_eff numbers, so
the properties are pinned down here rather than spot-checked by hand.

Run: ./.venv/bin/python -m pytest tests/ -q
  (or ./.venv/bin/python tests/test_association_measures.py for a plain run)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.neff import (  # noqa: E402
    max_attainable_phi,
    n_eff,
    pairwise_phi,
    pairwise_phi_ratio,
    pairwise_yules_q,
)

TOL = 1e-9


def test_identical_vectors_are_perfectly_associated():
    a = [1, 0, 1, 0, 1, 0]
    b = [1, 0, 1, 0, 1, 0]
    assert abs(pairwise_phi(a, b) - 1.0) < TOL
    assert abs(pairwise_yules_q(a, b) - 1.0) < TOL
    # Equal marginals means phi is free to reach 1.
    assert abs(max_attainable_phi(a, b) - 1.0) < TOL


def test_phi_is_capped_by_unequal_marginals_but_q_is_not():
    """The paper's core point, as a test.

    A is wrong only where B is wrong (a strict subset), which is total
    dependence: knowing B's error tells you everything available about A's.
    phi still cannot exceed ~0.378 because the error rates differ (1/8 vs
    4/8). Q correctly reports 1.0.
    """
    a = [1, 0, 0, 0, 0, 0, 0, 0]  # p = 1/8
    b = [1, 1, 1, 1, 0, 0, 0, 0]  # q = 4/8
    phi = pairwise_phi(a, b)
    ceiling = max_attainable_phi(a, b)

    assert phi < 0.99, "phi should be held well below 1 by the marginal cap"
    assert abs(phi - ceiling) < TOL, "this pair should sit exactly at its ceiling"
    assert abs(pairwise_phi_ratio(a, b) - 1.0) < TOL
    assert abs(pairwise_yules_q(a, b) - 1.0) < TOL, "Q must not be marginal-capped"


def test_max_attainable_phi_matches_closed_form():
    a = [1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 0, 0, 0]
    p, q = 0.125, 0.5
    expected = math.sqrt((p * (1 - q)) / (q * (1 - p)))
    assert abs(max_attainable_phi(a, b) - expected) < TOL


def test_no_haldane_correction_is_applied():
    """Regression test for a real bug.

    An earlier version applied the Haldane-Anscombe +0.5 correction whenever a
    cell was zero. That is standard for raw odds ratios but wrong for Q: it
    shrank Q away from 1 on exactly the empty-off-diagonal tables where Q's
    marginal-robustness matters, which would have understated the correction
    the paper reports.
    """
    a = [1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 0, 0, 0]
    assert abs(pairwise_yules_q(a, b) - 1.0) < TOL


def test_independence_gives_zero_association():
    a = [1, 1, 0, 0] * 5
    b = [1, 0, 1, 0] * 5
    assert abs(pairwise_phi(a, b)) < TOL
    assert abs(pairwise_yules_q(a, b)) < TOL


def test_anticorrelation_is_negative():
    a = [1, 1, 0, 0]
    b = [0, 0, 1, 1]
    assert pairwise_phi(a, b) < 0
    assert abs(pairwise_yules_q(a, b) + 1.0) < TOL


def test_constant_vectors_are_undefined_not_zero():
    """Zero variance means no association is defined. Returning 0.0 here would
    silently claim independence, which is the 'fail loud on data quality' rule
    this project applies elsewhere."""
    constant = [1, 1, 1, 1]
    varying = [1, 0, 1, 0]
    assert pairwise_phi(constant, varying) is None
    assert pairwise_yules_q(constant, varying) is None
    assert max_attainable_phi(constant, varying) is None


def test_ungraded_cells_are_excluded_pairwise():
    """None entries (ungraded cells) must be dropped pairwise, not treated as
    correct or wrong."""
    a = [1, None, 1, 0, 0]
    b = [1, 0, 1, 0, 0]
    # Overlap is positions 0,2,3,4 -> identical there.
    assert abs(pairwise_phi(a, b) - 1.0) < TOL


def test_insufficient_overlap_returns_none():
    assert pairwise_phi([1, None], [None, 0]) is None
    assert pairwise_yules_q([1, None], [None, 0]) is None


def test_n_eff_endpoints():
    assert abs(n_eff(3, 0.0) - 3.0) < TOL, "uncorrelated members = N independent voters"
    assert abs(n_eff(3, 1.0) - 1.0) < TOL, "perfectly correlated = 1 effective voter"
    # Negative correlation is floored at N rather than allowed to blow up.
    assert abs(n_eff(3, -0.6) - 3.0) < TOL


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
