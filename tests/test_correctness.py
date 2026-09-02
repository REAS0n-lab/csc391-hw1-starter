"""Correctness tests for the HW1 starter.

These check that the objective, gradients, and both methods do what they
claim. They check nothing about performance, and a passing test suite is not
evidence that a comparison is fair.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import objective as obj, problems  # noqa: E402
from src.harness import run  # noqa: E402
from src.rcd import RCD  # noqa: E402
from src.sgd import SGD  # noqa: E402


@pytest.fixture(scope="module")
def small():
    return problems.synthetic(n=200, d=20, condition=10.0, seed=0, lam=1e-2)


def test_objective_matches_definition(small):
    w = np.random.default_rng(0).standard_normal(small.d)
    r = small.X @ w - small.y
    expected = 0.5 * float(r @ r) / small.n + 0.5 * small.lam * float(w @ w)
    assert obj.objective(small, w) == pytest.approx(expected, rel=1e-12)


def test_gradient_matches_finite_differences(small):
    rng = np.random.default_rng(1)
    w = rng.standard_normal(small.d)
    g = obj.gradient(small, w)
    h = 1e-6
    for j in rng.choice(small.d, size=5, replace=False):
        e = np.zeros(small.d)
        e[j] = h
        fd = (obj.objective(small, w + e) - obj.objective(small, w - e)) / (2 * h)
        assert g[j] == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_coordinate_derivative_matches_gradient(small):
    w = np.random.default_rng(2).standard_normal(small.d)
    g = obj.gradient(small, w)
    for j in (0, 3, 7):
        assert obj.coordinate_derivative(small, w, j) == pytest.approx(g[j], rel=1e-10)


def test_exact_solution_has_zero_gradient(small):
    w_star = obj.solve_exact(small)
    assert np.linalg.norm(obj.gradient(small, w_star)) < 1e-10


def test_exact_solution_is_a_lower_bound(small):
    f_star = obj.objective(small, obj.solve_exact(small))
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = rng.standard_normal(small.d)
        assert obj.objective(small, w) >= f_star - 1e-12


def test_coordinate_lipschitz_is_the_second_derivative(small):
    Lj = obj.coordinate_lipschitz(small)
    w = np.zeros(small.d)
    h = 1e-4
    for j in (0, 5, 11):
        e = np.zeros(small.d)
        e[j] = h
        second = (obj.objective(small, w + e) - 2 * obj.objective(small, w)
                  + obj.objective(small, w - e)) / h ** 2
        assert Lj[j] == pytest.approx(second, rel=1e-4)


def test_rcd_residual_stays_consistent(small):
    w = np.zeros(small.d)
    m = RCD(small, seed=0)
    m.reset(w)
    for _ in range(200):
        m.step(w)
    assert np.allclose(m.residual, small.X @ w - small.y, atol=1e-9)


def test_rcd_reaches_the_exact_solution(small):
    r = run(RCD(small, seed=0), small, max_epochs=400, checkpoints_per_epoch=1)
    assert r.checkpoints[-1].gap < 1e-10


def test_sgd_decreases_the_objective(small):
    r = run(SGD(small, seed=0, batch_size=8), small, max_epochs=40,
            checkpoints_per_epoch=1)
    assert r.checkpoints[-1].gap < r.checkpoints[0].gap / 10


def test_harness_records_four_progress_axes(small):
    r = run(RCD(small, seed=0), small, max_epochs=2, checkpoints_per_epoch=2)
    c = r.checkpoints[-1]
    for field in ("iteration", "epoch", "seconds", "effective_passes"):
        assert getattr(c, field) is not None


def test_stopping_rule_is_recorded(small):
    r = run(RCD(small, seed=0), small, max_epochs=500, gap_tol=1e-6)
    assert r.stopped_by == "gap_tol"
