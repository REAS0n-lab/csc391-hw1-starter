"""Tests for the cost model you write in src/costs.py.

These check consistency and scaling, not the exact operation count. The
derivation is yours, and two defensible derivations can differ by a small
constant. What they cannot do is disagree about how the cost grows.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.costs import effective_passes, rcd_update_cost, sgd_update_cost  # noqa: E402

N, D = 10_000, 500


def test_effective_passes_is_defined_by_one_matvec():
    # One pass over a dense n by d matrix is 2*n*d flops.
    assert effective_passes(2.0 * N * D, N, D) == pytest.approx(1.0)


def test_sgd_cost_returns_two_numbers():
    flops, byts = sgd_update_cost(N, D, batch_size=32)
    assert flops > 0 and byts > 0


def test_sgd_cost_is_linear_in_batch_size():
    f1, b1 = sgd_update_cost(N, D, batch_size=1)
    f32, b32 = sgd_update_cost(N, D, batch_size=32)
    # The batch-dependent term dominates once the batch is large. Doubling
    # the batch must not more than double the work.
    assert f32 < 32 * f1 + 1e-9
    assert f32 > f1


def test_sgd_cost_is_linear_in_d():
    f_small, _ = sgd_update_cost(N, D, batch_size=8)
    f_big, _ = sgd_update_cost(N, 2 * D, batch_size=8)
    assert f_big == pytest.approx(2 * f_small, rel=0.25)


def test_sgd_cost_does_not_depend_on_n():
    # A mini-batch update touches the batch, not the full data set.
    assert sgd_update_cost(N, D, 8)[0] == pytest.approx(sgd_update_cost(4 * N, D, 8)[0])


def test_rcd_cost_returns_two_numbers():
    flops, byts = rcd_update_cost(N, D)
    assert flops > 0 and byts > 0


def test_rcd_cost_is_linear_in_n_with_a_maintained_residual():
    f_small, _ = rcd_update_cost(N, D, maintain_residual=True)
    f_big, _ = rcd_update_cost(2 * N, D, maintain_residual=True)
    assert f_big == pytest.approx(2 * f_small, rel=0.25)


def test_rcd_cost_does_not_depend_on_d_with_a_maintained_residual():
    a, _ = rcd_update_cost(N, D, maintain_residual=True)
    b, _ = rcd_update_cost(N, 4 * D, maintain_residual=True)
    assert a == pytest.approx(b, rel=0.05)


def test_recomputing_the_residual_is_more_expensive():
    kept, _ = rcd_update_cost(N, D, maintain_residual=True)
    recomputed, _ = rcd_update_cost(N, D, maintain_residual=False)
    assert recomputed > 10 * kept


def test_a_full_pass_of_rcd_costs_about_one_matvec_pair():
    # d coordinate updates touch every entry of X once, so the work should be
    # within a small factor of a matrix-vector product and its transpose.
    per_update, _ = rcd_update_cost(N, D, maintain_residual=True)
    assert 0.25 <= (D * per_update) / (2.0 * N * D) <= 4.0
