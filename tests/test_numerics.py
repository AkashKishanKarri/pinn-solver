import numpy as np
import pytest
import torch

from pinnsolver.benchmarks import BENCHMARKS, get
from pinnsolver.parser import parse_problem
from pinnsolver.pinn import PINN, condition_values, residual, to_tensor
from pinnsolver.references import (AnalyticalReference, FiniteDifferenceReference, ScipyBVPReference,
                                   ScipyIVPReference)
from pinnsolver.spec import ProblemInput
from pinnsolver.validation import Validator


def spec_of(bid):
    return parse_problem(ProblemInput.from_dict(get(bid)["problem"]))


def test_all_benchmark_exact_solutions_verify():
    for b in BENCHMARKS:
        s = spec_of(b["id"])
        assert (s.exact is not None) == (b["problem"]["exact"] is not None), b["id"]


@pytest.mark.parametrize("bid,cls,tol", [
    ("decay", ScipyIVPReference, 1e-8), ("logistic", ScipyIVPReference, 1e-8), ("oscillator", ScipyIVPReference, 1e-8),
    ("stiff", ScipyIVPReference, 1e-7), ("bvp", ScipyBVPReference, 1e-8), ("bvp_neumann", ScipyBVPReference, 1e-8),
    ("poisson", ScipyBVPReference, 1e-6), ("heat", FiniteDifferenceReference, 1e-5),
])
def test_numerical_references_match_exact(bid, cls, tol):
    s = spec_of(bid)
    num, ex = cls().solve(s), AnalyticalReference().solve(s)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(lo, hi, 2000) for lo, hi in s.bounds])
    err = np.max(np.abs(num(X) - ex(X)))
    assert err < tol
    assert num.estimated_error is not None and num.estimated_error < 100 * tol


def test_burgers_reference_matches_cole_hopf():
    s = spec_of("burgers")
    ref = FiniteDifferenceReference().solve(s)
    nu = 0.01 / np.pi
    z, w = np.polynomial.hermite.hermgauss(200)

    def cole_hopf(x, t):
        y = x - np.sqrt(4 * nu * t) * z
        e = -np.cos(np.pi * y) / (2 * np.pi * nu)
        f = np.exp(e - e.max())
        return np.sum(w * (-np.sin(np.pi * y)) * f) / np.sum(w * f)

    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(0.05, 1, 200)])
    ex = np.array([cole_hopf(a, b) for a, b in X])
    assert np.linalg.norm(ref(X) - ex) / np.linalg.norm(ex) < 1e-3
    assert np.max(np.abs(ref(X) - ex)) < 5e-3


@pytest.mark.parametrize("bid", ["decay", "oscillator", "bvp", "heat", "burgers"])
def test_hard_constraints_exact_for_random_weights(bid):
    s = spec_of(bid)
    torch.manual_seed(3)
    m = PINN(s, hidden=[16, 16], hard_constraints=True)
    assert m.hard
    v = Validator(s, {"rel_l2": 1, "condition_max": 1, "residual_rms": 1}, 64, seed=0)
    for c, X in v.cond_points:
        val, target = condition_values(m, s, c, to_tensor(X, requires_grad=True), create_graph=False)
        assert torch.max(torch.abs(val - target)) < 1e-12, c.text


def test_neumann_bvp_uses_penalty():
    s = spec_of("bvp_neumann")
    assert not PINN(s, hidden=[8], hard_constraints=True).hard


class ExactModel(torch.nn.Module):
    """A 'network' that returns the exact solution, to test the residual operator."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        return self.f(X)


@pytest.mark.parametrize("bid,f", [
    ("decay", lambda X: torch.exp(-2 * X[:, 0])),
    ("oscillator", lambda X: torch.cos(2 * X[:, 0])),
    ("poisson", lambda X: torch.sin(np.pi * X[:, 0]) + 0.1 * torch.sin(8 * np.pi * X[:, 0])),
    ("heat", lambda X: torch.exp(-0.1 * np.pi**2 * X[:, 1]) * torch.sin(np.pi * X[:, 0])),
    ("logistic", lambda X: 1 / (1 + 9 * torch.exp(-X[:, 0]))),
])
def test_autograd_residual_vanishes_on_exact_solution(bid, f):
    s = spec_of(bid)
    rng = np.random.default_rng(0)
    X = to_tensor(np.column_stack([rng.uniform(lo, hi, 500) for lo, hi in s.bounds]), requires_grad=True)
    r = residual(ExactModel(f), s, X, create_graph=False)
    assert torch.max(torch.abs(r)) < 1e-9


def test_validator_metrics_and_independence():
    s = spec_of("decay")
    ex = AnalyticalReference().solve(s)
    v = Validator(s, {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3}, 500, seed=0)
    perfect = v.check(ExactModel(lambda X: torch.exp(-2 * X[:, 0])), ex)
    assert perfect.passed and perfect.metrics["rel_l2"] < 1e-14 and perfect.metrics["residual_rms"] < 1e-12
    off = v.check(ExactModel(lambda X: 1.01 * torch.exp(-2 * X[:, 0])), ex)
    assert not off.passed and abs(off.metrics["rel_l2"] - 0.01) < 1e-9
    assert any("relative L2" in f for f in off.failures) and any("condition" in f for f in off.failures)
    # validation points are not the Sobol training points
    from pinnsolver.training import sobol_points
    train = sobol_points(s.bounds, 512, seed=0)
    assert not np.any(np.isin(np.round(v.points, 12), np.round(train, 12)))


def test_residual_only_criterion_when_no_reference():
    s = spec_of("decay")
    v = Validator(s, {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3}, 200, seed=0)
    out = v.check(ExactModel(lambda X: torch.exp(-2 * X[:, 0])), None)
    assert out.passed and out.criterion == "residual-only"
    assert any("NOT established" in n for n in out.notes)
