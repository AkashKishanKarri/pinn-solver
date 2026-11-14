"""Natural notation, early rejection of ill-posed problems, and solution-scale normalisation."""
import numpy as np
import pytest
import sympy as sp
import torch

from pinnsolver import benchmarks
from pinnsolver.parser import ValidationError, parse_problem
from pinnsolver.pinn import PINN, output_scale
from pinnsolver.references import ReferenceSolution
from pinnsolver.solver import solve
from pinnsolver.spec import ODE_IVP, PDE_PARABOLIC, ProblemInput
from pinnsolver.validation import Validator


def P(**kw):
    base = dict(equation="u' = -2*u", variables=["t"], domain={"t": (0, 1)}, conditions=["u(0)=1"])
    base.update(kw)
    return ProblemInput(**base)


CANONICAL = parse_problem(P()).residual


@pytest.mark.parametrize("equation", [
    "du/dt = -2*u", "du/dt = -2u", "u'(t) = -2*u(t)", "u' = -2 u", "u' + 2*u", "u′ = −2·u", "u_t = -2*u",
])
def test_alternative_notations_give_the_same_equation(equation):
    assert sp.simplify(parse_problem(P(equation=equation)).residual - CANONICAL) == 0


def test_missing_equals_sign_is_reported():
    assert any("no '=' sign" in w for w in parse_problem(P(equation="u' + 2*u")).warnings)


def test_parameters_constants_and_second_order_leibniz():
    s = parse_problem(P(equation="u' = -k u", parameters={"k": 2.0}))
    assert sp.simplify(s.residual - CANONICAL) == 0
    s = parse_problem(P(equation="u' = e^(-t)", exact="u = 2 - e^(-t)"))
    assert s.exact is not None
    s = parse_problem(P(equation="d^2u/dt^2 + 4u = 0", conditions=["u(0)=1", "du/dt(0)=0"], domain={"t": (0, 3)}))
    assert s.kind == ODE_IVP and s.order == 2
    s = parse_problem(ProblemInput(equation="∂u/∂t = 0.1 ∂²u/∂x²", variables=["x", "t"],
                                   domain={"x": (0, 1), "t": (0, 1)},
                                   conditions=["u(x,0)=sin(π x)", "u(0,t)=0", "u(1,t)=0"]))
    assert s.kind == PDE_PARABOLIC and "heat" in s.subtype


def test_other_unknown_name_and_hints():
    assert parse_problem(P(equation="y' = -2y", conditions=["y(0)=1"], dependent="y")).dependent == "y"
    with pytest.raises(ValidationError, match="set the unknown function to 'y'"):
        parse_problem(P(equation="y' = -2*y"))
    with pytest.raises(ValidationError, match=r"Did you mean k\*u"):
        parse_problem(P(equation="u' = -ku", parameters={"k": 2.0}))
    with pytest.raises(ValidationError, match="reserved"):
        parse_problem(P(equation="u' = -u", parameters={"e": 2.0}))


@pytest.mark.parametrize("equation,reason", [
    ("u' = 1/(t-0.5)", "division by zero near t = 0.5"),
    ("u' = u/(t - 1/3)", "division by zero"),
    ("u' = log(t - 0.5)", "logarithm"),
    ("u' = sqrt(t - 0.5)*u", "root"),
])
def test_equation_undefined_inside_the_domain_is_rejected(equation, reason):
    with pytest.raises(ValidationError, match=reason):
        parse_problem(P(equation=equation))


def test_blow_up_is_reported_before_training():
    with pytest.raises(ValidationError, match="cannot be continued past t = 1"):
        solve(P(equation="u' = u^2", domain={"t": (0, 2)}), mode="D", time_budget=5, save=False)


def test_bvp_without_unique_solution_is_rejected():
    kw = dict(equation="u'' = -u", variables=["x"])
    with pytest.raises(ValidationError, match="no unique solution"):
        parse_problem(P(domain={"x": (0, np.pi)}, conditions=["u(0)=0", "u(pi)=0"], **kw))
    with pytest.raises(ValidationError, match="no unique solution"):
        parse_problem(P(domain={"x": (0, np.pi / 2)}, conditions=["u(0)=0", "u'(pi/2)=0"], **kw))
    near = parse_problem(P(domain={"x": (0, 3.14159)}, conditions=["u(0)=0", "u(3.14159)=0"], **kw))
    assert any("very close" in w for w in near.warnings)
    fine = parse_problem(P(domain={"x": (0, 1)}, conditions=["u(0)=0", "u(1)=1"], **kw))
    assert not fine.warnings and fine.features["bvp_uniqueness_margin"] > 0.1


def test_zero_reference_uses_absolute_error():
    s = parse_problem(P(equation="u'' = -u", variables=["x"], domain={"x": (0, 1)}, conditions=["u(0)=0", "u(1)=0"]))
    v = Validator(s, {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3}, 100, seed=0)
    zero = ReferenceSolution("zero", "analytical", {}, lambda X: np.zeros(X.shape[0]), 0.0, 0.0)
    out = v.check_values(np.full(v.grid.shape[0], 1e-6), zero)
    assert out.passed and abs(out.metrics["rel_l2"] - 1e-6) < 1e-12
    assert any("zero everywhere" in n for n in out.notes)
    assert not v.check_values(np.full(v.grid.shape[0], 0.5), zero).passed


def test_output_scale_only_for_badly_scaled_linear_data():
    scale = lambda **kw: output_scale(parse_problem(P(**kw)))
    assert scale(conditions=["u(0)=0.001"]) == 0.001
    assert scale(conditions=["u(0)=1000"]) == 1000
    assert scale(conditions=["u(0)=1"]) == 1.0 and scale(conditions=["u(0)=0.1"]) == 1.0
    assert scale(equation="u' = cos(t)", conditions=["u(0)=0.001"]) == 1.0  # source term dominates the data
    assert scale(equation="u' = u*(1-u)", conditions=["u(0)=0.001"]) == 1.0  # nonlinear: no scaling symmetry
    # every benchmark is O(1): untouched, so the published benchmark results stay valid
    for b in benchmarks.BENCHMARKS:
        assert output_scale(parse_problem(ProblemInput.from_dict(b["problem"]))) == 1.0, b["id"]


def test_scaled_problem_trains_exactly_like_the_unit_problem():
    torch.manual_seed(0)
    unit = PINN(parse_problem(P()), hidden=[8, 8])
    torch.manual_seed(0)
    small = PINN(parse_problem(P(conditions=["u(0)=0.001"])), hidden=[8, 8])
    X = torch.linspace(0, 1, 7, dtype=torch.float64)[:, None]
    assert torch.allclose(small(X), 0.001 * unit(X), rtol=1e-12, atol=0)


def test_small_amplitude_problem_now_passes():
    r = solve(P(conditions=["u(0)=0.001"], domain={"t": (0, 2)}), mode="D", seed=0, time_budget=20, save=False)
    assert r["status"] == "passed" and r["final_method"] == "pinn"
    assert r["validation"]["metrics"]["rel_l2"] < 1e-3
