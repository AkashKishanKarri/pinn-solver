"""Edge cases: hostile or degenerate input must be rejected quickly and clearly, never crash, hang or pass falsely."""
import time

import numpy as np
import pytest

from pinnsolver.expr import ParseError, const_value, expanded_terms, parse_expression
from pinnsolver.parser import ValidationError, parse_problem
from pinnsolver.references import ReferenceSolution
from pinnsolver.solver import solve
from pinnsolver.spec import PDE_PARABOLIC, ProblemInput
from pinnsolver.validation import Validator

HEAT_CONDS = ["u(x,0)=sin(pi*x)", "u(0,t)=0", "u(1,t)=0"]


def P(**kw):
    base = dict(equation="u' = -2*u", variables=["t"], domain={"t": (0, 1)}, conditions=["u(0)=1"])
    base.update(kw)
    return ProblemInput(**base)


def PDE(**kw):
    base = dict(equation="u_t = u_xx", variables=["x", "t"], domain={"x": (0, 1), "t": (0, 1)}, conditions=list(HEAT_CONDS))
    base.update(kw)
    return ProblemInput(**base)


# ----------------------------------------------------------------------
# expressions that used to hang the parser (exact big-number arithmetic / huge expansions)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("equation", [
    "u' = 9^9^9*u", "u' = 2^2^2^2^2^2*u", "u' = 99999999^99999999*u", "u' = 1.5^99999999*u",
    "u' = 0.1^(-99999999)*u", "u' = exp(exp(exp(100)))*u", "u' = exp(1000)*u", "u' = (1+u+t)^200",
])
def test_huge_constants_and_expansions_are_rejected_quickly(equation):
    tic = time.perf_counter()
    with pytest.raises(ValidationError):
        parse_problem(P(equation=equation))
    assert time.perf_counter() - tic < 2.0


def test_ordinary_powers_still_work():
    assert parse_problem(P(equation="u' = -u^3 + 2^10*t + e^(-t) + t^0.5")).linear is False
    assert const_value(parse_expression("2^0.5 * 10^-3 + sinh(1)", lambda n, p: None)) == pytest.approx(2**0.5 * 1e-3 + np.sinh(1))
    assert expanded_terms(parse_problem(P(equation="u' = (1+u)^3")).residual) <= 5


# ----------------------------------------------------------------------
# constants that are not real numbers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(equation="u' = sqrt(-1)*u"),
    dict(equation="u' = log(-1)*u"),
    dict(equation="u' = (-8)^(1/3)*u"),
    dict(conditions=["u(0) = sqrt(-4)"]),
    dict(equation="u' = k^0.5*u", parameters={"k": -1.0}),  # complex only once k is substituted
    dict(equation="u' = u/(k-1)", parameters={"k": 1.0}),  # division by zero only once k is substituted
])
def test_complex_or_undefined_constants_are_rejected(kw):
    with pytest.raises(ValidationError):
        parse_problem(P(**kw))


def test_ode_condition_at_a_variable_is_rejected_not_a_crash():
    with pytest.raises(ValidationError, match="number"):
        parse_problem(P(conditions=["u(t) = 1"]))


# ----------------------------------------------------------------------
# abs() and equations that are undefined for some values of u
# ----------------------------------------------------------------------
def test_abs_of_the_unknown_is_supported_everywhere():
    assert parse_problem(P(equation="u' = -abs(u)", exact="exp(-t)")).exact is not None
    spec = parse_problem(PDE(equation="u_t = u_xx - abs(u)"))
    assert spec.kind == PDE_PARABOLIC and not spec.linear
    r = solve(P(equation="u' = -abs(u)", exact="exp(-t)"), mode="E", save=False)
    assert r["status"] == "passed"


def test_pde_undefined_at_u_zero_is_accepted_with_a_warning():
    spec = parse_problem(PDE(equation="u_t = u_xx - 1/u", conditions=["u(x,0)=1", "u(0,t)=1", "u(1,t)=1"]))
    assert any("undefined for some values" in w for w in spec.warnings)


# ----------------------------------------------------------------------
# exact solutions
# ----------------------------------------------------------------------
def test_exact_solution_that_is_nan_at_an_end_point_is_not_used():
    # sin(t)/tan(t) equals cos(t) but is 0/0 at t = 0, where the initial condition sits
    spec = parse_problem(P(equation="u' = -sin(t)", exact="sin(t)/tan(t)"))
    assert spec.exact is None and spec.features["exact_solution_verified"] is False


def test_a_non_finite_reference_can_never_pass():
    spec = parse_problem(P())
    v = Validator(spec, {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3}, 50, 0)
    nan_ref = ReferenceSolution("nan", "analytical", {}, lambda X: np.full(X.shape[0], np.nan), 0.0, 0.0)
    res = v.check_values(np.exp(-2 * v.grid[:, 0]), nan_ref)
    assert not res.passed and res.failures


# ----------------------------------------------------------------------
# problems whose solution does not exist on the whole domain: fast, clear rejection (never a hang)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kw, text", [
    (dict(equation="u' = sqrt(u)", conditions=["u(0) = -1"]), "initial condition"),
    (dict(equation="u' = 1/u", conditions=["u(0) = 0"]), "initial condition"),
    (dict(equation="u' = log(u)", conditions=["u(0) = 0"]), "initial condition"),
    (dict(equation="u' = tan(t)", conditions=["u(0) = 0"], domain={"t": (0, 2)}), "cannot be continued"),
    (dict(equation="u' = -1/u", conditions=["u(0) = 1"]), "cannot be continued"),
    (dict(equation="u' = 1000*u", conditions=["u(0) = 1"], domain={"t": (0, 10)}), "cannot be continued"),
])
def test_solutions_that_leave_the_domain_of_the_equation_are_rejected(kw, text):
    tic = time.perf_counter()
    with pytest.raises(ValidationError, match=text):
        solve(P(**kw), mode="E", save=False)
    assert time.perf_counter() - tic < 20.0


def test_failed_pde_reference_is_reported_not_rejected():
    # the finite-difference grid cannot resolve this front; that is not evidence that the PDE has no solution
    r = solve(PDE(equation="u_t + u*u_x = 0.0001*u_xx", domain={"x": (-1, 1), "t": (0, 1)},
                  conditions=["u(x,0)=-sin(pi*x)", "u(-1,t)=0", "u(1,t)=0"]), mode="E", save=False)
    assert r["status"] == "failed" and r["final_method"] == "none"
    assert any("No reference solution" in w for w in r["warnings"])


def test_unreliable_numerical_solution_is_flagged():
    # u = 1/(1 - t) is singular exactly at the end of the domain
    r = solve(P(equation="u' = u^2", conditions=["u(0) = 1"]), mode="E", save=False)
    assert any("own error estimate is large" in w for w in r["warnings"])


# ----------------------------------------------------------------------
# malformed input files
# ----------------------------------------------------------------------
@pytest.mark.parametrize("d", [
    None, [], {"equation": "u'=-u"},
    {"equation": "u'=-u", "variables": ["t"], "domain": None, "conditions": ["u(0)=1"]},
    {"equation": "u'=-u", "variables": ["t"], "domain": {"t": [0, 1, 2]}, "conditions": ["u(0)=1"]},
    {"equation": "u'=-u", "variables": ["t"], "domain": {"t": "01"}, "conditions": ["u(0)=1"]},
    {"equation": "u'=-u", "variables": ["t"], "domain": {"t": ["a", "b"]}, "conditions": ["u(0)=1"]},
])
def test_malformed_problem_dicts_raise_value_error(d):
    with pytest.raises(ValueError, match="Malformed"):
        ProblemInput.from_dict(d)


def test_single_condition_or_variable_may_be_a_plain_string():
    p = ProblemInput.from_dict({"equation": "u'=-u", "variables": "t", "domain": {"t": [0, 1]}, "conditions": "u(0)=1"})
    assert parse_problem(p).conditions[0].text == "u(0)=1"


# ----------------------------------------------------------------------
# degenerate but valid problems run to a sensible result in every mode
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["A", "B", "C", "D"])
def test_tiny_budget_never_crashes(mode):
    r = solve(PDE(), mode=mode, time_budget=1.0, save=False)
    assert r["status"] in ("passed", "failed", "fallback")
    assert r["timings"]["training_s"] < 15.0


def test_zero_solution_passes_on_absolute_error():
    r = solve(P(equation="u' = -u", conditions=["u(0) = 0"], exact="0"), mode="D", time_budget=10, save=False)
    assert r["status"] == "passed" and r["final_method"] == "pinn"
