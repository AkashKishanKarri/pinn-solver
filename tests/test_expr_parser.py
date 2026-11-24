import numpy as np
import pytest
import sympy as sp

from pinnsolver.expr import ParseError, evaluate, parse_expression
from pinnsolver.parser import ValidationError, parse_problem
from pinnsolver.spec import ODE_BVP, ODE_IVP, PDE_PARABOLIC, ProblemInput

x = sp.Symbol("x")


def res(name, pos):
    if name == "x":
        return x
    raise ParseError(f"unknown {name}")


@pytest.mark.parametrize("text,expected", [
    ("2x + 1", 2 * x + 1), ("-x^2", -x**2), ("2^3^2", sp.Integer(512)), ("x**2/4", x**2 / 4),
    ("sin(pi*x)", sp.sin(sp.pi * x)), ("exp(-x)*cos(x)", sp.exp(-x) * sp.cos(x)), ("1.5e-3*x", sp.Float(1.5e-3) * x),
    ("sqrt(x) + abs(x)", sp.sqrt(x) + sp.Abs(x)), ("0.1", sp.Rational(1, 10)),
])
def test_grammar(text, expected):
    assert sp.simplify(parse_expression(text, res) - expected) == 0


@pytest.mark.parametrize("text", [
    "__import__('os')", "x.__class__", "lambda: 1", "eval(x)", "open(x)", "x; 1", "x[0]", "'a'", "x y", "sin x",
    "sin(x, x)", "1/0", "", "x" * 500, "((((((((((((((((((((((((((((((((((((((((((x))))))))))))))))))))))))))))))))))))))))))",
])
def test_rejects_unsafe_or_invalid(text):
    with pytest.raises(ParseError):
        parse_expression(text, res)


def test_evaluate_numpy_and_torch_agree():
    import torch
    e = parse_expression("exp(-x)*sin(2*x) + x^2/3 + sinh(1) + sqrt(x+1)", res)
    xs = np.linspace(0, 2, 11)
    a = evaluate(e, {x: xs})
    b = evaluate(e, {x: torch.tensor(xs, dtype=torch.float64)}, backend="torch").numpy()
    ref = np.exp(-xs) * np.sin(2 * xs) + xs**2 / 3 + np.sinh(1) + np.sqrt(xs + 1)
    assert np.allclose(a, ref, atol=1e-14) and np.allclose(b, ref, atol=1e-14)


def P(**kw):
    kw.setdefault("variables", ["t"])
    kw.setdefault("domain", {"t": (0, 1)})
    return ProblemInput(**kw)


def test_classifies_supported_classes():
    assert parse_problem(P(equation="u' = -2*u", conditions=["u(0)=1"])).kind == ODE_IVP
    s = parse_problem(P(equation="u'' + 4*u = 0", conditions=["u(0)=1", "u'(0)=0"], domain={"t": (0, 10)}))
    assert s.kind == ODE_IVP and s.order == 2 and s.features["oscillatory"]
    s = parse_problem(P(equation="-u'' = 1", variables=["x"], domain={"x": (0, 1)}, conditions=["u(0)=0", "u(1)=0"]))
    assert s.kind == ODE_BVP and "Poisson" in s.subtype
    s = parse_problem(ProblemInput(equation="u_t + u*u_x = nu*u_xx", variables=["x", "t"], domain={"x": (-1, 1), "t": (0, 1)},
                                   conditions=["u(x,0)=-sin(pi*x)", "u(-1,t)=0", "u(1,t)=0"], parameters={"nu": 0.01 / np.pi}))
    assert s.kind == PDE_PARABOLIC and s.features["steep_gradient"] and not s.linear and "Burgers" in s.subtype


@pytest.mark.parametrize("kw,msg", [
    (dict(equation="u' = y", conditions=["u(0)=1"]), "Unknown symbol"),
    (dict(equation="u''' = u", conditions=["u(0)=1"]), "order 3"),
    (dict(equation="u'^2 = u", conditions=["u(0)=1"]), "linear in its highest derivative"),
    (dict(equation="t*u' = u", conditions=["u(0)=1"]), "vanishes"),
    (dict(equation="u'' = u", conditions=["u(0)=1"]), "second-order ODE needs"),
    (dict(equation="u' = u", conditions=["u(1)=1"]), "start of the interval"),
    (dict(equation="u' = u", conditions=["u(0.5)=1"]), "not an end point"),
    (dict(equation="u'' = u", conditions=["u'(0)=1", "u'(1)=0"]), "Neumann"),
    (dict(equation="u' = u", conditions=["u(0)=1", "u(0)=2"]), "duplicate or conflicting"),
    (dict(equation="u = 1", conditions=["u(0)=1"]), "no derivative"),
    (dict(equation="u' = u", conditions=[]), "No initial"),
    (dict(equation="u' = u", conditions=["u(0)=1"], domain={"t": (1, 0)}), "lower < upper"),
    (dict(equation="u' = k*u", conditions=["u(0)=1"], parameters={"sin": 1.0}), "clashes"),
])
def test_rejections_have_reasons(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        parse_problem(P(**kw))


def test_pde_rejections():
    base = dict(variables=["x", "t"], domain={"x": (0, 1), "t": (0, 1)}, conditions=["u(x,0)=0", "u(0,t)=0", "u(1,t)=0"])
    with pytest.raises(ValidationError, match="positive"):
        parse_problem(ProblemInput(equation="u_t = -u_xx", **base))
    with pytest.raises(ValidationError, match="diffusion term"):
        parse_problem(ProblemInput(equation="u_t = u_x", **base))
    with pytest.raises(ValidationError, match="first-order time"):
        parse_problem(ProblemInput(equation="u_tt = u_xx", **base))
    with pytest.raises(ValidationError, match="exactly three"):
        parse_problem(ProblemInput(equation="u_t = u_xx", **{**base, "conditions": ["u(x,0)=0", "u(0,t)=0"]}))


def test_exact_solution_is_verified():
    good = parse_problem(P(equation="u' = -2*u", conditions=["u(0)=1"], exact="exp(-2*t)"))
    assert good.exact is not None
    bad = parse_problem(P(equation="u' = -2*u", conditions=["u(0)=1"], exact="exp(-3*t)"))
    assert bad.exact is None and any("NOT used" in w for w in bad.warnings)
    wrong_ic = parse_problem(P(equation="u' = -2*u", conditions=["u(0)=2"], exact="exp(-2*t)"))
    assert wrong_ic.exact is None


def test_corner_incompatibility_warns_and_disables_hard_constraints():
    from pinnsolver.pinn import hard_constraint_plan
    s = parse_problem(ProblemInput(equation="u_t = u_xx", variables=["x", "t"], domain={"x": (0, 1), "t": (0, 1)},
                                   conditions=["u(x,0)=1", "u(0,t)=0", "u(1,t)=0"]))
    assert not s.features["corners_compatible"] and s.warnings
    assert hard_constraint_plan(s)[0] is False
