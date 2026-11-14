"""Parser and validator (M2): ProblemInput -> EquationSpec.

Supported first-version problem classes (anything else is rejected with an
explanation, never "solved" silently):

* ``ode_ivp``  - scalar ODE of order 1 or 2, explicit in its highest
  derivative, with all conditions u(a), u'(a) at the start of the interval.
* ``ode_bvp``  - scalar second-order ODE on [a, b] with one condition at each
  end (Dirichlet ``u(a)=..`` or Neumann ``u'(a)=..``), at least one Dirichlet.
* ``pde_parabolic_1d`` - u_t = F(x, t, u, u_x, u_xx) on [a, b] x [t0, T],
  linear in u_xx with a positive diffusion coefficient, one initial condition
  u(x, t0) = g(x) and Dirichlet conditions u(a, t), u(b, t).

Notation: derivatives are written ``u'``/``u''`` (ODEs) or with subscripts
``u_t``, ``u_x``, ``u_xx`` (any problem). Variable names are single letters;
for PDEs the time variable must be ``t``.

Common alternative notations are rewritten to that form before parsing
(:meth:`Parser._rewrite_notation`): Leibniz derivatives ``du/dt``,
``d^2u/dt^2``, function-call style ``u(t)``/``u'(t)``, typographic characters,
and an equation without ``=`` is read as ``... = 0`` (with a warning).

Problems that cannot have a trustworthy answer are rejected here rather than
after training: a right-hand side that is undefined inside the domain
(division by zero, logarithm/root of a negative number) and linear boundary
value problems without a unique solution.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from .expr import (CONSTANTS, RESERVED, ParseError, check_constants, check_expandable, evaluate, normalize_text,
                   parse_expression)
from .spec import ODE_BVP, ODE_IVP, PDE_PARABOLIC, Condition, EquationSpec, ProblemInput

MAX_ORDER = 2
MAX_CONDITIONS = 6
MAX_ABS_DOMAIN = 1e4
MIN_DOMAIN_WIDTH = 1e-100  # derivatives scale like 1/width^2, which must stay inside float range
MAX_ABS_PARAM = 1e6

# A linear BVP is rejected when the homogeneous solution that satisfies the left condition misses the
# right condition by less than this (relative to its size); below the second value a warning is given.
BVP_SINGULAR_TOL = 1e-6
BVP_ILL_CONDITIONED_TOL = 1e-3
# The network output is rescaled only for clearly badly scaled data; problems whose data are O(1)
# (inside this band) are left exactly as they were.
OUTPUT_SCALE_BAND = (0.1, 10.0)

# Thresholds used to derive the problem features that the rule table (D2) reads.
STIFFNESS_INDEX_LIMIT = 50.0  # max(-Re lambda) * T
OSCILLATION_CYCLES_LIMIT = 2.0  # max |Im lambda| * T / (2 pi)
HIGH_FREQUENCY_CYCLES = 8.0
GROWTH_INDEX_LIMIT = 10.0
PECLET_STEEP_LIMIT = 100.0

_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ValidationError(ValueError):
    """Problem is malformed, inconsistent or outside the supported scope."""


def _fail(msg: str) -> None:
    raise ValidationError(msg)


class Parser:
    """Reads a :class:`ProblemInput` with the restricted grammar and validates it."""

    def __init__(self, problem: ProblemInput):
        self.problem = problem
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # symbol tables
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        p = self.problem
        dep = p.dependent
        if not (isinstance(dep, str) and len(dep) == 1 and dep.isalpha()):
            _fail("The dependent variable must be a single letter, e.g. 'u'.")
        if dep in RESERVED:
            _fail(f"{dep!r} cannot be the unknown function: it is reserved for the constant {dep} = {float(CONSTANTS[dep]):.6g}.")
        if not (1 <= len(p.variables) <= 2):
            _fail("Give one independent variable (ODE) or two, 'x' and 't' style (1D time-dependent PDE).")
        for v in p.variables:
            if not (isinstance(v, str) and len(v) == 1 and v.isalpha()):
                _fail(f"Independent variable {v!r} must be a single letter.")
            if v == dep:
                _fail(f"Independent variable {v!r} clashes with the dependent variable.")
            if v in RESERVED:
                _fail(f"{v!r} cannot be a variable name: it is reserved for the constant {v} = {float(CONSTANTS[v]):.6g}.")
        if len(set(p.variables)) != len(p.variables):
            _fail("Independent variables must be distinct.")
        if len(p.variables) == 2:
            if "t" not in p.variables:
                _fail("For a PDE the time variable must be named 't' (e.g. variables ['x', 't']).")
            space = [v for v in p.variables if v != "t"][0]
            self.variables: Tuple[str, ...] = (space, "t")
        else:
            self.variables = (p.variables[0],)

        if set(p.domain) != set(self.variables):
            _fail(f"Domain must give an interval for exactly these variables: {', '.join(self.variables)}.")
        self.domain: Dict[str, Tuple[float, float]] = {}
        for v in self.variables:
            lo, hi = p.domain[v]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                _fail(f"Domain of {v} must be finite.")
            if max(abs(lo), abs(hi)) > MAX_ABS_DOMAIN:
                _fail(f"Domain of {v} exceeds the allowed range |{v}| <= {MAX_ABS_DOMAIN:g}.")
            if not hi > lo:
                _fail(f"Domain of {v} must satisfy lower < upper (got [{lo}, {hi}]).")
            if hi - lo < MIN_DOMAIN_WIDTH or hi - lo <= 1e-9 * max(abs(lo), abs(hi)):
                _fail(f"Domain of {v} is too narrow to resolve numerically (got [{lo}, {hi}]); rescale the variable.")
            self.domain[v] = (float(lo), float(hi))

        self.params: Dict[str, float] = {}
        self.param_symbols: Dict[str, sp.Symbol] = {}
        for name, value in p.parameters.items():
            if not _IDENT.match(name) or "'" in name:
                _fail(f"Parameter name {name!r} is not a valid identifier.")
            if name in CONSTANTS:
                _fail(f"Parameter name {name!r} is reserved for the constant {name} = {float(CONSTANTS[name]):.6g}; use another name.")
            if name in RESERVED or name in self.variables or name == dep or name.startswith(dep + "_"):
                _fail(f"Parameter name {name!r} clashes with a variable, function or derivative name.")
            if not (isinstance(value, (int, float)) and math.isfinite(value)) or abs(value) > MAX_ABS_PARAM:
                _fail(f"Parameter {name!r} must be a finite number with |value| <= {MAX_ABS_PARAM:g}.")
            self.params[name] = float(value)
            self.param_symbols[name] = sp.Symbol(name, real=True)

        self.coords = tuple(sp.Symbol(v, real=True) for v in self.variables)
        self.coord_by_name = dict(zip(self.variables, self.coords))
        self.dep = dep
        self.deriv_symbols: Dict[Tuple[int, ...], sp.Symbol] = {}
        d = len(self.variables)
        for total in range(MAX_ORDER + 1):
            for key in _multi_indices(d, total):
                self.deriv_symbols[key] = sp.Symbol(_deriv_name(dep, self.variables, key), real=True)

    def _deriv_key(self, name: str, pos: int) -> Optional[Tuple[int, ...]]:
        """Return the derivative multi-index if ``name`` denotes u or a derivative of u."""
        dep, d = self.dep, len(self.variables)
        if name == dep:
            return tuple([0] * d)
        if name.startswith(dep) and set(name[1:]) == {"'"}:
            if d != 1:
                _fail(f"Prime notation {name!r} is only allowed for ODEs; use subscripts such as {dep}_x, {dep}_t.")
            k = len(name) - 1
            if k > MAX_ORDER:
                _fail(f"Derivative {name!r} has order {k}; at most order {MAX_ORDER} is supported.")
            return (k,)
        if name.startswith(dep + "_"):
            sub = name[len(dep) + 1:]
            if not sub:
                _fail(f"Incomplete derivative {name!r} at position {pos + 1}.")
            key = [0] * d
            for ch in sub:
                if ch not in self.variables:
                    _fail(f"In {name!r}: {ch!r} is not an independent variable ({', '.join(self.variables)}).")
                key[self.variables.index(ch)] += 1
            if sum(key) > MAX_ORDER:
                _fail(f"Derivative {name!r} has order {sum(key)}; at most order {MAX_ORDER} is supported.")
            return tuple(key)
        return None

    def _resolver(self, allow_u: bool, allow_coords: Tuple[str, ...], keep_params: bool):
        def resolve(name: str, pos: int) -> sp.Basic:
            key = self._deriv_key(name, pos)
            if key is not None:
                if not allow_u:
                    raise ParseError(f"{name!r} is not allowed here (only constants, parameters and {', '.join(allow_coords) or 'no variables'}).")
                return self.deriv_symbols[key]
            if name in self.coord_by_name:
                if name not in allow_coords:
                    raise ParseError(f"Variable {name!r} is not allowed in this expression.")
                return self.coord_by_name[name]
            if name in self.params:
                return self.param_symbols[name] if keep_params else sp.Float(self.params[name])
            allowed = sorted(set(allow_coords) | set(self.params)) + ([self.dep] if allow_u else [])
            raise ParseError(f"Unknown symbol {name!r} at position {pos + 1}. Allowed names: {', '.join(allowed) or 'none'}."
                             f"{self._hint(name, allowed)}")

        return resolve

    def _hint(self, name: str, allowed: List[str]) -> str:
        """A 'did you mean' suffix for an unknown name."""
        base = name.rstrip("'").split("_")[0]
        if (name.endswith("'") or "_" in name) and len(base) == 1 and base.isalpha() and base != self.dep:
            return (f" The unknown function is currently {self.dep!r}; if yours is {base!r}, "
                    f"set the unknown function to {base!r}.")
        parts = _split_into(name, set(allowed) | set(CONSTANTS))
        if parts:
            return f" Did you mean {'*'.join(parts)}? (Multiplication needs '*' or a space.)"
        return ""

    def _rewrite_notation(self, text: str, strip_calls: bool) -> str:
        """Rewrite common alternative notations to the grammar's own: du/dt -> u_t, u'(t) -> u'."""
        if not isinstance(text, str):
            return text
        text = normalize_text(text).replace("\u2202", "d")  # partial-derivative sign
        dep = re.escape(self.dep)
        var = "|".join(re.escape(v) for v in self.variables)
        pre, post = r"(?<![A-Za-z0-9_])", r"(?![A-Za-z0-9_'])"
        if f"d{self.dep}" not in self.params:
            text = re.sub(rf"{pre}d\s*\^?\s*2\s*{dep}\s*/\s*d\s*({var})\s*\^?\s*2{post}",
                          lambda m: f"{self.dep}_{m.group(1) * 2}", text)
            text = re.sub(rf"{pre}d\s*{dep}\s*/\s*d\s*({var}){post}", lambda m: f"{self.dep}_{m.group(1)}", text)
        if strip_calls:
            args = r"\s*,\s*".join(re.escape(v) for v in self.variables)
            text = re.sub(rf"{pre}({dep}(?:'+|_[A-Za-z]+)?)\s*\(\s*{args}\s*\)", r"\1", text)
        return text

    def _parse(self, text: str, what: str, allow_u: bool, allow_coords, keep_params=False) -> sp.Basic:
        try:
            e = parse_expression(text, self._resolver(allow_u, tuple(allow_coords), keep_params))
        except ParseError as exc:
            raise ValidationError(f"{what}: {exc}") from None
        if e.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            _fail(f"{what}: expression is undefined (division by zero or infinity).")
        return e

    @staticmethod
    def _split_eq(text: str, what: str) -> Tuple[str, str]:
        if not isinstance(text, str) or not text.strip():
            _fail(f"{what} is empty.")
        if "==" in text or text.count("=") != 1:
            _fail(f"{what} must contain exactly one '=' sign, e.g. \"u' = -2*u\".")
        lhs, rhs = text.split("=")
        if not lhs.strip() or not rhs.strip():
            _fail(f"{what} has an empty side of '='.")
        return lhs, rhs

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def validate(self) -> EquationSpec:
        self._setup()
        equation = self._rewrite_notation(self.problem.equation, strip_calls=True)
        if isinstance(equation, str) and equation.strip() and "=" not in equation:
            self.warnings.append(f"The equation has no '=' sign; it was read as  {equation.strip()} = 0.")
            equation = equation + " = 0"
        lhs, rhs = self._split_eq(equation, "Equation")
        everything = self.variables
        sides = (self._parse(lhs, "Equation (left side)", True, everything, keep_params=True),
                 self._parse(rhs, "Equation (right side)", True, everything, keep_params=True))
        res_disp = sides[0] - sides[1]
        residual = res_disp.subs({self.param_symbols[k]: sp.Float(v) for k, v in self.params.items()})
        try:
            check_constants(residual)  # e.g. k^0.5 with k < 0 only becomes complex once k is substituted
            check_expandable(residual)
        except ParseError as exc:
            raise ValidationError(f"Equation: {exc}") from None
        if residual.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            _fail("Equation: expression is undefined (division by zero or infinity) for the given parameter values.")
        residual = sp.expand(residual)
        present = [k for k, s in self.deriv_symbols.items() if s in residual.free_symbols and sum(k) > 0]
        if not present:
            _fail(f"The equation contains no derivative of {self.dep}.")
        if not self.problem.conditions:
            _fail("No initial or boundary conditions were given.")
        if len(self.problem.conditions) > MAX_CONDITIONS:
            _fail(f"Too many conditions (limit {MAX_CONDITIONS}).")

        if len(self.variables) == 1:
            spec = self._validate_ode(residual, res_disp, present)
        else:
            spec = self._validate_pde(residual, res_disp, present)
        spec.sides = sides
        self._attach_exact(spec)
        spec.warnings.extend(self.warnings)
        return spec

    # ------------------------------------------------------------------
    # conditions
    # ------------------------------------------------------------------
    def _parse_condition(self, text: str) -> Tuple[Tuple[int, ...], List[Optional[float]], sp.Basic]:
        """Parse ``u'(0) = 1`` / ``u(x, 0) = sin(pi*x)``.

        Returns (derivative key, argument list with None for a free variable, rhs).
        """
        lhs, rhs = self._split_eq(self._rewrite_notation(text, strip_calls=False), f"Condition {text!r}")
        m = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*'*)\s*\((.*)\)\s*", lhs)
        if not m:
            _fail(f"Condition {text!r}: the left side must look like {self.dep}(0) or {self.dep}'(0) or {self.dep}(x, 0).")
        key = self._deriv_key(m.group(1), 0)
        if key is None:
            _fail(f"Condition {text!r}: {m.group(1)!r} is not {self.dep} or a derivative of {self.dep}.")
        args = _split_args(m.group(2))
        if len(args) != len(self.variables):
            _fail(f"Condition {text!r}: expected {len(self.variables)} argument(s) in the order ({', '.join(self.variables)}).")
        values: List[Optional[float]] = []
        free: List[str] = []
        for i, a in enumerate(args):
            a = a.strip()
            if a in self.coord_by_name:
                if a != self.variables[i]:
                    _fail(f"Condition {text!r}: arguments must be in the order ({', '.join(self.variables)}).")
                values.append(None)
                free.append(a)
                continue
            e = self._parse(a, f"Condition {text!r} (argument)", False, ())
            if e.free_symbols:
                _fail(f"Condition {text!r}: argument {a!r} must be a constant.")
            values.append(float(e))
        if len(self.variables) == 2 and len(free) != 1:
            _fail(f"Condition {text!r}: exactly one argument must be a variable, e.g. {self.dep}(x, 0) or {self.dep}(0, t).")
        value = self._parse(rhs, f"Condition {text!r} (right side)", False, tuple(free))
        return key, values, value

    def _snap(self, var: str, v: float, text: str) -> Tuple[float, str]:
        lo, hi = self.domain[var]
        tol = 1e-9 * (1.0 + hi - lo)
        if abs(v - lo) <= tol:
            return lo, "left"
        if abs(v - hi) <= tol:
            return hi, "right"
        _fail(f"Condition {text!r}: {var} = {v:g} is not an end point of the domain [{lo:g}, {hi:g}] (interior conditions are not supported).")
        raise AssertionError  # unreachable

    # ------------------------------------------------------------------
    # ODEs
    # ------------------------------------------------------------------
    def _validate_ode(self, residual, res_disp, present) -> EquationSpec:
        var = self.variables[0]
        lo, hi = self.domain[var]
        n = max(k[0] for k in present)
        Dn = self.deriv_symbols[(n,)]
        a = sp.diff(residual, Dn)
        if Dn in a.free_symbols:
            _fail(f"The equation must be linear in its highest derivative {_pretty(self.dep, n)} (it is not).")
        u_syms = {self.deriv_symbols[(k,)] for k in range(n + 1)}
        if a.free_symbols & u_syms:
            _fail(f"The coefficient of {_pretty(self.dep, n)} may depend only on {var}, not on {self.dep} or its derivatives.")
        _check_nonvanishing(a, self.coords, [self.domain[var]], f"coefficient of {_pretty(self.dep, n)}")
        explicit = sp.expand(-(residual.subs(Dn, 0)) / a)
        _check_defined(explicit, self.coords, [self.domain[var]], "The equation")

        conds: List[Condition] = []
        seen = set()
        for text in self.problem.conditions:
            key, vals, value = self._parse_condition(text)
            if key[0] >= n:
                _fail(f"Condition {text!r}: for an order-{n} equation conditions may involve at most {_pretty(self.dep, n - 1)}.")
            if vals[0] is None:
                _fail(f"Condition {text!r}: give the point as a number, e.g. {_pretty(self.dep, key[0])}({lo:g}) = ...")
            at, side = self._snap(var, vals[0], text)
            if (side, key) in seen:
                _fail(f"Condition {text!r} duplicates another condition.")
            seen.add((side, key))
            conds.append(Condition(text=text, var=var, at=at, deriv=key, value=value, role="boundary", side=side))

        sides = {c.side for c in conds}
        orders = sorted(c.order for c in conds)
        if sides == {"left"} and orders == list(range(n)):
            kind = ODE_IVP
            for c in conds:
                c.role, c.side = "initial", None
            subtype = "first-order initial value problem" if n == 1 else "second-order initial value problem"
        elif n == 2 and sides == {"left", "right"} and len(conds) == 2:
            kind = ODE_BVP
            if all(c.order == 1 for c in conds):
                _fail("Both boundary conditions are Neumann (on u'); the solution is not unique. Give at least one condition on u itself.")
            only_second = not (residual.free_symbols & {self.deriv_symbols[(0,)], self.deriv_symbols[(1,)]})
            subtype = "1D Poisson-type boundary value problem" if only_second else "second-order boundary value problem"
        elif sides == {"right"}:
            _fail(f"Conditions must be given at the start of the interval {var} = {lo:g}; terminal-value problems are not supported.")
        else:
            if n == 1:
                _fail(f"A first-order ODE needs exactly one initial condition {self.dep}({lo:g}) = ...")
            _fail(
                f"A second-order ODE needs either {self.dep}({lo:g}) and {self.dep}'({lo:g}) (initial value problem) "
                f"or one condition at each end, {var} = {lo:g} and {var} = {hi:g} (boundary value problem)."
            )
        conds.sort(key=lambda c: (c.side or "", c.order))

        linear = _is_linear(residual, list(u_syms))
        spec = EquationSpec(
            problem=self.problem, kind=kind, subtype=subtype, variables=self.variables, coords=self.coords,
            dependent=self.dep, domain=self.domain, residual=residual, residual_display=res_disp,
            deriv_symbols=self.deriv_symbols, order=n, linear=linear, conditions=conds, explicit_rhs=explicit,
        )
        spec.features.update(self._ode_features(spec))
        if kind == ODE_BVP and linear:
            self._check_bvp_unique(spec)
        spec.features["output_scale"] = self._output_scale(spec, spec.features.pop("data_scale"), T_pow=(hi - lo) ** n)
        return spec

    def _check_bvp_unique(self, spec: EquationSpec) -> None:
        """Reject a linear BVP whose homogeneous problem has a non-zero solution (no unique solution).

        Shooting: integrate the homogeneous equation from the left end with the left condition
        satisfied (u=0,u'=1 for Dirichlet; u=1,u'=0 for Neumann) and test the right condition.
        """
        from scipy.integrate import solve_ivp

        x = self.coords[0]
        s0, s1 = self.deriv_symbols[(0,)], self.deriv_symbols[(1,)]
        q, p = sp.diff(spec.explicit_rhs, s0), sp.diff(spec.explicit_rhs, s1)
        lo, hi = spec.bounds[0]
        left, right = spec.condition("boundary", "left"), spec.condition("boundary", "right")

        def rhs(xv, y):
            env = {x: np.asarray(xv, dtype=float)}
            return [y[1], float(evaluate(q, env, like=env[x])) * y[0] + float(evaluate(p, env, like=env[x])) * y[1]]

        y0 = [0.0, 1.0] if left.order == 0 else [1.0, 0.0]
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(rhs, (lo, hi), y0, method="DOP853", rtol=1e-10, atol=1e-13, t_eval=np.linspace(lo, hi, 2001))
        except (ParseError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            return
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return
        row = sol.y[right.order]
        size = float(np.max(np.abs(row)))
        if size <= 0:
            return
        miss = abs(float(row[-1])) / size
        spec.features["bvp_uniqueness_margin"] = miss
        if miss < BVP_SINGULAR_TOL:
            _fail(
                "This boundary value problem has no unique solution: the equation without its source term has a "
                f"non-zero solution that satisfies both boundary conditions on [{lo:g}, {hi:g}] (a resonance of the "
                "domain length). Depending on the data there are infinitely many solutions or none. "
                "Change the domain or the boundary conditions."
            )
        if miss < BVP_ILL_CONDITIONED_TOL:
            self.warnings.append(
                f"This boundary value problem is very close to one without a unique solution (margin {miss:.1e}): "
                "small changes in the data change the solution a lot, so every method loses accuracy here."
            )

    def _output_scale(self, spec: EquationSpec, data_scale: float, T_pow: float) -> float:
        """Size of the solution used to rescale the network output and the loss (1.0 = no rescaling).

        Rescaling u = U*v is an exact symmetry only for linear equations, so it is used only there, only when
        the data are clearly not O(1), and only when the source term cannot dominate the data.
        """
        lo_band, hi_band = OUTPUT_SCALE_BAND
        if not spec.linear or not math.isfinite(data_scale) or data_scale <= 1e-12 or lo_band <= data_scale <= hi_band:
            return 1.0
        zero = {sym: 0 for sym in spec.deriv_symbols.values()}
        try:
            with np.errstate(all="ignore"):
                src = _grid_values(spec.explicit_rhs.subs(zero), self.coords, spec.bounds)
            forcing = float(np.max(np.abs(src))) * T_pow
        except (ValidationError, ValueError, TypeError):
            return 1.0
        if not math.isfinite(forcing) or forcing > data_scale:
            return 1.0
        return float(data_scale)

    def _ode_features(self, spec: EquationSpec) -> Dict[str, object]:
        var = self.variables[0]
        lo, hi = self.domain[var]
        T = hi - lo
        n = spec.order
        t = self.coords[0]
        U = [self.deriv_symbols[(k,)] for k in range(n)]
        F = spec.explicit_rhs
        dF = [sp.diff(F, s) for s in U]
        # State at which the Jacobian is evaluated.
        if spec.kind == ODE_IVP:
            state = [float(spec.condition("initial", order=k).value) for k in range(n)]
            where = "initial state"
        else:
            dir_vals = [float(c.value) for c in spec.conditions if c.order == 0]
            state = [float(np.mean(dir_vals)), 0.0][:n]
            where = "mean boundary value"
        ts = np.linspace(lo, hi, 41)
        stiff, growth, freq = 0.0, 0.0, 0.0
        ok = True
        for tv in ts:
            env = {t: np.array(tv)}
            env.update({s: np.array(v) for s, v in zip(U, state)})
            try:
                with np.errstate(all="ignore"):
                    vals = [float(evaluate(e, env)) for e in dF]
            except (ParseError, ValueError, TypeError, ZeroDivisionError, OverflowError):
                ok = False
                break
            if n == 1:
                eig = np.array([vals[0]], dtype=complex)
            else:
                eig = np.linalg.eigvals(np.array([[0.0, 1.0], [vals[0], vals[1]]]))
            if not np.all(np.isfinite(eig)):
                ok = False
                break
            stiff = max(stiff, float(np.max(-eig.real)))
            growth = max(growth, float(np.max(eig.real)))
            freq = max(freq, float(np.max(np.abs(eig.imag))))
        feats: Dict[str, object] = {"interval_length": T}
        # size of the prescribed data: |u| values, and slopes times the interval length
        feats["data_scale"] = max([abs(float(c.value)) * (T if c.order else 1.0) for c in spec.conditions] or [0.0])
        if ok:
            feats.update(
                stiffness_index=stiff * T,
                growth_index=growth * T,
                oscillation_cycles=freq * T / (2 * math.pi),
                jacobian_evaluated_at=where if not spec.linear else "exact (linear equation)",
            )
            feats["stiff"] = feats["stiffness_index"] > STIFFNESS_INDEX_LIMIT
            feats["oscillatory"] = feats["oscillation_cycles"] > OSCILLATION_CYCLES_LIMIT
            feats["high_frequency"] = feats["oscillation_cycles"] > HIGH_FREQUENCY_CYCLES
            if feats["growth_index"] > GROWTH_INDEX_LIMIT:
                self.warnings.append(
                    f"Linearised growth rate times interval length is {feats['growth_index']:.1f}; the solution may grow "
                    "very quickly, which makes errors hard to control."
                )
        else:
            feats.update(stiff=False, oscillatory=False, high_frequency=False, jacobian_evaluated_at="unavailable")
            self.warnings.append("Could not estimate stiffness/oscillation features; default strategy rules will be used.")
        return feats

    # ------------------------------------------------------------------
    # PDEs
    # ------------------------------------------------------------------
    def _validate_pde(self, residual, res_disp, present) -> EquationSpec:
        xs, ts_ = self.variables
        k_t, k_x, k_xx = (0, 1), (1, 0), (2, 0)
        for k in present:
            if k[1] >= 1 and k != k_t:
                _fail(f"Only first-order time derivatives ({self.dep}_t) are supported; found {_deriv_name(self.dep, self.variables, k)}.")
        if k_t not in present:
            _fail(f"A time-dependent PDE must contain {self.dep}_t. Steady problems in one variable should be entered as ODEs.")
        if k_xx not in present:
            _fail(f"The PDE must contain a diffusion term {self.dep}_{xs}{xs}; pure advection / first-order PDEs are not supported.")
        u_t = self.deriv_symbols[k_t]
        u_syms = [self.deriv_symbols[k] for k in [(0, 0), k_x, k_xx, k_t]]
        a = sp.diff(residual, u_t)
        if a.free_symbols & set(u_syms):
            _fail(f"The coefficient of {self.dep}_t may depend only on {xs} and t.")
        bounds = [self.domain[xs], self.domain[ts_]]
        _check_nonvanishing(a, self.coords, bounds, f"coefficient of {self.dep}_t")
        F = sp.expand(-(residual.subs(u_t, 0)) / a)
        _check_defined(F, self.coords, bounds, "The equation")
        u_xx = self.deriv_symbols[k_xx]
        D = sp.diff(F, u_xx)
        if D.free_symbols & set(u_syms):
            _fail(f"The equation must be linear in {self.dep}_{xs}{xs} with a coefficient that depends only on {xs} and t.")
        Dv = _grid_values(D, self.coords, bounds)
        if not np.all(np.isfinite(Dv)) or Dv.min() <= 0:
            _fail(
                f"Written as {self.dep}_t = F, the coefficient of {self.dep}_{xs}{xs} must be positive everywhere "
                "(a negative coefficient gives the ill-posed backward heat equation)."
            )

        conds: List[Condition] = []
        have = set()
        for text in self.problem.conditions:
            key, vals, value = self._parse_condition(text)
            if sum(key) != 0:
                _fail(f"Condition {text!r}: only Dirichlet conditions on {self.dep} itself are supported for PDEs.")
            if vals[1] is not None and vals[0] is None:
                at, side = self._snap("t", vals[1], text)
                if side != "left":
                    _fail(f"Condition {text!r}: the initial condition must be at t = {self.domain['t'][0]:g}.")
                role, var, side = "initial", "t", None
            else:
                at, side = self._snap(xs, vals[0], text)
                role, var = "boundary", xs
            tag = (role, side)
            if tag in have:
                _fail(f"Condition {text!r} duplicates another condition.")
            have.add(tag)
            free = self.coord_by_name[xs if role == "initial" else "t"]
            _check_defined(value, (free,), [self.domain[str(free)]], f"Condition {text!r}")
            conds.append(Condition(text=text, var=var, at=at, deriv=(0, 0), value=value, role=role, side=side))
        if have != {("initial", None), ("boundary", "left"), ("boundary", "right")}:
            lo, hi = self.domain[xs]
            t0 = self.domain["t"][0]
            _fail(
                f"Give exactly three conditions: {self.dep}({xs}, {t0:g}) = g({xs}), "
                f"{self.dep}({lo:g}, t) = ... and {self.dep}({hi:g}, t) = ..."
            )
        conds.sort(key=lambda c: (c.role != "initial", c.side or ""))

        linear = _is_linear(residual, u_syms)
        spec = EquationSpec(
            problem=self.problem, kind=PDE_PARABOLIC, subtype="", variables=self.variables, coords=self.coords,
            dependent=self.dep, domain=self.domain, residual=residual, residual_display=res_disp,
            deriv_symbols=self.deriv_symbols, order=2, linear=linear, conditions=conds, explicit_rhs=F,
        )
        spec.features.update(self._pde_features(spec, D))
        T = self.domain[ts_][1] - self.domain[ts_][0]
        spec.features["output_scale"] = self._output_scale(spec, float(spec.features["solution_scale"]), T_pow=T)
        return spec

    def _pde_features(self, spec: EquationSpec, D: sp.Basic) -> Dict[str, object]:
        xs = self.variables[0]
        (a, b), (t0, T1) = self.domain[xs], self.domain["t"]
        L, T = b - a, T1 - t0
        x, t = self.coords
        k = lambda kk: self.deriv_symbols[kk]
        F = spec.explicit_rhs
        bounds = [self.domain[xs], self.domain["t"]]
        Dv = _grid_values(D, self.coords, bounds)
        g = spec.condition("initial").value
        h0 = spec.condition("boundary", "left").value
        h1 = spec.condition("boundary", "right").value
        gx = _grid_values(g, (x,), [bounds[0]])
        h0v = _grid_values(h0, (t,), [bounds[1]])
        h1v = _grid_values(h1, (t,), [bounds[1]])
        M = float(max(np.max(np.abs(gx)), np.max(np.abs(h0v)), np.max(np.abs(h1v)), 1e-12))
        # Linearised advection speed and reaction rate, sampled for u in [-M, M].
        dF_dux = sp.diff(F, k((1, 0)))
        dF_du = sp.diff(F, k((0, 0)))
        U_adv, R = 0.0, 0.0
        skipped = False
        for uval in (-M, 0.0, M):
            sub = {k((0, 0)): uval, k((1, 0)): 0.0, k((2, 0)): 0.0}
            # A sample state can lie where the equation is undefined (1/u or log(u) at u = 0); such a
            # sample says nothing about the problem itself, so it is skipped rather than rejected.
            try:
                with np.errstate(all="ignore"):
                    adv = np.abs(_grid_values(dF_dux.subs(sub), self.coords, bounds))
                    rea = np.abs(_grid_values(dF_du.subs(sub), self.coords, bounds))
            except ValidationError:
                skipped = True
                continue
            if not (np.all(np.isfinite(adv)) and np.all(np.isfinite(rea))):
                skipped = True
                continue
            U_adv = max(U_adv, float(np.max(adv)))
            R = max(R, float(np.max(rea)))
        if skipped:
            self.warnings.append(
                f"The equation is undefined for some values of {self.dep} between -{M:.3g} and {M:.3g} (for example a "
                f"division by {self.dep} or a logarithm of {self.dep}); the solution must stay away from those values, "
                "and the advection/reaction estimates used to choose the strategy are incomplete.")
        Dmin, Dmax = float(Dv.min()), float(Dv.max())
        peclet = U_adv * L / Dmin
        feats: Dict[str, object] = {
            "diffusion_min": Dmin,
            "diffusion_max": Dmax,
            "advection_speed": U_adv,
            "reaction_rate": R,
            "peclet_number": peclet,
            "damkohler_number": R * T,
            "diffusion_time_ratio": Dmax * T / L**2,
            "solution_scale": M,
        }
        feats["steep_gradient"] = peclet > PECLET_STEEP_LIMIT
        # Corner compatibility of the initial and boundary data.
        g_a = float(evaluate(g, {x: np.array(a)}))
        g_b = float(evaluate(g, {x: np.array(b)}))
        h0_t0 = float(evaluate(h0, {t: np.array(t0)}))
        h1_t0 = float(evaluate(h1, {t: np.array(t0)}))
        mismatch = max(abs(g_a - h0_t0), abs(g_b - h1_t0))
        compatible = mismatch <= 1e-8 * (1.0 + M)
        feats["corner_mismatch"] = mismatch
        feats["corners_compatible"] = compatible
        if not compatible:
            self.warnings.append(
                f"Initial and boundary data disagree at a corner (mismatch {mismatch:.3g}); the exact solution is "
                "discontinuous there, hard constraints are disabled and errors near the corner will be large."
            )
        has_ux = k((1, 0)) in F.free_symbols
        u_dependent_adv = bool(dF_dux.free_symbols & {k((0, 0))})
        if spec.linear and not has_ux and k((0, 0)) not in F.free_symbols:
            spec.subtype = "1D heat / diffusion equation"
        elif u_dependent_adv:
            spec.subtype = "nonlinear advection-diffusion (Burgers-type)"
        elif spec.linear:
            spec.subtype = "linear advection-diffusion-reaction equation"
        else:
            spec.subtype = "nonlinear reaction-diffusion equation"
        return feats

    # ------------------------------------------------------------------
    # optional exact solution supplied by the user
    # ------------------------------------------------------------------
    def _attach_exact(self, spec: EquationSpec) -> None:
        text = self.problem.exact
        if not text or not str(text).strip():
            return
        text = re.sub(rf"^\s*{re.escape(self.dep)}\s*(\([^()]*\))?\s*=", "", normalize_text(str(text)))
        exact = self._parse(text, "Exact solution", False, self.variables)
        ok, detail = verify_exact(spec, exact)
        if ok:
            spec.exact = exact
            spec.features["exact_solution_verified"] = True
        else:
            spec.features["exact_solution_verified"] = False
            self.warnings.append(f"The supplied exact solution was NOT used as a reference: {detail}")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def verify_exact(spec: EquationSpec, exact: sp.Basic, n_points: int = 256) -> Tuple[bool, str]:
    """Check numerically that ``exact`` satisfies the equation and all conditions."""
    rng = np.random.default_rng(12345)
    pts = [rng.uniform(lo, hi, n_points) for lo, hi in spec.bounds]
    env = {c: p for c, p in zip(spec.coords, pts)}
    try:
        values = {}
        for key, sym in spec.deriv_symbols.items():
            if sym in spec.residual.free_symbols:
                d = exact
                for c, cnt in zip(spec.coords, key):
                    if cnt:
                        d = sp.diff(d, c, cnt)
                values[sym] = d
        res = spec.residual.subs(values)
        with np.errstate(all="ignore"):
            r = np.asarray(evaluate(res, env, like=pts[0]), dtype=float)
            # the solution itself must be finite on the whole closed domain (end points and corners included),
            # otherwise errors measured against it would be NaN
            mesh = np.meshgrid(*[np.linspace(lo, hi, 1001 if spec.dim == 1 else 101) for lo, hi in spec.bounds], indexing="ij")
            on_grid = np.asarray(evaluate(exact, {c: m.ravel() for c, m in zip(spec.coords, mesh)}, like=mesh[0].ravel()), dtype=float)
        if not np.all(np.isfinite(on_grid)):
            return False, "it is not finite everywhere in the domain (for example 0/0 at an end point)."
        scale = 1.0 + max(float(np.max(np.abs(np.asarray(evaluate(v, env, like=pts[0]), dtype=float)))) for v in values.values())
        if not np.all(np.isfinite(r)):
            return False, "it is not finite everywhere in the domain."
        if np.max(np.abs(r)) > 1e-7 * scale:
            return False, f"it does not satisfy the equation (max residual {np.max(np.abs(r)):.3g} at random points)."
        for c in spec.conditions:
            d = exact
            for cs, cnt in zip(spec.coords, c.deriv):
                if cnt:
                    d = sp.diff(d, cs, cnt)
            idx = spec.variables.index(c.var)
            cenv = dict(env)
            cenv[spec.coords[idx]] = np.full(n_points, c.at)
            lhs = np.asarray(evaluate(d, cenv, like=pts[0]), dtype=float)
            rhs = np.asarray(evaluate(c.value, cenv, like=pts[0]), dtype=float)
            if not (np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))):
                return False, f"it is not finite where the condition {c.text!r} is imposed."
            err = float(np.max(np.abs(lhs - rhs)))
            if err > 1e-7 * (1.0 + float(np.max(np.abs(rhs)))):
                return False, f"it does not satisfy the condition {c.text!r} (error {err:.3g})."
    except (ParseError, ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
        return False, f"it could not be checked ({exc})."
    return True, "verified"


def _multi_indices(d: int, total: int):
    if d == 1:
        yield (total,)
        return
    for i in range(total, -1, -1):
        for rest in _multi_indices(d - 1, total - i):
            yield (i,) + rest


def _deriv_name(dep: str, variables, key) -> str:
    if sum(key) == 0:
        return dep
    return dep + "_" + "".join(v * c for v, c in zip(variables, key))


def _pretty(dep: str, n: int) -> str:
    return dep + "'" * n


def _split_args(s: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _is_linear(expr: sp.Basic, syms) -> bool:
    for i, a in enumerate(syms):
        da = sp.diff(expr, a)
        for b in syms[i:]:
            if sp.simplify(sp.diff(da, b)) != 0:
                return False
    return True


def _grid_values(e: sp.Basic, coords, bounds, n: int = 41) -> np.ndarray:
    axes = [np.linspace(lo, hi, n) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    env = {c: m.ravel() for c, m in zip(coords, mesh)}
    try:
        v = np.asarray(evaluate(e, env, like=mesh[0].ravel()), dtype=float)
    except (ParseError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValidationError(f"Could not evaluate {e} on the domain: {exc}") from None
    return v


def _split_into(name: str, known) -> Optional[List[str]]:
    """Split ``name`` into two or more known names (e.g. 'ku' -> ['k', 'u']), or None."""
    known = sorted((k for k in known if k), key=len, reverse=True)

    def go(rest: str) -> Optional[List[str]]:
        if not rest:
            return []
        for k in known:
            if rest.startswith(k):
                tail = go(rest[len(k):])
                if tail is not None:
                    return [k] + tail
        return None

    if len(name) > 12:
        return None
    parts = go(name)
    return parts if parts and len(parts) > 1 else None


def _check_defined(e: sp.Basic, coords, bounds, what: str) -> None:
    """Reject expressions whose coordinate-only parts are undefined somewhere in the domain.

    Looks at every sub-expression that depends only on the independent variables: a denominator that
    reaches zero, a logarithm of a non-positive number, a root of a negative number, or any non-finite value.
    """
    cs = set(coords)
    n = 2001 if len(coords) == 1 else 201

    def vals(node):
        with np.errstate(all="ignore"):
            return _grid_values(node, coords, bounds, n=n)

    def at(index: int) -> str:
        axes = [np.linspace(lo, hi, n) for lo, hi in bounds]
        idx = np.unravel_index(int(index), [n] * len(coords))
        return ", ".join(f"{c} = {axes[i][j]:.4g}" for i, (c, j) in enumerate(zip(coords, idx)))

    def bad(reason: str, index: int) -> None:
        _fail(f"{what} is undefined inside the domain ({reason} near {at(index)}). "
              "Choose a domain that does not contain this point.")

    for node in sp.preorder_traversal(e):
        fs = node.free_symbols
        if not fs or not fs <= cs:
            continue
        if node.is_Pow and node.exp.is_number and node.exp.is_negative:
            b = vals(node.base)
            if np.all(np.isfinite(b)):
                i = int(np.argmin(np.abs(b)))
                if abs(b[i]) <= 1e-12 * max(1.0, float(np.max(np.abs(b)))) or (b.min() < 0 < b.max()):
                    bad("division by zero", i)
        elif node.func is sp.log:
            b = vals(node.args[0])
            if np.all(np.isfinite(b)) and b.min() <= 0:
                bad("logarithm of a number that is not positive", int(np.argmin(b)))
        elif node.is_Pow and node.exp.is_number and not node.exp.is_integer:
            b = vals(node.base)
            if np.all(np.isfinite(b)) and b.min() < 0:
                bad("root or fractional power of a negative number", int(np.argmin(b)))
        v = vals(node)
        if not np.all(np.isfinite(v)):
            bad("the value is not finite", int(np.argmax(~np.isfinite(v))))


def _check_nonvanishing(a: sp.Basic, coords, bounds, what: str) -> None:
    with np.errstate(all="ignore"):
        v = _grid_values(a, coords, bounds, n=101 if len(coords) == 1 else 41)
    if not np.all(np.isfinite(v)):
        _fail(f"The {what} is not finite everywhere in the domain.")
    if np.min(np.abs(v)) < 1e-10 or (v.min() < 0 < v.max()):
        _fail(f"The {what} vanishes or changes sign inside the domain (singular equation; not supported).")


def parse_problem(problem: ProblemInput) -> EquationSpec:
    """Convenience wrapper: validate ``problem`` and return its EquationSpec."""
    return Parser(problem).validate()
