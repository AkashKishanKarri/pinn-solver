"""Problem input (M1) and structured equation representation (M3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sympy as sp

# Problem kinds supported by the first version (Table 4.2 of the report).
ODE_IVP = "ode_ivp"
ODE_BVP = "ode_bvp"
PDE_PARABOLIC = "pde_parabolic_1d"


@dataclass
class ProblemInput:
    """Raw user input (M1). All mathematical fields are plain strings."""

    equation: str
    variables: List[str]
    domain: Dict[str, Tuple[float, float]]
    conditions: List[str]
    dependent: str = "u"
    parameters: Dict[str, float] = field(default_factory=dict)
    exact: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "equation": self.equation,
            "variables": list(self.variables),
            "dependent": self.dependent,
            "domain": {k: [float(v[0]), float(v[1])] for k, v in self.domain.items()},
            "conditions": list(self.conditions),
            "parameters": {k: float(v) for k, v in self.parameters.items()},
            "exact": self.exact,
        }

    @staticmethod
    def from_dict(d: dict) -> "ProblemInput":
        try:
            if not isinstance(d, dict):
                raise TypeError("the problem must be a JSON object")
            for k, v in d["domain"].items():
                if isinstance(v, (str, bytes)) or len(v) != 2:
                    raise ValueError(f"domain of {k} must be [lower, upper]")
            conditions = d.get("conditions") or []
            variables = d["variables"]
            return ProblemInput(
                equation=d["equation"],
                variables=[variables] if isinstance(variables, str) else list(variables),
                domain={k: (float(v[0]), float(v[1])) for k, v in d["domain"].items()},
                conditions=[conditions] if isinstance(conditions, str) else list(conditions),
                dependent=d.get("dependent", "u") or "u",
                parameters={k: float(v) for k, v in (d.get("parameters") or {}).items()},
                exact=(d.get("exact") or None),
                name=d.get("name"),
            )
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise ValueError(f"Malformed problem input: {exc}") from exc


@dataclass
class Condition:
    """One initial or boundary condition, e.g. ``u'(0) = 0`` or ``u(0, t) = sin(t)``."""

    text: str
    var: str  # variable that is held fixed
    at: float  # location of the condition along ``var``
    deriv: Tuple[int, ...]  # derivative multi-index over spec.variables
    value: sp.Basic  # right-hand side; may depend on the free variable (PDE) only
    role: str  # 'initial' | 'boundary'
    side: Optional[str] = None  # 'left' | 'right' for boundary conditions

    @property
    def order(self) -> int:
        return sum(self.deriv)

    @property
    def kind(self) -> str:
        return "dirichlet" if self.order == 0 else "neumann"

    def latex(self, dependent: str, variables: Tuple[str, ...]) -> str:
        """Typeset form, e.g. ``u'(0) = 1`` or ``u(x, 0) = \\sin(\\pi x)``."""
        if len(variables) == 1:
            name = dependent + "'" * self.order
        else:
            sub = "".join(v * k for v, k in zip(variables, self.deriv))
            name = dependent + (f"_{{{sub}}}" if sub else "")
        args = ", ".join(_num_latex(self.at) if v == self.var else v for v in variables)
        return f"{name}({args}) = {sp.latex(self.value)}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "role": self.role,
            "side": self.side,
            "type": self.kind,
            "variable": self.var,
            "at": self.at,
            "derivative_order": self.order,
            "value": str(self.value),
        }


@dataclass
class EquationSpec:
    """Structured, validated problem (M3); consumed by every later module."""

    problem: ProblemInput
    kind: str
    subtype: str
    variables: Tuple[str, ...]
    coords: Tuple[sp.Symbol, ...]
    dependent: str
    domain: Dict[str, Tuple[float, float]]
    residual: sp.Basic  # F(...) with F = 0, parameters substituted
    residual_display: sp.Basic  # same, with parameter symbols kept
    deriv_symbols: Dict[Tuple[int, ...], sp.Symbol]
    order: int
    linear: bool
    conditions: List[Condition]
    explicit_rhs: sp.Basic  # highest derivative (ODE) or u_t (PDE) solved explicitly
    exact: Optional[sp.Basic] = None
    features: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    sides: Optional[Tuple[sp.Basic, sp.Basic]] = None  # (left, right) of the equation as entered, for display

    # convenience --------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def u_symbol(self) -> sp.Symbol:
        return self.deriv_symbols[tuple([0] * self.dim)]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [self.domain[v] for v in self.variables]

    @property
    def bc_types(self) -> List[str]:
        return [c.kind for c in self.conditions if c.role == "boundary"]

    def needed_derivatives(self) -> List[Tuple[int, ...]]:
        free = self.residual.free_symbols
        return [k for k, s in self.deriv_symbols.items() if s in free and sum(k) > 0]

    def condition(self, role: str, side: Optional[str] = None, order: Optional[int] = None) -> Optional[Condition]:
        for c in self.conditions:
            if c.role == role and (side is None or c.side == side) and (order is None or c.order == order):
                return c
        return None

    def to_dict(self) -> dict:
        def num(v):
            if isinstance(v, (bool, str)) or v is None:
                return v
            try:
                return float(v)
            except (TypeError, ValueError):
                return str(v)

        return {
            "kind": self.kind,
            "subtype": self.subtype,
            "variables": list(self.variables),
            "dependent": self.dependent,
            "domain": {k: list(v) for k, v in self.domain.items()},
            "equation": f"{sp.sstr(self.residual_display)} = 0",
            "equation_latex": self.equation_latex(),
            "conditions_latex": [c.latex(self.dependent, self.variables) for c in self.conditions],
            "exact_latex": (f"{self.dependent}({', '.join(self.variables)}) = {sp.latex(self.exact)}"
                            if self.exact is not None else None),
            "equation_numeric": f"{sp.sstr(self.residual)} = 0",
            "explicit_form": f"{self._lhs_name()} = {sp.sstr(self.explicit_rhs)}",
            "order": self.order,
            "linear": self.linear,
            "conditions": [c.to_dict() for c in self.conditions],
            "boundary_condition_types": self.bc_types,
            "parameters": dict(self.problem.parameters),
            "exact_solution": str(self.exact) if self.exact is not None else None,
            "features": {k: num(v) for k, v in self.features.items()},
            "warnings": list(self.warnings),
        }

    def equation_latex(self) -> str:
        """The equation as entered (left = right), with derivatives typeset as u', u'' or u_t, u_{xx}."""
        names = {}
        for key, sym in self.deriv_symbols.items():
            if sum(key) == 0:
                continue
            if self.dim == 1:
                names[sym] = self.dependent + "'" * key[0]
            else:
                names[sym] = f"{self.dependent}_{{{''.join(v * k for v, k in zip(self.variables, key))}}}"
        if self.sides is None:
            return sp.latex(self.residual_display, symbol_names=names) + " = 0"
        return " = ".join(sp.latex(side, symbol_names=names) for side in self.sides)

    def _lhs_name(self) -> str:
        if self.kind == PDE_PARABOLIC:
            return f"{self.dependent}_t"
        return f"{self.dependent}{chr(39) * self.order}"


def _num_latex(v: float) -> str:
    """A condition location: integers without a decimal point, simple multiples of pi as such."""
    for k in (1, 2, 3, 4, 6):
        m = v * k / float(sp.pi)
        if abs(m) > 1e-12 and abs(m - round(m)) < 1e-9 and abs(round(m)) <= 12:
            return sp.latex(sp.Rational(int(round(m)), k) * sp.pi)
    return f"{v:g}"
