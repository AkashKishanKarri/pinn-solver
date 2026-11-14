"""PINN network, hard-constraint trial solutions and physics residuals (part of M5).

Inputs are scaled to [-1, 1] inside :meth:`PINN.network`, but all derivatives
are taken by automatic differentiation with respect to the *physical*
coordinates, so the chain-rule factors (2/T)^k of Eq. (6.1) are applied
exactly and automatically.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .expr import evaluate
from .spec import ODE_BVP, ODE_IVP, PDE_PARABOLIC, EquationSpec

DTYPE = torch.float64


def output_scale(spec: EquationSpec) -> float:
    """Factor U by which the raw network output is multiplied (chosen by the parser; 1.0 = none).

    With u = U*v the network learns an O(1) function v, and the loss divides residuals and condition
    errors by U, so a problem with tiny or huge data trains exactly like its O(1) counterpart.
    """
    s = spec.features.get("output_scale", 1.0)
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 1.0
    return s if math.isfinite(s) and s > 0 else 1.0


def hard_constraint_plan(spec: EquationSpec) -> Tuple[bool, str]:
    """Whether an exact (Lagaris-style) trial solution exists for ``spec``, and why."""
    if spec.kind == ODE_IVP:
        return True, "u(t) = [Taylor polynomial of the initial data] + ((t - t0)/T)^n N(t) satisfies all initial conditions exactly."
    if spec.kind == ODE_BVP:
        if all(c.order == 0 for c in spec.conditions):
            return True, "u(x) = a(1 - s) + b s + s(1 - s) N(x), s = (x - x0)/L, satisfies both Dirichlet conditions exactly."
        return False, "No hard-constraint construction is used for Neumann conditions; they are imposed as penalty terms."
    if spec.kind == PDE_PARABOLIC:
        if spec.features.get("corners_compatible", False):
            return True, "u = A(x, t) + tau s(1 - s) N(x, t) (transfinite interpolation of initial and boundary data) satisfies all three conditions exactly."
        return False, "Initial and boundary data are incompatible at a corner, so no exact construction exists; penalty terms are used."
    return False, "No construction available."


class PINN(nn.Module):
    """Fully connected tanh network with an optional hard-constraint trial solution."""

    def __init__(self, spec: EquationSpec, hidden: Sequence[int] = (32, 32, 32), hard_constraints: bool = True):
        super().__init__()
        self.spec = spec
        self.dim = spec.dim
        lo = torch.tensor([b[0] for b in spec.bounds], dtype=DTYPE)
        hi = torch.tensor([b[1] for b in spec.bounds], dtype=DTYPE)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        layers: List[nn.Module] = []
        width_in = self.dim
        for w in hidden:
            lin = nn.Linear(width_in, w, dtype=DTYPE)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            width_in = w
        out = nn.Linear(width_in, 1, dtype=DTYPE)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = nn.Sequential(*layers)
        possible, _ = hard_constraint_plan(spec)
        self.hard = bool(hard_constraints and possible)
        self.hidden = list(hidden)
        self.scale = output_scale(spec)

    # ------------------------------------------------------------------
    def network(self, X: torch.Tensor) -> torch.Tensor:
        Z = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        out = self.mlp(Z)[:, 0]
        return out if self.scale == 1.0 else out * self.scale

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.trial_solution(X)

    def trial_solution(self, X: torch.Tensor) -> torch.Tensor:
        N = self.network(X)
        if not self.hard:
            return N
        spec = self.spec
        if spec.kind == ODE_IVP:
            t0, t1 = spec.bounds[0]
            dt = X[:, 0] - t0
            s = dt / (t1 - t0)
            u0 = float(spec.condition("initial", order=0).value)
            if spec.order == 1:
                return u0 + s * N
            v0 = float(spec.condition("initial", order=1).value)
            return u0 + v0 * dt + s**2 * N
        if spec.kind == ODE_BVP:
            a, b = spec.bounds[0]
            s = (X[:, 0] - a) / (b - a)
            ua = float(spec.condition("boundary", "left").value)
            ub = float(spec.condition("boundary", "right").value)
            return ua * (1 - s) + ub * s + s * (1 - s) * N
        if spec.kind == PDE_PARABOLIC:
            (a, b), (t0, t1) = spec.bounds
            x_sym, t_sym = spec.coords
            x, t = X[:, 0], X[:, 1]
            s = (x - a) / (b - a)
            tau = (t - t0) / (t1 - t0)
            g = evaluate(spec.condition("initial").value, {x_sym: x}, backend="torch", like=x)
            h0 = spec.condition("boundary", "left").value
            h1 = spec.condition("boundary", "right").value
            h0_t = evaluate(h0, {t_sym: t}, backend="torch", like=t)
            h1_t = evaluate(h1, {t_sym: t}, backend="torch", like=t)
            h0_0 = float(evaluate(h0, {t_sym: torch.tensor(t0, dtype=DTYPE)}, backend="torch"))
            h1_0 = float(evaluate(h1, {t_sym: torch.tensor(t0, dtype=DTYPE)}, backend="torch"))
            A = g + (h0_t - h0_0) * (1 - s) + (h1_t - h1_0) * s
            return A + tau * s * (1 - s) * N
        raise ValueError(f"Unsupported kind {spec.kind}")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {
            "type": "fully connected (MLP)",
            "activation": "tanh",
            "hidden_layers": self.hidden,
            "inputs_scaled_to": "[-1, 1]",
            "parameters": self.n_parameters(),
            "precision": "float64",
            "hard_constraints": self.hard,
            "output_scale": self.scale,
        }


# ----------------------------------------------------------------------
# derivatives and residuals
# ----------------------------------------------------------------------

def derivatives(model: nn.Module, X: torch.Tensor, keys: Sequence[Tuple[int, ...]], create_graph: bool = True) -> Dict[Tuple[int, ...], torch.Tensor]:
    """u and the requested partial derivatives of u at X (autograd, exact)."""
    if not X.requires_grad:
        X = X.requires_grad_(True)
    d = X.shape[1]
    u = model(X)
    out: Dict[Tuple[int, ...], torch.Tensor] = {tuple([0] * d): u}
    keys = [k for k in keys if sum(k) > 0]
    if not keys:
        return out
    grad = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    for i in range(d):
        e = [0] * d
        e[i] = 1
        out[tuple(e)] = grad[:, i]
    for k in keys:
        if sum(k) != 2:
            continue
        i = next(j for j in range(d) if k[j] > 0)
        rest = list(k)
        rest[i] -= 1
        j = next(jj for jj in range(d) if rest[jj] > 0)
        g2 = torch.autograd.grad(grad[:, i], X, torch.ones_like(u), create_graph=create_graph)[0]
        out[k] = g2[:, j]
    return out


def residual(model: nn.Module, spec: EquationSpec, X: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """r(X) = LHS - RHS of the equation as entered, evaluated on the model."""
    if not X.requires_grad:
        X = X.requires_grad_(True)
    ders = derivatives(model, X, spec.needed_derivatives(), create_graph=create_graph)
    env = {c: X[:, i] for i, c in enumerate(spec.coords)}
    for k, sym in spec.deriv_symbols.items():
        if k in ders:
            env[sym] = ders[k]
    return evaluate(spec.residual, env, backend="torch", like=X[:, 0])


def condition_values(model: nn.Module, spec: EquationSpec, cond, X: torch.Tensor, create_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """(model value of the condition's left side, prescribed value) at points X on the condition's set."""
    if not X.requires_grad:
        X = X.requires_grad_(True)
    if cond.order == 0:
        val = model(X)
    else:
        val = derivatives(model, X, [cond.deriv], create_graph=create_graph)[cond.deriv]
    env = {c: X[:, i] for i, c in enumerate(spec.coords)}
    target = evaluate(cond.value, env, backend="torch", like=X[:, 0])
    return val, target


def to_tensor(a, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(a, dtype=DTYPE)
    if requires_grad:
        t = t.clone().requires_grad_(True)
    return t
