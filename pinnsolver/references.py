"""Reference solvers (M8): exact solutions, SciPy solutions, finite differences.

Reference solutions are used only to *evaluate* the PINN and as the
transparent numerical fallback; they never enter the training loss.
Each numerical reference reports its settings and an estimate of its own
error obtained by re-solving with looser settings (tolerance or grid).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import RectBivariateSpline

from .expr import evaluate
from .spec import ODE_BVP, ODE_IVP, PDE_PARABOLIC, EquationSpec


class ReferenceUnavailable(RuntimeError):
    """A reference solution could not be produced for this problem.

    For an initial value problem whose integration broke down part-way, ``stopped_at`` is the last
    time reached and ``last_value`` the size of the solution there (evidence of blow-up).
    """

    stopped_at: Optional[float] = None
    last_value: Optional[float] = None


# Wall-clock limit for one numerical reference solve. The reference is not part of the PINN time budget,
# so without a limit an extremely stiff or fine-grid problem could keep a job busy indefinitely.
REFERENCE_TIME_LIMIT = 120.0


class _Stop(Exception):
    """Raised inside a right-hand side to abandon an integration (non-finite value or time limit)."""

    def __init__(self, reason: str, at: Optional[float] = None, size: Optional[float] = None):
        super().__init__(reason)
        self.reason, self.at, self.size = reason, at, size


class _Guard:
    """Checks every right-hand-side evaluation: SciPy's integrators can loop forever on NaN/Inf."""

    def __init__(self, limit: float = None):
        self.deadline = time.perf_counter() + (REFERENCE_TIME_LIMIT if limit is None else limit)
        self.calls = 0

    def check(self, t, y, value) -> None:
        if not np.all(np.isfinite(value)):
            size = float(np.max(np.abs(y))) if np.all(np.isfinite(y)) else float("inf")
            raise _Stop("the equation's right-hand side is not finite", at=float(np.max(t)) if np.ndim(t) else float(t), size=size)
        self.calls += 1
        if self.calls % 64 == 0 and time.perf_counter() > self.deadline:
            raise _Stop(f"the numerical solver exceeded its time limit ({REFERENCE_TIME_LIMIT:.0f} s)")


@dataclass
class ReferenceSolution:
    method: str
    kind: str  # 'analytical' | 'numerical'
    settings: Dict[str, object]
    func: Callable[[np.ndarray], np.ndarray]
    runtime_s: float
    estimated_error: Optional[float] = None  # estimated max abs error of this reference
    notes: list = field(default_factory=list)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(self.func(X), dtype=float).reshape(-1)

    def describe(self) -> dict:
        return {
            "method": self.method,
            "kind": self.kind,
            "settings": self.settings,
            "estimated_max_error": self.estimated_error,
            "runtime_s": round(self.runtime_s, 4),
            "notes": list(self.notes),
        }


class ReferenceSolver:
    """Base class; ``solve(spec)`` returns a :class:`ReferenceSolution`."""

    name = "base"

    def solve(self, spec: EquationSpec) -> ReferenceSolution:  # pragma: no cover - interface
        raise NotImplementedError


class AnalyticalReference(ReferenceSolver):
    name = "analytical"

    def solve(self, spec: EquationSpec) -> ReferenceSolution:
        if spec.exact is None:
            raise ReferenceUnavailable("No verified exact solution is available.")
        exact = spec.exact
        coords = spec.coords

        def f(X):
            env = {c: X[:, i] for i, c in enumerate(coords)}
            return evaluate(exact, env, like=X[:, 0])

        return ReferenceSolution(
            method="Exact solution " + f"{spec.dependent} = {sp.sstr(exact)}",
            kind="analytical",
            settings={"expression": sp.sstr(exact)},
            func=f,
            runtime_s=0.0,
            estimated_error=0.0,
            notes=["Exact solution supplied with the problem and verified symbolically/numerically."],
        )


# ----------------------------------------------------------------------
# ODE initial value problems: scipy.integrate.solve_ivp
# ----------------------------------------------------------------------
class ScipyIVPReference(ReferenceSolver):
    name = "scipy_solve_ivp"

    def __init__(self, rtol: float = 1e-10, atol: float = 1e-12):
        self.rtol, self.atol = rtol, atol

    def _rhs(self, spec: EquationSpec):
        n = spec.order
        t = spec.coords[0]
        syms = [spec.deriv_symbols[(k,)] for k in range(n)]
        F = spec.explicit_rhs

        guard = _Guard()

        def rhs(tv, y):
            env = {t: np.asarray(tv, dtype=float)}
            env.update({s: np.asarray(y[k], dtype=float) for k, s in enumerate(syms)})
            top = float(evaluate(F, env, like=np.asarray(0.0)))
            guard.check(tv, y, top)
            return np.array(list(y[1:]) + [top], dtype=float)

        return rhs

    def _run(self, spec, method, rtol, atol):
        lo, hi = spec.bounds[0]
        y0 = [float(spec.condition("initial", order=k).value) for k in range(spec.order)]
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(self._rhs(spec), (lo, hi), y0, method=method, rtol=rtol, atol=atol, dense_output=True)
        except _Stop as stop:
            exc = ReferenceUnavailable(f"solve_ivp ({method}) stopped: {stop.reason}.")
            if stop.at is not None:
                exc.stopped_at = min(max(stop.at, lo), hi)
                exc.last_value = stop.size
            raise exc from None
        if sol.status != 0 or not sol.success:
            exc = ReferenceUnavailable(f"solve_ivp ({method}) failed: {sol.message}")
            if sol.t.size and sol.t[-1] < hi:
                exc.stopped_at = float(sol.t[-1])
                exc.last_value = float(np.max(np.abs(sol.y[:, -1])))
            raise exc
        grid = np.linspace(lo, hi, 2001)
        vals = sol.sol(grid)[0]
        if not np.all(np.isfinite(vals)):
            raise ReferenceUnavailable("solve_ivp produced non-finite values.")
        return sol, grid, vals

    def solve(self, spec: EquationSpec) -> ReferenceSolution:
        if spec.kind != ODE_IVP:
            raise ReferenceUnavailable("solve_ivp applies to initial value problems only.")
        method = "Radau" if spec.features.get("stiff") else "DOP853"
        t0 = time.perf_counter()
        sol, grid, vals = self._run(spec, method, self.rtol, self.atol)
        _, _, coarse = self._run(spec, method, self.rtol * 100, self.atol * 100)
        est = float(np.max(np.abs(vals - coarse)))
        rt = time.perf_counter() - t0
        return ReferenceSolution(
            method=f"SciPy solve_ivp ({method})",
            kind="numerical",
            settings={"method": method, "rtol": self.rtol, "atol": self.atol, "dense_output": True, "steps": int(sol.t.size)},
            func=lambda X: sol.sol(X[:, 0])[0],
            runtime_s=rt,
            estimated_error=est,
            notes=[f"Error estimated by re-solving with 100x looser tolerances (max difference {est:.2e})."],
        )


# ----------------------------------------------------------------------
# ODE boundary value problems: scipy.integrate.solve_bvp
# ----------------------------------------------------------------------
class ScipyBVPReference(ReferenceSolver):
    name = "scipy_solve_bvp"

    def __init__(self, tol: float = 1e-8, nodes: int = 201, max_nodes: int = 200000):
        self.tol, self.nodes, self.max_nodes = tol, nodes, max_nodes

    def _run(self, spec: EquationSpec, tol: float):
        t = spec.coords[0]
        s0, s1 = spec.deriv_symbols[(0,)], spec.deriv_symbols[(1,)]
        F = spec.explicit_rhs
        lo, hi = spec.bounds[0]
        left, right = spec.condition("boundary", "left"), spec.condition("boundary", "right")

        guard = _Guard()

        def fun(x, y):
            top = evaluate(F, {t: x, s0: y[0], s1: y[1]}, like=x)
            guard.check(x, y, top)
            return np.vstack([y[1], top])

        def bc(ya, yb):
            return np.array([ya[left.order] - float(left.value), yb[right.order] - float(right.value)])

        x = np.linspace(lo, hi, self.nodes)
        dir_vals = {c.side: float(c.value) for c in (left, right) if c.order == 0}
        a_val = dir_vals.get("left", dir_vals.get("right"))
        b_val = dir_vals.get("right", dir_vals.get("left"))
        y_init = np.vstack([a_val + (b_val - a_val) * (x - lo) / (hi - lo), np.full_like(x, (b_val - a_val) / (hi - lo))])
        try:
            with np.errstate(all="ignore"):
                sol = solve_bvp(fun, bc, x, y_init, tol=tol, max_nodes=self.max_nodes)
        except _Stop as stop:
            raise ReferenceUnavailable(f"solve_bvp stopped: {stop.reason}.") from None
        if sol.status != 0:
            raise ReferenceUnavailable(f"solve_bvp did not converge: {sol.message}")
        grid = np.linspace(lo, hi, 2001)
        vals = sol.sol(grid)[0]
        if not np.all(np.isfinite(vals)):
            raise ReferenceUnavailable("solve_bvp produced non-finite values.")
        return sol, vals

    def solve(self, spec: EquationSpec) -> ReferenceSolution:
        if spec.kind != ODE_BVP:
            raise ReferenceUnavailable("solve_bvp applies to boundary value problems only.")
        t0 = time.perf_counter()
        sol, vals = self._run(spec, self.tol)
        _, coarse = self._run(spec, self.tol * 100)
        est = float(np.max(np.abs(vals - coarse)))
        notes = [f"Error estimated by re-solving with a 100x looser tolerance (max difference {est:.2e})."]
        if not spec.linear:
            notes.append("Nonlinear BVPs can have several solutions; solve_bvp returns the one reached from a linear initial guess.")
        return ReferenceSolution(
            method="SciPy solve_bvp",
            kind="numerical",
            settings={"tol": self.tol, "initial_nodes": self.nodes, "final_nodes": int(sol.x.size), "max_nodes": self.max_nodes},
            func=lambda X: sol.sol(X[:, 0])[0],
            runtime_s=time.perf_counter() - t0,
            estimated_error=est,
            notes=notes,
        )


# ----------------------------------------------------------------------
# 1D parabolic PDEs: finite differences in x + stiff time integration (method of lines)
# ----------------------------------------------------------------------
class FiniteDifferenceReference(ReferenceSolver):
    name = "finite_difference_mol"

    def __init__(self, nx: Optional[int] = None, nt_out: int = 401, rtol: float = 1e-8, atol: float = 1e-10, max_nx: int = 4000):
        self.nx, self.nt_out, self.rtol, self.atol, self.max_nx = nx, nt_out, rtol, atol, max_nx

    def choose_nx(self, spec: EquationSpec) -> int:
        if self.nx:
            return int(self.nx)
        (a, b) = spec.bounds[0]
        L = b - a
        nx = 400
        U, D = spec.features.get("advection_speed", 0.0), spec.features.get("diffusion_min", 1.0)
        if U > 0:
            # keep the cell Peclet number U*dx/D at or below 0.25 (central differences stay non-oscillatory)
            nx = max(nx, int(np.ceil(L * U / (0.25 * D))))
        nx = int(min(self.max_nx, nx))
        return nx + (nx % 2)  # even, so that the coarse grid (nx/2) shares nodes

    def _run(self, spec: EquationSpec, nx: int):
        x_sym, t_sym = spec.coords
        k = spec.deriv_symbols
        su, sux, suxx = k[(0, 0)], k[(1, 0)], k[(2, 0)]
        F = spec.explicit_rhs
        (a, b), (t0, T1) = spec.bounds
        g = spec.condition("initial").value
        h0 = spec.condition("boundary", "left").value
        h1 = spec.condition("boundary", "right").value
        x = np.linspace(a, b, nx + 1)
        dx = x[1] - x[0]
        xi = x[1:-1]

        def bvals(tv):
            tt = np.asarray(tv, dtype=float)
            return float(evaluate(h0, {t_sym: tt}, like=tt)), float(evaluate(h1, {t_sym: tt}, like=tt))

        guard = _Guard()

        def rhs(tv, ui):
            left, right = bvals(tv)
            u = np.concatenate([[left], ui, [right]])
            ux = (u[2:] - u[:-2]) / (2 * dx)
            uxx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            env = {x_sym: xi, t_sym: np.full_like(xi, tv), su: ui, sux: ux, suxx: uxx}
            out = np.asarray(evaluate(F, env, like=xi), dtype=float)
            guard.check(tv, ui, out)
            return out

        u0 = np.asarray(evaluate(g, {x_sym: xi}, like=xi), dtype=float)
        n = xi.size
        sparsity = sparse.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])
        t_out = np.linspace(t0, T1, self.nt_out)
        if not np.all(np.isfinite(u0)):
            raise ReferenceUnavailable("The initial condition is not finite on the finite-difference grid.")
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(rhs, (t0, T1), u0, method="BDF", t_eval=t_out, rtol=self.rtol, atol=self.atol, jac_sparsity=sparsity)
        except _Stop as stop:
            # Unlike an ODE, a failure here does not prove that the solution blows up: an under-resolved
            # steep front makes the scheme fail too. So the PDE is not rejected; it just has no reference.
            where = "" if stop.at is None else f" near t = {min(max(stop.at, t0), T1):.4g}"
            raise ReferenceUnavailable(
                f"Finite-difference time integration stopped{where}: {stop.reason} (the solution blows up, reaches a "
                f"value where the equation is undefined, or is too steep for the {nx}-interval grid).") from None
        if not sol.success or sol.y.shape[1] != t_out.size:
            where = f" near t = {float(sol.t[-1]):.4g}" if sol.t.size else ""
            raise ReferenceUnavailable(
                f"Finite-difference time integration failed{where}: {sol.message} (the solution blows up or is too "
                f"steep for the {nx}-interval grid).")
        U = np.empty((nx + 1, t_out.size))
        U[1:-1, :] = sol.y
        for j, tv in enumerate(t_out):
            U[0, j], U[-1, j] = bvals(tv)
        if not np.all(np.isfinite(U)):
            raise ReferenceUnavailable("Finite-difference solution is not finite.")
        return x, t_out, U

    def solve(self, spec: EquationSpec) -> ReferenceSolution:
        if spec.kind != PDE_PARABOLIC:
            raise ReferenceUnavailable("The finite-difference reference applies to 1D parabolic PDEs only.")
        tic = time.perf_counter()
        nx = self.choose_nx(spec)
        x, t, U = self._run(spec, nx)
        xc, tc, Uc = self._run(spec, nx // 2)
        # coarse nodes coincide with every second fine node; 2nd-order scheme => error(fine) ~ diff / 3
        diff = np.abs(U[::2, :] - Uc)
        est = float(np.max(diff) / 3.0)
        spline = RectBivariateSpline(x, t, U, kx=3, ky=3, s=0)
        return ReferenceSolution(
            method=f"Finite differences (method of lines, {nx} intervals) + SciPy BDF",
            kind="numerical",
            settings={
                "nx_intervals": nx, "dx": float(x[1] - x[0]), "space_scheme": "2nd-order central differences",
                "time_integrator": "solve_ivp BDF", "rtol": self.rtol, "atol": self.atol,
                "output_times": self.nt_out, "interpolation": "bicubic spline (RectBivariateSpline)",
            },
            func=lambda X: spline.ev(X[:, 0], X[:, 1]),
            runtime_s=time.perf_counter() - tic,
            estimated_error=est,
            notes=[f"Error estimated from a grid-halving study (Richardson, 2nd order): about {est:.2e} (max abs)."],
        )


def numerical_reference(spec: EquationSpec) -> ReferenceSolver:
    if spec.kind == ODE_IVP:
        return ScipyIVPReference()
    if spec.kind == ODE_BVP:
        return ScipyBVPReference()
    if spec.kind == PDE_PARABOLIC:
        return FiniteDifferenceReference()
    raise ReferenceUnavailable(f"No numerical reference for problem kind {spec.kind!r}.")


def best_reference(spec: EquationSpec) -> ReferenceSolution:
    """Exact solution if one was supplied and verified, otherwise the numerical reference."""
    if spec.exact is not None:
        return AnalyticalReference().solve(spec)
    return numerical_reference(spec).solve(spec)
