"""Independent validation (M7).

All validation points are drawn from a random stream that is never used for
training (different generator and seed from the Sobol training/candidate
points). Metrics (report Eqs. 6.5-6.6):

* RMS and maximum residual on Nv fresh interior points;
* initial/boundary-condition errors, reported per condition;
* relative L2 error  ||u - u_ref||_2 / (||u_ref||_2 + eps), eps = 1e-12, and
  maximum absolute error, both on a uniform evaluation grid
  (1001 points in 1D, 101 x 101 in 2D) - the documented discretisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .pinn import condition_values, residual, to_tensor
from .references import ReferenceSolution
from .spec import EquationSpec

EPS = 1e-12
ZERO_REFERENCE = 1e-12  # RMS size below which a reference counts as identically zero
GRID_1D = 1001
GRID_2D = 101


def evaluation_grid(spec: EquationSpec) -> np.ndarray:
    if spec.dim == 1:
        lo, hi = spec.bounds[0]
        return np.linspace(lo, hi, GRID_1D)[:, None]
    (a, b), (t0, t1) = spec.bounds
    xs, ts = np.linspace(a, b, GRID_2D), np.linspace(t0, t1, GRID_2D)
    X, T = np.meshgrid(xs, ts, indexing="ij")
    return np.column_stack([X.ravel(), T.ravel()])


@dataclass
class ValidationResult:
    passed: bool
    criterion: str  # 'reference' | 'residual-only'
    metrics: Dict[str, Optional[float]]
    thresholds: Dict[str, float]
    conditions: List[dict]
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    n_validation_points: int = 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "criterion": self.criterion,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "conditions": self.conditions,
            "failures": self.failures,
            "notes": self.notes,
            "n_validation_points": self.n_validation_points,
        }


class Validator:
    def __init__(self, spec: EquationSpec, thresholds: Dict[str, float], n_validation: int, seed: int,
                 scale: float = 1.0):
        self.spec = spec
        # ``scale`` is the size of the solution (pinn.output_scale). The relative-L2 threshold is already
        # scale-free; the absolute ones (condition error, residual) are stated for an O(1) solution and are
        # multiplied by it, so that the same relative accuracy is asked of small and large solutions.
        self.scale = float(scale)
        self.thresholds = dict(thresholds)
        if self.scale != 1.0:
            for k in ("condition_max", "residual_rms"):
                if k in self.thresholds:
                    self.thresholds[k] = self.thresholds[k] * self.scale
        self.n_validation = int(n_validation)
        # Independent stream: numpy PCG64 with an offset seed; training uses scrambled Sobol.
        rng = np.random.default_rng(seed + 1_000_003)
        lo = np.array([b[0] for b in spec.bounds])
        hi = np.array([b[1] for b in spec.bounds])
        self.points = lo + rng.random((self.n_validation, spec.dim)) * (hi - lo)
        self.cond_points = []
        for c in spec.conditions:
            if spec.dim == 1:
                X = np.array([[c.at]])
            else:
                idx = spec.variables.index(c.var)
                other = 1 - idx
                lo_o, hi_o = spec.bounds[other]
                line = lo_o + rng.random(201) * (hi_o - lo_o)
                X = np.zeros((201, 2))
                X[:, idx] = c.at
                X[:, other] = line
            self.cond_points.append((c, X))
        self.grid = evaluation_grid(spec)

    # ------------------------------------------------------------------
    def residual_values(self, model) -> np.ndarray:
        out = []
        for i in range(0, self.points.shape[0], 4096):
            X = to_tensor(self.points[i:i + 4096], requires_grad=True)
            out.append(residual(model, self.spec, X, create_graph=False).detach().numpy())
        return np.concatenate(out)

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate([model(to_tensor(X[i:i + 8192])).numpy() for i in range(0, X.shape[0], 8192)])

    def check(self, model, ref: Optional[ReferenceSolution]) -> ValidationResult:
        """Validate a trained PINN on fresh points and against ``ref`` if available."""
        r = self.residual_values(model)
        u_grid = self.predict(model, self.grid)
        conds = []
        for c, X in self.cond_points:
            val, target = condition_values(model, self.spec, c, to_tensor(X, requires_grad=True), create_graph=False)
            err = np.abs(val.detach().numpy() - target.detach().numpy())
            conds.append({
                "condition": c.text, "role": c.role, "type": c.kind,
                "built_in": bool(getattr(model, "hard", False)),
                "max_abs_error": float(np.max(err)) if np.all(np.isfinite(err)) else float("inf"),
                "rms_error": float(np.sqrt(np.mean(err**2))) if np.all(np.isfinite(err)) else float("inf"),
            })
        return self._assemble(u_grid, r, conds, ref)

    def check_values(self, u_grid: np.ndarray, ref: Optional[ReferenceSolution], conds: Optional[List[dict]] = None) -> ValidationResult:
        """Validate a solution given only by its values on the evaluation grid (numerical fallback)."""
        return self._assemble(u_grid, None, conds or [], ref)

    def _assemble(self, u_grid, r, conds, ref) -> ValidationResult:
        th = self.thresholds
        metrics: Dict[str, Optional[float]] = {
            "residual_rms": None, "residual_max": None, "condition_max": None,
            "rel_l2": None, "max_abs_error": None,
        }
        failures: List[str] = []
        notes: List[str] = []
        finite = bool(np.all(np.isfinite(u_grid)))
        if r is not None:
            if np.all(np.isfinite(r)):
                metrics["residual_rms"] = float(np.sqrt(np.mean(r**2)))
                metrics["residual_max"] = float(np.max(np.abs(r)))
                metrics["residual_concentration"] = float(metrics["residual_max"] / max(metrics["residual_rms"], 1e-300))
            else:
                finite = False
        if conds:
            metrics["condition_max"] = float(max(c["max_abs_error"] for c in conds))
        if not finite:
            failures.append("Solution or residual contains NaN/Inf.")
        criterion = "residual-only"
        u_ref = ref(self.grid) if ref is not None else None
        if u_ref is not None and not np.all(np.isfinite(u_ref)):
            # NaN compares false with every threshold, so a non-finite reference must never count as a pass.
            failures.append("The reference solution is not finite on the evaluation grid, so the error cannot be measured.")
            criterion = "reference"
        elif ref is not None and finite:
            criterion = "reference"
            diff = u_grid - u_ref
            metrics["max_abs_error"] = float(np.max(np.abs(diff)))
            ref_rms = float(np.sqrt(np.mean(u_ref**2)))
            if ref_rms < ZERO_REFERENCE * self.scale:
                # A relative error against a zero reference is meaningless (it divides by ~0).
                metrics["rel_l2"] = float(np.sqrt(np.mean(diff**2)) / self.scale)
                notes.append(
                    "The reference solution is zero everywhere, so a relative error is undefined; the value reported "
                    "as the relative L2 error is the absolute RMS error instead.")
            else:
                metrics["rel_l2"] = float(np.linalg.norm(diff) / (np.linalg.norm(u_ref) + EPS))
            scale = float(np.max(np.abs(u_ref))) or 1.0
            if ref.estimated_error is not None and ref.estimated_error > 0.1 * th["rel_l2"] * scale:
                notes.append(
                    f"The reference's own estimated error ({ref.estimated_error:.2e}) is not small compared with the "
                    "threshold; errors near the threshold cannot be resolved reliably."
                )
            if not metrics["rel_l2"] <= th["rel_l2"]:
                failures.append(f"relative L2 error {metrics['rel_l2']:.3e} > threshold {th['rel_l2']:.1e}")
        elif finite and r is not None:
            notes.append("No reference solution is available: the true error is NOT established; only residual and condition checks were made.")
            if not metrics["residual_rms"] <= th["residual_rms"]:
                failures.append(f"RMS residual {metrics['residual_rms']:.3e} > threshold {th['residual_rms']:.1e}")
        if metrics["condition_max"] is not None and not metrics["condition_max"] <= th["condition_max"]:
            failures.append(f"max condition error {metrics['condition_max']:.3e} > threshold {th['condition_max']:.1e}")
        if self.scale != 1.0:
            notes.append(f"Absolute thresholds are scaled by the size of the data ({self.scale:.3g}).")
        if r is not None:
            notes.append("A small residual is not a proof of correctness; it is reported alongside the reference error.")
        passed = finite and not failures and (criterion == "reference" or r is not None)
        return ValidationResult(
            passed=passed, criterion=criterion, metrics=metrics, thresholds=dict(th), conditions=conds,
            failures=failures, notes=notes, n_validation_points=int(self.points.shape[0]),
        )
