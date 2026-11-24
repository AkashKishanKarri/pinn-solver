"""Strategy selector (M4) and the rule table (data store D2).

The rules are hand-written and fully visible (report Section 6.3, Table 6.1).
A rule only chooses a *starting* configuration; the validation stage decides
whether it worked.

Configurations compared in the experiments (report Table 6.2):

====  ===============================================================
A     standard PINN: penalty IC/BC, uniform points, Adam
B     A + hard constraints where a valid construction exists
C     B + residual-based adaptive collocation (RAR)
D     proposed: rule-based selection, C, Adam -> L-BFGS, optional
      gPINN, validation-driven refinement/retries, numerical fallback
E     classical numerical solver only
====  ===============================================================

A, B, C and D share the architecture and point budget chosen by the rules,
so that the ablation isolates the component under test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .pinn import hard_constraint_plan
from .spec import ODE_BVP, ODE_IVP, PDE_PARABOLIC, EquationSpec

MODES = ("A", "B", "C", "D", "E")
MODE_NAMES = {
    "A": "Standard PINN (baseline)",
    "B": "PINN + hard constraints",
    "C": "PINN + hard constraints + adaptive sampling",
    "D": "Proposed adaptive solver",
    "E": "Classical numerical solver",
}

# Pre-declared acceptance thresholds (fixed before any run; report Section 6.7).
DEFAULT_THRESHOLDS = {
    ODE_IVP: {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3},
    ODE_BVP: {"rel_l2": 1e-3, "condition_max": 1e-4, "residual_rms": 1e-3},
    PDE_PARABOLIC: {"rel_l2": 1e-2, "condition_max": 1e-3, "residual_rms": 1e-2},
}

# Default compute budgets (wall-clock seconds for all PINN training of one solve).
DEFAULT_TIME_BUDGET = {ODE_IVP: 60.0, ODE_BVP: 60.0, PDE_PARABOLIC: 180.0}
MAX_TIME_BUDGET = 900.0
MAX_POINTS = 20000


@dataclass
class TrainingConfig:
    mode: str = "D"
    seed: int = 0
    hidden: List[int] = field(default_factory=lambda: [32, 32, 32])
    hard_constraints: bool = True
    n_collocation: int = 256  # initial interior points
    max_points: int = 256  # point budget (interior collocation points)
    n_condition_points: int = 64  # per boundary / initial line (PDE soft constraints)
    adaptive: bool = False
    rar_candidates: int = 4096
    rar_k: int = 64
    rar_every: int = 1000  # Adam iterations between refinement rounds (modes C/D)
    use_gpinn: bool = False
    gpinn_candidate: bool = False  # selector allows gPINN to be tried on retry
    lambda_g: float = 0.01
    weights: Dict[str, float] = field(default_factory=lambda: {"residual": 1.0, "boundary": 1.0, "initial": 1.0})
    lr: float = 1e-3
    lr_final: float = 1e-4
    adam_iters: int = 5000
    lbfgs_iters: int = 0
    stall_window: int = 1000
    stall_tol: float = 0.01
    stop_on_stall: bool = True  # D: a stalled Adam phase hands over to L-BFGS; A-C keep training
    time_marching_windows: int = 4
    max_time_s: float = 60.0
    max_attempts: int = 1
    fallback: bool = False
    thresholds: Dict[str, float] = field(default_factory=dict)
    n_validation: int = 2000
    reasons: List[str] = field(default_factory=list)
    rules_fired: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("reasons", None)
        d.pop("rules_fired", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def copy(self) -> "TrainingConfig":
        return copy.deepcopy(self)


# ----------------------------------------------------------------------
# Rule table (D2). Each rule: id, problem feature, strategy, caution.
# ----------------------------------------------------------------------
RULE_TABLE = [
    {"id": "R1", "feature": "Smooth, low-dimensional ODE", "strategy": "Small PINN (3x32) with the initial condition built in; Adam then L-BFGS", "caution": "A numerical solver may be faster; it is always run as the reference/fallback."},
    {"id": "R2", "feature": "Boundary value problem with Dirichlet conditions", "strategy": "Trial solution that satisfies both conditions exactly", "caution": "Only where an exact form exists; Neumann conditions fall back to penalty terms."},
    {"id": "R3", "feature": "Oscillatory solution (> 2 cycles)", "strategy": "Wider/deeper network (4x64), 128 collocation points per cycle (at least 512), double default time budget", "caution": "Spectral bias: high frequencies train slowly."},
    {"id": "R4", "feature": "High-frequency solution (> 8 cycles)", "strategy": "As R3 with a larger budget; numerical fallback expected", "caution": "Fourier features are not implemented in this version."},
    {"id": "R5", "feature": "Stiff / fast transient (stiffness index > 50)", "strategy": "Hard IC, adaptive points from the start, compare with an implicit numerical solver", "caution": "PINNs may struggle here; fallback enabled."},
    {"id": "R6", "feature": "1D parabolic PDE (smooth)", "strategy": "Hard IC/BC by transfinite interpolation when the data are compatible; 4x32 network", "caution": "Corner-incompatible data disables hard constraints."},
    {"id": "R7", "feature": "Steep gradients (Peclet number > 100)", "strategy": "Global points + residual-based refinement from the start; larger network; gPINN tried on retry", "caution": "gPINN adds higher-derivative cost; it is kept only if validation improves."},
    {"id": "R8", "feature": "Validation fails within budget", "strategy": "Refine collocation points and continue training; then R9 (time-dependent problems), gPINN (R7 problems) or a wider network; then numerical fallback", "caution": "Stay within the compute budget and report every attempt."},
    {"id": "R9", "feature": "Time-dependent problem (IVP or parabolic PDE) still failing after refinement", "strategy": "Time decomposition: causal time-marching over growing time windows from a fresh initialisation", "caution": "Costs a full retraining; kept only if validation improves."},
]


class StrategySelector:
    """Applies :data:`RULE_TABLE` to an :class:`EquationSpec`."""

    rule_table = RULE_TABLE

    def select(self, spec: EquationSpec, mode: str = "D", seed: int = 0, time_budget: Optional[float] = None,
               thresholds: Optional[Dict[str, float]] = None) -> TrainingConfig:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose one of {', '.join(MODES)}.")
        cfg = TrainingConfig(mode=mode, seed=int(seed))
        cfg.thresholds = dict(DEFAULT_THRESHOLDS[spec.kind])
        if thresholds:
            for k, v in thresholds.items():
                if k in cfg.thresholds and v is not None and float(v) > 0:
                    cfg.thresholds[k] = float(v)
        budget = DEFAULT_TIME_BUDGET[spec.kind] if time_budget is None else float(time_budget)
        if time_budget is None and spec.features.get("oscillatory"):
            budget *= 2  # R3: oscillatory solutions need more points and therefore more time
            cfg.reasons.append(f"[R3] default time budget doubled to {budget:.0f} s for an oscillatory solution.")
        cfg.max_time_s = max(1.0, min(MAX_TIME_BUDGET, budget))
        self._apply_rules(spec, cfg)
        self._apply_mode(spec, cfg, mode)
        return cfg

    # ------------------------------------------------------------------
    def _fire(self, cfg: TrainingConfig, rule_id: str, reason: str) -> None:
        cfg.rules_fired.append(rule_id)
        cfg.reasons.append(f"[{rule_id}] {reason}")

    def _apply_rules(self, spec: EquationSpec, cfg: TrainingConfig) -> None:
        f = spec.features
        hard_ok, hard_why = hard_constraint_plan(spec)
        cfg.hard_constraints = hard_ok
        if spec.kind in (ODE_IVP, ODE_BVP):
            cycles = float(f.get("oscillation_cycles", 0.0) or 0.0)
            cfg.hidden = [32, 32, 32]
            cfg.n_collocation = 128
            cfg.adam_iters, cfg.lbfgs_iters = 4000, 1000
            cfg.n_validation = 2000
            if spec.kind == ODE_IVP:
                self._fire(cfg, "R1", f"{spec.subtype}: small network, initial condition(s) built into the trial solution. {hard_why}")
            else:
                self._fire(cfg, "R2", hard_why)
            if f.get("oscillatory"):
                cfg.hidden = [64, 64, 64, 64]
                cfg.n_collocation = int(min(4096, max(512, 128 * math.ceil(cycles))))
                cfg.adam_iters = 6000
                self._fire(cfg, "R3", f"about {cycles:.1f} oscillation cycles: 4x64 network and {cfg.n_collocation} collocation points.")
            if f.get("high_frequency"):
                cfg.adam_iters = 10000
                self._fire(cfg, "R4", f"{cycles:.1f} cycles is high-frequency for a plain tanh PINN; the numerical fallback is likely to be needed.")
            if f.get("stiff"):
                cfg.n_collocation = max(cfg.n_collocation, 256)
                cfg.adaptive = True
                self._fire(cfg, "R5", f"stiffness index {float(f['stiffness_index']):.0f} (> 50): fast transient; adaptive points from the start; implicit solver (Radau) used as reference.")
            cfg.max_points = int(min(MAX_POINTS, cfg.n_collocation * 4))
            cfg.rar_k = max(16, cfg.n_collocation // 4)
            cfg.rar_candidates = 4096
            cfg.rar_every = 1000
        else:  # PDE_PARABOLIC
            cfg.hidden = [32, 32, 32, 32]
            cfg.n_collocation = 2048
            cfg.n_condition_points = 128
            cfg.adam_iters, cfg.lbfgs_iters = 5000, 1500
            cfg.n_validation = 4096
            self._fire(cfg, "R6", hard_why)
            if f.get("steep_gradient"):
                cfg.hidden = [48, 48, 48, 48, 48]
                cfg.n_collocation = 4096
                cfg.adaptive = True
                cfg.gpinn_candidate = True
                cfg.adam_iters, cfg.lbfgs_iters = 6000, 3000
                self._fire(cfg, "R7", f"Peclet number {float(f['peclet_number']):.0f} (> 100) suggests a steep front: 5x48 network, residual-based refinement from the start; gPINN may be tried on retry.")
            cfg.max_points = int(min(MAX_POINTS, cfg.n_collocation * 2))
            cfg.rar_k = 256
            cfg.rar_candidates = 16384
            cfg.rar_every = 1000

    def _apply_mode(self, spec: EquationSpec, cfg: TrainingConfig, mode: str) -> None:
        hard_ok, _ = hard_constraint_plan(spec)
        if mode == "E":
            cfg.reasons.append("Mode E: classical numerical solver only (no PINN training).")
            return
        # A, B, C: fixed recipes for the ablation; same architecture/budgets as D.
        if mode in ("A", "B", "C"):
            total = cfg.adam_iters + cfg.lbfgs_iters
            cfg.lbfgs_iters = 0
            cfg.adam_iters = total * 3  # Adam-only runs are stopped by the shared time budget or stall detector
            cfg.use_gpinn = False
            cfg.gpinn_candidate = False
            cfg.max_attempts = 1
            cfg.fallback = False
            cfg.stop_on_stall = False
            cfg.hard_constraints = (mode != "A") and hard_ok
            if mode in ("A", "B"):
                cfg.adaptive = False
                cfg.n_collocation = cfg.max_points  # uniform points use the whole point budget
            else:
                cfg.adaptive = True
                cfg.n_collocation = max(cfg.rar_k, cfg.max_points // 2)
            overrides = ["Adam only (no L-BFGS)", "one attempt, no numerical fallback"]
            if mode == "A" and hard_ok:
                overrides.insert(0, "all conditions imposed as penalty terms, NOT built into the trial solution (overrides the rule above)")
            overrides.append("adaptive collocation" if mode == "C" else "fixed uniform collocation points")
            cfg.reasons.append(f"Mode {mode}: {MODE_NAMES[mode]} (fixed recipe for the ablation study): " + "; ".join(overrides) + ".")
            return
        # D: proposed solver.
        cfg.hard_constraints = hard_ok
        cfg.max_attempts = 4
        cfg.fallback = True
        cfg.reasons.append("Mode D: Adam then L-BFGS; validation on fresh points after each attempt; refine/retry while budget remains; numerical fallback if the thresholds are not met.")
