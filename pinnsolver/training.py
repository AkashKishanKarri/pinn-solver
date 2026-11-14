"""PINN training (M5) and residual-based adaptive collocation (M6)."""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import qmc

from .pinn import PINN, condition_values, residual, to_tensor
from .spec import EquationSpec
from .strategy import TrainingConfig


def time_axis(spec: EquationSpec) -> Optional[int]:
    """Index of the time-like input: the variable of an ODE IVP, or t of a parabolic PDE; None for BVPs."""
    if spec.kind == "ode_ivp":
        return 0
    if spec.kind == "pde_parabolic_1d":
        return spec.variables.index("t")
    return None


class Cancelled(RuntimeError):
    """Raised (via the progress callback) when the user cancels a job."""


class Budget:
    """Wall-clock compute budget shared by all training attempts of one solve."""

    def __init__(self, max_time_s: float):
        self.max_time_s = float(max_time_s)
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def remaining(self) -> float:
        return self.max_time_s - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0


def sobol_points(bounds, n: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points in the box ``bounds`` (domain-covering initial set)."""
    d = len(bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # balance warning when n is not a power of two
        s = qmc.Sobol(d, scramble=True, seed=seed).random(n)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + s * (hi - lo)


class AdaptiveSampler:
    """Residual-based adaptive refinement (RAR-style, report Section 6.5).

    Evaluates |r| on fresh candidate points spread over the whole domain and
    *adds* the k largest to the retained global set (existing points are kept,
    so coverage of the rest of the domain is not lost).
    """

    def __init__(self, spec: EquationSpec, n_candidates: int, k: int, seed: int):
        self.spec = spec
        self.n_candidates = int(n_candidates)
        self.k = int(k)
        self.seed = int(seed)
        self.rounds = 0

    def refine(self, model: PINN, points: torch.Tensor, max_points: int) -> Tuple[torch.Tensor, dict]:
        room = max_points - points.shape[0]
        k = min(self.k, room)
        if k <= 0:
            return points, {"added": 0, "reason": "point budget reached"}
        cand = sobol_points(self.spec.bounds, self.n_candidates, seed=self.seed * 7919 + 104729 + self.rounds)
        self.rounds += 1
        Xc = to_tensor(cand, requires_grad=True)
        r = torch.cat([residual(model, self.spec, Xc[i:i + 4096], create_graph=False).detach().abs()
                       for i in range(0, Xc.shape[0], 4096)])
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0)
        idx = torch.topk(r, k).indices
        new = Xc.detach()[idx]
        info = {
            "added": int(k),
            "candidate_residual_mean": float(r.mean()),
            "candidate_residual_max": float(r.max()),
            "selected_residual_min": float(r[idx].min()),
            "total_points": int(points.shape[0] + k),
        }
        return torch.cat([points, new], dim=0), info


@dataclass
class TrainResult:
    adam_iterations: int = 0
    lbfgs_iterations: int = 0
    stop_reasons: List[str] = field(default_factory=list)
    final_loss: float = float("nan")
    loss_parts: Dict[str, float] = field(default_factory=dict)
    n_points: int = 0
    refinements: List[dict] = field(default_factory=list)
    time_s: float = 0.0
    nonfinite: bool = False

    def to_dict(self) -> dict:
        return {
            "adam_iterations": self.adam_iterations,
            "lbfgs_iterations": self.lbfgs_iterations,
            "stop_reasons": self.stop_reasons,
            "final_loss": self.final_loss,
            "loss_parts": self.loss_parts,
            "n_collocation_points": self.n_points,
            "refinements": self.refinements,
            "time_s": round(self.time_s, 3),
            "nonfinite_detected": self.nonfinite,
        }


class Trainer:
    """Builds the loss and training points and runs Adam (-> L-BFGS)."""

    LOG_EVERY = 50

    def __init__(self, spec: EquationSpec, cfg: TrainingConfig, budget: Budget,
                 progress: Optional[Callable[[dict], None]] = None):
        self.spec, self.cfg, self.budget = spec, cfg, budget
        self.progress = progress
        self.history: List[dict] = []  # training log (D3)
        self.points = to_tensor(sobol_points(spec.bounds, cfg.n_collocation, seed=cfg.seed))
        self.sampler = AdaptiveSampler(spec, cfg.rar_candidates, cfg.rar_k, cfg.seed)
        self.attempt = 0
        self._cond_sets: Optional[List[Tuple[object, torch.Tensor]]] = None
        self._deadline = math.inf

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def condition_sets(self) -> List[Tuple[object, torch.Tensor]]:
        if self._cond_sets is not None:
            return self._cond_sets
        spec, cfg = self.spec, self.cfg
        sets = []
        for c in spec.conditions:
            if spec.dim == 1:
                X = np.array([[c.at]])
            else:
                idx = spec.variables.index(c.var)
                other = 1 - idx
                lo, hi = spec.bounds[other]
                line = sobol_points([(lo, hi)], cfg.n_condition_points, seed=cfg.seed + 31 + idx * 7 + (c.side == "right"))[:, 0]
                line = np.concatenate([line, [lo, hi]])
                X = np.zeros((line.size, 2))
                X[:, idx] = c.at
                X[:, other] = line
            sets.append((c, to_tensor(X)))
        self._cond_sets = sets
        return sets

    def loss(self, model: PINN, use_gpinn: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        X = self.points.clone().requires_grad_(True)
        scale = getattr(model, "scale", 1.0)
        r = residual(model, self.spec, X, create_graph=True)
        if scale != 1.0:
            r = r / scale  # loss in units of the solution scale (see pinn.output_scale)
        l_res = torch.mean(r**2)
        total = cfg.weights["residual"] * l_res
        parts = {"residual": float(l_res.detach())}
        if use_gpinn:
            gr = torch.autograd.grad(r, X, torch.ones_like(r), create_graph=True)[0]
            l_g = torch.mean(torch.sum(gr**2, dim=1))
            total = total + cfg.lambda_g * l_g
            parts["gpinn"] = float(l_g.detach())
        if not model.hard:
            for role in ("initial", "boundary"):
                terms = []
                for c, Xc in self.condition_sets():
                    if c.role != role:
                        continue
                    val, target = condition_values(model, self.spec, c, Xc.clone().requires_grad_(True), create_graph=True)
                    diff = val - target
                    terms.append(torch.mean((diff if scale == 1.0 else diff / scale) ** 2))
                if terms:
                    l_c = torch.stack(terms).sum()
                    total = total + cfg.weights[role] * l_c
                    parts[role] = float(l_c.detach())
        parts["total"] = float(total.detach())
        return total, parts

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _log(self, phase: str, it: int, parts: Dict[str, float], lr: Optional[float]) -> None:
        entry = {
            "attempt": self.attempt,
            "phase": phase,
            "iteration": it,
            "time_s": round(self.budget.elapsed(), 3),
            "n_points": int(self.points.shape[0]),
            "lr": lr,
            **{f"loss_{k}": v for k, v in parts.items()},
        }
        self.history.append(entry)
        if self.progress is not None:
            self.progress(entry)

    def train(self, model: PINN, adam_iters: int, lbfgs_iters: int, use_gpinn: bool = False,
              adaptive: bool = False, lr: Optional[float] = None, time_cap: Optional[float] = None,
              adam_frac: float = 0.6, new_attempt: bool = True) -> TrainResult:
        """Run one training attempt (from the model's current weights).

        ``time_cap`` limits this attempt (seconds, never beyond the shared budget). When L-BFGS
        follows, Adam may use at most ``adam_frac`` of the attempt's time so that L-BFGS is not starved.
        """
        if new_attempt:
            self.attempt += 1
        cfg = self.cfg
        res = TrainResult()
        t_start = time.perf_counter()
        avail = self.budget.remaining() if time_cap is None else min(time_cap, self.budget.remaining())
        self._deadline = t_start + max(0.0, avail)
        adam_deadline = t_start + max(0.0, avail) * (adam_frac if lbfgs_iters > 0 else 1.0)
        lr0 = cfg.lr if lr is None else lr
        lr_final = min(cfg.lr_final, lr0)
        params = [p for p in model.parameters() if p.requires_grad]
        good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # ---------------- Adam ----------------
        if adam_iters > 0:
            opt = torch.optim.Adam(params, lr=lr0)
            gamma = (lr_final / lr0) ** (1.0 / max(1, adam_iters))
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
            best_hist: List[Tuple[int, float]] = []
            best = math.inf
            window_start = 0
            last_refine = 0
            parts: Dict[str, float] = {}
            for it in range(1, adam_iters + 1):
                opt.zero_grad(set_to_none=True)
                loss, parts = self.loss(model, use_gpinn)
                if not torch.isfinite(loss):
                    res.nonfinite = True
                    res.stop_reasons.append(f"Adam: non-finite loss at iteration {it}; weights restored to last finite checkpoint")
                    model.load_state_dict(good_state)
                    break
                loss.backward()
                grads_ok = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                if not grads_ok:
                    res.nonfinite = True
                    res.stop_reasons.append(f"Adam: non-finite gradient at iteration {it}; weights restored to last finite checkpoint")
                    model.load_state_dict(good_state)
                    break
                opt.step()
                sched.step()
                res.adam_iterations = it
                best = min(best, parts["total"])
                if it % 100 == 0:
                    good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_hist.append((it, best))
                if it % self.LOG_EVERY == 0 or it == 1:
                    self._log("adam", it, parts, opt.param_groups[0]["lr"])
                if time.perf_counter() >= adam_deadline:
                    res.stop_reasons.append(f"Adam: time allocation exhausted at iteration {it}")
                    break
                # Scheduled refinement (modes C/D with adaptive sampling).
                if adaptive and it - last_refine >= cfg.rar_every and self.points.shape[0] < cfg.max_points:
                    self._refine(model, res, it)
                    last_refine = it
                    window_start, best, best_hist = it, math.inf, []
                    continue
                # Stall detector: relative improvement of the best loss over the last window.
                if it - window_start >= cfg.stall_window and it % 100 == 0:
                    old = [b for (i, b) in best_hist if i <= it - cfg.stall_window]
                    if old and best > (1.0 - cfg.stall_tol) * old[-1]:
                        if adaptive and self.points.shape[0] < cfg.max_points:
                            self._refine(model, res, it)
                            last_refine = it
                            window_start, best, best_hist = it, math.inf, []
                            continue
                        if not cfg.stop_on_stall:
                            # fixed-recipe baselines (A-C) keep training until their iteration/time budget
                            window_start, best, best_hist = it, math.inf, []
                            continue
                        res.stop_reasons.append(f"Adam: stalled at iteration {it} (best loss improved < {cfg.stall_tol:.0%} over {cfg.stall_window} iterations)")
                        break
            else:
                res.stop_reasons.append(f"Adam: completed {adam_iters} iterations")

        # ---------------- L-BFGS ----------------
        if lbfgs_iters > 0 and not res.nonfinite and time.perf_counter() < self._deadline:
            self._lbfgs(model, params, lbfgs_iters, use_gpinn, res)

        with torch.enable_grad():
            loss, parts = self.loss(model, use_gpinn)
        res.final_loss = parts["total"]
        res.loss_parts = parts
        res.n_points = int(self.points.shape[0])
        res.time_s = time.perf_counter() - t_start
        self._log("final", res.adam_iterations + res.lbfgs_iterations, parts, None)
        return res

    def train_marching(self, model: PINN, windows: int, adam_iters: int, lbfgs_iters: int,
                       lr: Optional[float] = None, time_cap: Optional[float] = None) -> TrainResult:
        """Causal time-marching attempt (time decomposition, rule R9).

        Training uses only the collocation points with t <= t0 + k T / K for k = 1..K, continuing from
        the previous window's weights, so that the solution is learnt forwards in time. The last window
        is the full domain and is followed by L-BFGS. Validation is unchanged (whole domain, fresh points).
        """
        ax = time_axis(self.spec)
        if ax is None:
            raise ValueError("time-marching needs a time-dependent problem")
        self.attempt += 1
        t0, t1 = self.spec.bounds[ax]
        full = self.points
        tic = time.perf_counter()
        cap = self.budget.remaining() if time_cap is None else min(time_cap, self.budget.remaining())
        total = TrainResult()
        per = max(200, adam_iters // windows)
        try:
            for k in range(1, windows + 1):
                last = k == windows
                t_end = t0 + (t1 - t0) * k / windows
                self.points = full if last else full[full[:, ax] <= t_end]
                left = cap - (time.perf_counter() - tic)
                share = left if last else left * 0.6 / (windows - k + 1)
                r = self.train(model, per, lbfgs_iters if last else 0, lr=lr, time_cap=max(0.0, share), new_attempt=False)
                total.adam_iterations += r.adam_iterations
                total.lbfgs_iterations += r.lbfgs_iterations
                total.stop_reasons += [f"window {k}/{windows} (t <= {t_end:.4g}, {self.points.shape[0]} points): {x}" for x in r.stop_reasons]
                total.nonfinite = total.nonfinite or r.nonfinite
                total.final_loss, total.loss_parts = r.final_loss, r.loss_parts
                if r.nonfinite or self.budget.expired():
                    break
        finally:
            self.points = full
        total.n_points = int(full.shape[0])
        total.time_s = time.perf_counter() - tic
        return total

    def _lbfgs(self, model: PINN, params, max_iters: int, use_gpinn: bool, res: TrainResult) -> None:
        chunk = 50
        # max_eval is set high so that a step returns early only when L-BFGS's own tolerances are met
        # (the default 1.25 * max_iter is easily used up by strong-Wolfe line searches).
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, max_eval=chunk * 25, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        state = {"parts": {}}

        def closure():
            opt.zero_grad(set_to_none=True)
            loss, parts = self.loss(model, use_gpinn)
            state["parts"] = parts
            if torch.isfinite(loss):
                loss.backward()
            return loss

        done = 0
        prev = math.inf
        good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        while done < max_iters:
            before = opt.state[opt._params[0]].get("n_iter", 0) if opt.state.get(opt._params[0]) else 0
            loss = opt.step(closure)
            after = opt.state[opt._params[0]].get("n_iter", 0)
            step_iters = max(1, after - before)
            done += step_iters
            res.lbfgs_iterations = done
            cur = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            if not math.isfinite(cur) or not all(torch.isfinite(p).all() for p in params):
                res.nonfinite = True
                model.load_state_dict(good_state)
                res.stop_reasons.append(f"L-BFGS: non-finite values after {done} iterations; weights restored")
                return
            good_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self._log("lbfgs", res.adam_iterations + done, state["parts"], None)
            if time.perf_counter() >= self._deadline:
                res.stop_reasons.append(f"L-BFGS: time allocation exhausted after {done} iterations")
                return
            if step_iters < chunk or abs(prev - cur) <= 1e-12 * max(1.0, abs(cur)):
                res.stop_reasons.append(f"L-BFGS: converged after {done} iterations (no further decrease)")
                return
            prev = cur
        res.stop_reasons.append(f"L-BFGS: completed {done} iterations")

    def _refine(self, model: PINN, res: TrainResult, it: int) -> None:
        self.points, info = self.sampler.refine(model, self.points, self.cfg.max_points)
        info["at_iteration"] = it
        info["attempt"] = self.attempt
        res.refinements.append(info)

    def refine(self, model: PINN) -> dict:
        """One refinement round triggered by the validation step (mode D, rule R8)."""
        self.points, info = self.sampler.refine(model, self.points, self.cfg.max_points)
        info["attempt"] = self.attempt
        info["trigger"] = "validation"
        return info

    def retrain(self, model: PINN, adam_iters: int, lbfgs_iters: int, use_gpinn: bool = False,
                adaptive: bool = False, lr: Optional[float] = None, time_cap: Optional[float] = None) -> TrainResult:
        """Continue training from the current weights (after refinement or a configuration change)."""
        return self.train(model, adam_iters, lbfgs_iters, use_gpinn=use_gpinn, adaptive=adaptive, lr=lr, time_cap=time_cap)
