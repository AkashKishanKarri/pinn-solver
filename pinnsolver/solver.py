"""Solver orchestration: the validation-driven control loop (report Fig. 6.1).

    parse -> select strategy -> build trial solution -> train -> validate
          -> (refine | gPINN | alternative configuration)* -> fallback -> report
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .parser import Parser, ValidationError
from .pinn import PINN, output_scale
from .references import ReferenceSolution, ReferenceUnavailable, best_reference, numerical_reference
from .report import ReportGenerator
from .spec import EquationSpec, ProblemInput
from .strategy import StrategySelector, TrainingConfig
from .training import Budget, Trainer, time_axis
from .validation import Validator

ABLATIONS = ("no_hard", "no_adaptive", "no_gpinn", "no_retries", "no_lbfgs", "no_fallback")
MIN_RETRY_SECONDS = 2.0
INITIAL_ATTEMPT_SHARE = 0.6
REFINE_KEEP_GOING = 0.8
# Continuation attempts start near an L-BFGS minimum: a long Adam phase at the base rate would knock the
# network out of it, so only a short, low-rate Adam phase precedes L-BFGS.
CONTINUE_ADAM_ITERS = 300
CONTINUE_LR_FACTOR = 0.01
POLISH_MIN_SECONDS = 5.0


@dataclass
class Attempt:
    number: int
    action: str
    description: str
    config: Dict[str, object]
    train: Dict[str, object]
    validation: Dict[str, object]
    time_s: float

    def to_dict(self) -> dict:
        return {
            "attempt": self.number, "action": self.action, "description": self.description,
            "config": self.config, "training": self.train, "validation": self.validation,
            "time_s": round(self.time_s, 3),
        }


def _score(v) -> float:
    m = v.metrics
    if m.get("rel_l2") is not None:
        return m["rel_l2"]
    if m.get("residual_rms") is not None:
        return m["residual_rms"]
    return math.inf


class Solver:
    """End-to-end solve of one problem in one mode (A-E)."""

    def __init__(self, progress: Optional[Callable[[dict], None]] = None, runs_dir: Optional[str] = "runs",
                 save: bool = True, threads: int = 4):
        self.progress = progress
        self.runs_dir = runs_dir
        self.save = save
        self.threads = threads

    def _emit(self, **event) -> None:
        if self.progress is not None:
            self.progress(event)

    # ------------------------------------------------------------------
    def solve(self, problem: ProblemInput, mode: str = "D", seed: int = 0, time_budget: Optional[float] = None,
              thresholds: Optional[Dict[str, float]] = None, ablate: Optional[List[str]] = None) -> dict:
        t_total = time.perf_counter()
        torch.set_num_threads(self.threads)
        ablate = list(ablate or [])
        bad = [a for a in ablate if a not in ABLATIONS]
        if bad:
            raise ValueError(f"Unknown ablation(s) {bad}; allowed: {', '.join(ABLATIONS)}")

        # Stages 1-3: parse, validate, represent.
        self._emit(stage="parse")
        spec = Parser(problem).validate()
        warnings_: List[str] = list(spec.warnings)

        # Stage 4: strategy.
        cfg = StrategySelector().select(spec, mode=mode, seed=seed, time_budget=time_budget, thresholds=thresholds)
        self._apply_ablations(cfg, ablate)
        self._emit(stage="strategy", reasons=cfg.reasons)

        # Reference for validation (never used in training).
        self._emit(stage="reference")
        ref: Optional[ReferenceSolution] = None
        ref_failure: Optional[str] = None
        try:
            ref = best_reference(spec)
        except ReferenceUnavailable as exc:
            if exc.stopped_at is not None:
                # The classical integrator could not pass this point: the solution does not exist on the
                # whole domain, so training a PINN on it could only produce a wrong answer.
                ax = time_axis(spec) or 0
                var = spec.variables[ax]
                start, end = spec.bounds[ax]
                if exc.stopped_at <= start + 1e-12 * (end - start):
                    raise ValidationError(
                        f"The equation is undefined at the initial condition ({var} = {start:g}): its right-hand side "
                        f"is not a finite real number for the given initial data (for example a division by zero or "
                        f"a root/logarithm of a negative number). Change the initial condition."
                    ) from None
                size = "" if exc.last_value is None else f" (|{spec.dependent}| had reached {exc.last_value:.3g})"
                raise ValidationError(
                    f"The solution cannot be continued past {var} = {exc.stopped_at:.4g} (the domain ends at "
                    f"{end:g}): it blows up or the equation becomes singular there{size}. "
                    "Choose a domain that ends before this point."
                ) from None
            ref_failure = str(exc) if spec.exact is None else None  # the numerical solver itself failed
            warnings_.append(f"No reference solution: {exc} The true error cannot be established.")
        if ref is not None and ref.kind == "numerical" and ref.estimated_error:
            # A classical solution that disagrees with its own re-solve is not a trustworthy yardstick
            # (typically a solution that is singular or grows without bound at the edge of the domain).
            try:
                size = float(np.max(np.abs(ref(Validator(spec, {}, 1, seed).grid))))
            except Exception:  # pragma: no cover - defensive: never lose a solve over a diagnostic
                size = 0.0
            if size > 0 and ref.estimated_error > 1e-2 * size:
                warnings_.append(
                    f"The numerical solver's own error estimate is large ({ref.estimated_error:.2e}, against a solution of "
                    f"size {size:.2e}): the solution is probably singular, extremely sensitive or too steep for the "
                    "solver's grid, so neither this numerical solution nor errors measured against it can be trusted.")

        gen = ReportGenerator(self.runs_dir if self.save else None)
        if mode == "E":
            return self._numerical_only(spec, cfg, ref, gen, warnings_, t_total, problem, ref_failure)

        # Stages 5-8: PINN training with validation-driven control.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        budget = Budget(cfg.max_time_s)
        model = PINN(spec, hidden=cfg.hidden, hard_constraints=cfg.hard_constraints)
        trainer = Trainer(spec, cfg, budget, progress=lambda e: self._emit(stage="training", **e))
        validator = Validator(spec, cfg.thresholds, cfg.n_validation, cfg.seed, scale=model.scale)
        if model.scale != 1.0:
            cfg.reasons.append(
                f"[scale] The data have size {model.scale:.3g}, so the network output, the loss and the absolute "
                "thresholds (condition error, residual) are rescaled by that factor; relative errors are unaffected.")
        attempts: List[Attempt] = []
        gpinn_report: Optional[dict] = None

        def run_attempt(action: str, desc: str, m: PINN, adam: int, lbfgs: int, gpinn: bool, adaptive: bool, lr=None, time_cap=None):
            tic = time.perf_counter()
            self._emit(stage="attempt", attempt=len(attempts) + 1, action=action, description=desc)
            tr = trainer.train(m, adam, lbfgs, use_gpinn=gpinn, adaptive=adaptive, lr=lr, time_cap=time_cap)
            val = validator.check(m, ref)
            at = Attempt(
                number=len(attempts) + 1, action=action, description=desc,
                config={"hidden": list(m.hidden), "hard_constraints": m.hard, "gpinn": gpinn, "adaptive": adaptive,
                        "adam_iters": adam, "lbfgs_iters": lbfgs, "lr": lr or cfg.lr},
                train=tr.to_dict(), validation=val.to_dict(), time_s=time.perf_counter() - tic,
            )
            attempts.append(at)
            self._emit(stage="validated", attempt=at.number, passed=val.passed, metrics=val.metrics)
            return tr, val

        # In mode D the first attempt may use at most 60% of the budget, leaving time for refinement/retries.
        first_cap = INITIAL_ATTEMPT_SHARE * cfg.max_time_s if (mode == "D" and cfg.max_attempts > 1) else None
        tr, val = run_attempt("initial", "Configuration chosen by the rule table", model,
                              cfg.adam_iters, cfg.lbfgs_iters, cfg.use_gpinn, cfg.adaptive, time_cap=first_cap)
        best = {"state": _state(model), "val": val, "model_hidden": list(model.hidden), "attempt": 1}

        # Mode D: refine / gPINN / alternative configuration while the budget allows (rule R8).
        if mode == "D":
            gpinn_tried = False
            marching_tried = False
            refine_gain = None
            while (not best["val"].passed and len(attempts) < cfg.max_attempts
                   and budget.remaining() > MIN_RETRY_SECONDS):
                last_tr = tr
                if (not last_tr.nonfinite and list(model.hidden) == best["model_hidden"]
                        and _score(best["val"]) < _score(val)):
                    model.load_state_dict(best["state"])  # continue from the best weights, not a worse later state
                    val = best["val"]
                if last_tr.nonfinite:
                    cfg_seed = cfg.seed + 101 * len(attempts)
                    torch.manual_seed(cfg_seed)
                    model = PINN(spec, hidden=cfg.hidden, hard_constraints=cfg.hard_constraints)
                    tr, val = run_attempt("restart", f"Non-finite values: re-initialised (seed {cfg_seed}) with learning rate {cfg.lr / 10:g}",
                                          model, cfg.adam_iters, cfg.lbfgs_iters, False, cfg.adaptive, lr=cfg.lr / 10)
                elif (trainer.points.shape[0] < cfg.max_points and cfg.rar_k > 0 and "no_adaptive" not in ablate
                      and (refine_gain is None or refine_gain < REFINE_KEEP_GOING)):
                    before_score = _score(val)
                    infos = []
                    for _ in range(2):
                        info = trainer.refine(model)
                        if info.get("added", 0):
                            infos.append(info)
                    tr, val = run_attempt(
                        "refine",
                        f"Validation failed; added {sum(i['added'] for i in infos)} high-residual points "
                        f"(now {trainer.points.shape[0]}) and continued from the best weights (short low-rate Adam, then L-BFGS)",
                        model, CONTINUE_ADAM_ITERS, cfg.lbfgs_iters, False, False, lr=cfg.lr * CONTINUE_LR_FACTOR)
                    attempts[-1].train["validation_triggered_refinements"] = infos
                    # refine again only while refinement clearly helps (error ratio below REFINE_KEEP_GOING)
                    refine_gain = _score(val) / before_score if before_score > 0 else 1.0
                elif time_axis(spec) is not None and not marching_tried:
                    marching_tried = True
                    cfg_seed = cfg.seed + 101 * len(attempts)
                    torch.manual_seed(cfg_seed)
                    model = PINN(spec, hidden=cfg.hidden, hard_constraints=cfg.hard_constraints)
                    tic = time.perf_counter()
                    k = cfg.time_marching_windows
                    desc = (f"[R9] Causal time-marching over {k} growing time windows from a fresh initialisation "
                            f"(seed {cfg_seed}), then the full domain with L-BFGS")
                    self._emit(stage="attempt", attempt=len(attempts) + 1, action="time_marching", description=desc)
                    tr = trainer.train_marching(model, k, cfg.adam_iters, cfg.lbfgs_iters)
                    val = validator.check(model, ref)
                    attempts.append(Attempt(
                        number=len(attempts) + 1, action="time_marching", description=desc,
                        config={"hidden": list(model.hidden), "hard_constraints": model.hard, "gpinn": False, "adaptive": False,
                                "adam_iters": cfg.adam_iters, "lbfgs_iters": cfg.lbfgs_iters, "lr": cfg.lr, "windows": k},
                        train=tr.to_dict(), validation=val.to_dict(), time_s=time.perf_counter() - tic))
                    self._emit(stage="validated", attempt=len(attempts), passed=val.passed, metrics=val.metrics)
                elif cfg.gpinn_candidate and not gpinn_tried:
                    gpinn_tried = True
                    prev_time = attempts[-1].time_s
                    # Half-length retrain at up to ~2x the per-iteration cost (Yu et al. report about 1.4-2x).
                    est_cost = 2.0 * prev_time * 0.5
                    if est_cost > budget.remaining():
                        gpinn_report = {"tried": False, "reason": f"estimated cost {est_cost:.0f}s exceeds the remaining budget {budget.remaining():.0f}s"}
                        continue
                    before_state, before_val = _state(model), val
                    tr, val = run_attempt("gpinn", f"gPINN term added (lambda_g = {cfg.lambda_g:g}) and training continued",
                                          model, CONTINUE_ADAM_ITERS, cfg.lbfgs_iters, True, False, lr=cfg.lr * CONTINUE_LR_FACTOR)
                    improved = _score(val) < _score(before_val)
                    gpinn_report = {
                        "tried": True, "lambda_g": cfg.lambda_g,
                        "rel_l2_before": before_val.metrics.get("rel_l2"), "rel_l2_after": val.metrics.get("rel_l2"),
                        "residual_rms_before": before_val.metrics.get("residual_rms"), "residual_rms_after": val.metrics.get("residual_rms"),
                        "time_s": attempts[-1].time_s, "kept": improved,
                    }
                    if not improved:
                        model.load_state_dict(before_state)
                        val = before_val
                else:
                    hidden = [min(128, 2 * w) for w in cfg.hidden]
                    cfg_seed = cfg.seed + 101 * len(attempts)
                    torch.manual_seed(cfg_seed)
                    model = PINN(spec, hidden=hidden, hard_constraints=cfg.hard_constraints)
                    tr, val = run_attempt("alternate", f"Alternative configuration: wider network {hidden}, new initialisation (seed {cfg_seed})",
                                          model, cfg.adam_iters, cfg.lbfgs_iters, False, cfg.adaptive)
                if _score(val) < _score(best["val"]) or (val.passed and not best["val"].passed):
                    best = {"state": _state(model), "val": val, "model_hidden": list(model.hidden), "attempt": len(attempts)}
            if (not best["val"].passed and len(attempts) >= cfg.max_attempts and budget.remaining() > POLISH_MIN_SECONDS
                    and "no_retries" not in ablate):
                # attempts used up but budget left: polish the best model with L-BFGS on all retained points
                model = PINN(spec, hidden=best["model_hidden"], hard_constraints=cfg.hard_constraints)
                model.load_state_dict(best["state"])
                tr, val = run_attempt("polish", "Attempts used up with budget left: L-BFGS polishing of the best model on all retained points",
                                      model, 0, max(cfg.lbfgs_iters, 2000), False, False)
                if _score(val) < _score(best["val"]) or (val.passed and not best["val"].passed):
                    best = {"state": _state(model), "val": val, "model_hidden": list(model.hidden), "attempt": len(attempts)}
            if not best["val"].passed and budget.remaining() <= MIN_RETRY_SECONDS:
                warnings_.append("Compute budget exhausted before the thresholds were met.")

        # Rebuild the best model.
        final_model = PINN(spec, hidden=best["model_hidden"], hard_constraints=cfg.hard_constraints)
        final_model.load_state_dict(best["state"])
        final_val = best["val"]
        train_time = budget.elapsed()

        status = "passed" if final_val.passed else "failed"
        if final_val.passed and final_val.criterion == "residual-only":
            status = "passed_residual_only"
        final_method = "pinn"
        fallback = None
        if not final_val.passed and cfg.fallback:
            self._emit(stage="fallback")
            fallback = self._fallback(spec, ref, validator, ref_failure)
            if fallback["available"]:
                status, final_method = "fallback", "numerical_fallback"
            else:
                warnings_.append(f"Numerical fallback failed: {fallback['error']}")

        self._emit(stage="report")
        return gen.build(
            problem=problem, spec=spec, cfg=cfg, mode=mode, model=final_model, validator=validator, ref=ref,
            attempts=[a.to_dict() for a in attempts], best_attempt=best["attempt"], final_val=final_val,
            status=status, final_method=final_method, fallback=fallback, gpinn=gpinn_report,
            history=trainer.history, warnings=warnings_, ablate=ablate,
            timings={"training_s": round(train_time, 3), "reference_s": round(ref.runtime_s, 3) if ref else None,
                     "total_s": round(time.perf_counter() - t_total, 3), "budget_s": cfg.max_time_s},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_ablations(cfg: TrainingConfig, ablate: List[str]) -> None:
        if "no_hard" in ablate:
            cfg.hard_constraints = False
        if "no_adaptive" in ablate:
            cfg.adaptive = False
            cfg.n_collocation = cfg.max_points
        if "no_gpinn" in ablate:
            cfg.gpinn_candidate = False
            cfg.use_gpinn = False
        if "no_retries" in ablate:
            cfg.max_attempts = 1
        if "no_lbfgs" in ablate:
            cfg.adam_iters += cfg.lbfgs_iters
            cfg.lbfgs_iters = 0
        if "no_fallback" in ablate:
            cfg.fallback = False
        if ablate:
            cfg.reasons.append(f"Ablation: {', '.join(ablate)}")

    def _numerical_solution(self, spec: EquationSpec, ref: Optional[ReferenceSolution]) -> ReferenceSolution:
        if ref is not None and ref.kind == "numerical":
            return ref
        return numerical_reference(spec).solve(spec)

    def _fallback(self, spec: EquationSpec, ref: Optional[ReferenceSolution], validator: Validator,
                  ref_failure: Optional[str] = None) -> dict:
        if ref is None and ref_failure is not None:
            # the same numerical solver already failed while computing the reference: do not run it again
            return {"available": False, "error": ref_failure}
        try:
            num = self._numerical_solution(spec, ref)
        except ReferenceUnavailable as exc:
            return {"available": False, "error": str(exc)}
        u = num(validator.grid)
        independent = ref is not None and ref.kind == "analytical"
        val = validator.check_values(u, ref if independent else None)
        return {
            "available": True, "solver": num.describe(), "solution": num,
            "validated_against_exact": independent,
            "validation": val.to_dict() if independent else None,
            "note": ("Validated against the exact solution." if independent else
                     "The displayed numerical solution is the same solver used as the reference, so it is not independently "
                     f"validated; its own estimated error is {num.estimated_error:.2e}."),
        }

    def _numerical_only(self, spec, cfg, ref, gen, warnings_, t_total, problem, ref_failure=None) -> dict:
        validator = Validator(spec, cfg.thresholds, cfg.n_validation, cfg.seed, scale=output_scale(spec))
        self._emit(stage="numerical")
        fb = self._fallback(spec, ref, validator, ref_failure)
        if fb["available"]:
            status = "passed" if (fb["validation"] and fb["validation"]["passed"]) else "numerical_unvalidated"
            if fb["validation"] and not fb["validation"]["passed"]:
                status = "failed"
        else:
            status = "failed"
            warnings_.append(f"Numerical solver failed: {fb['error']}")
        return gen.build(
            problem=problem, spec=spec, cfg=cfg, mode="E", model=None, validator=validator, ref=ref,
            attempts=[], best_attempt=None, final_val=None, status=status,
            final_method="numerical" if fb["available"] else "none", fallback=fb, gpinn=None, history=[],
            warnings=warnings_, ablate=[],
            timings={"training_s": 0.0, "reference_s": round(ref.runtime_s, 3) if ref else None,
                     "numerical_s": round(fb["solution"].runtime_s, 3) if fb["available"] else None,
                     "total_s": round(time.perf_counter() - t_total, 3), "budget_s": None},
        )


def _state(model: PINN) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def solve(problem, mode: str = "D", seed: int = 0, **kw) -> dict:
    """Functional wrapper: ``solve(ProblemInput | dict, mode, seed, time_budget=..)``."""
    if isinstance(problem, dict):
        problem = ProblemInput.from_dict(problem)
    solver_kw = {k: kw.pop(k) for k in ("progress", "runs_dir", "save", "threads") if k in kw}
    return Solver(**solver_kw).solve(problem, mode=mode, seed=seed, **kw)
