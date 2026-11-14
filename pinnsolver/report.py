"""Results and report (M9).

Produces the user-facing report of brief Section 13 and stores the run in the
data stores of the level-1 DFD (report Fig. 7.2)::

    runs/<run_id>/problem.json        D1  problem as entered and as parsed
    runs/<run_id>/training_log.csv    D3  training log (every 50 iterations)
    runs/<run_id>/reference.json      D4  reference solver settings + values on the grid
    runs/<run_id>/report.json|.md     D5  results, metrics, provenance
    runs/<run_id>/*.png                   plots
The rule table (D2) is :data:`pinnsolver.strategy.RULE_TABLE` and is copied into every report.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from .pinn import residual, to_tensor
from .spec import EquationSpec
from .strategy import MODE_NAMES, RULE_TABLE

PLOT_N_1D = 501
PLOT_N_2D = 81

STATUS_TEXT = {
    "passed": "PASSED: the PINN solution met every pre-declared threshold against the reference solution.",
    "passed_residual_only": "PASSED (residual-only): no reference solution was available, so the true error is NOT established.",
    "fallback": "PINN did NOT meet the thresholds within the budget. The displayed solution comes from the numerical fallback solver.",
    "failed": "FAILED: the solution did not meet the pre-declared thresholds within the budget.",
    "numerical_unvalidated": "Numerical solution. No exact solution was available, so it is not independently validated (see its own error estimate).",
}
METHOD_TEXT = {
    "pinn": "Physics-informed neural network (PINN)",
    "numerical_fallback": "Numerical fallback solver (NOT the PINN)",
    "numerical": "Classical numerical solver (mode E)",
    "none": "No solution produced",
}
LIMITATIONS = [
    "Only the equation classes listed in the documentation are supported; other inputs are rejected.",
    "Residual and condition checks are evidence, not a proof of correctness; the true error is known only where a reference exists.",
    "Numerical references have their own (estimated) error, reported with the reference settings.",
    "Results are for one random seed; use the benchmark runner for multi-seed statistics.",
    "Not suitable for safety-critical engineering use without domain-specific validation and expert review.",
]


def _clean(o):
    """Make ``o`` JSON-safe (NaN/Inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def versions() -> Dict[str, str]:
    import scipy
    import sympy

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def plot_grid(spec: EquationSpec):
    if spec.dim == 1:
        lo, hi = spec.bounds[0]
        x = np.linspace(lo, hi, PLOT_N_1D)
        return {"x": x}, x[:, None]
    (a, b), (t0, t1) = spec.bounds
    xs, ts = np.linspace(a, b, PLOT_N_2D), np.linspace(t0, t1, PLOT_N_2D)
    X, T = np.meshgrid(xs, ts, indexing="ij")
    return {"x": xs, "t": ts}, np.column_stack([X.ravel(), T.ravel()])


class ReportGenerator:
    def __init__(self, runs_dir: Optional[str] = "runs"):
        self.runs_dir = runs_dir

    # ------------------------------------------------------------------
    def build(self, *, problem, spec: EquationSpec, cfg, mode, model, validator, ref, attempts, best_attempt,
              final_val, status, final_method, fallback, gpinn, history, warnings, ablate, timings) -> dict:
        created = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        fp = cfg.fingerprint()
        digest = hashlib.sha256(json.dumps(problem.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
        run_id = f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}-{mode}-s{cfg.seed}-{digest}"

        axes, P = plot_grid(spec)
        shape = (PLOT_N_1D,) if spec.dim == 1 else (PLOT_N_2D, PLOT_N_2D)
        plots: Dict[str, object] = {"axes": axes, "dim": spec.dim, "variables": list(spec.variables)}
        u_pinn = u_res = None
        if model is not None:
            with torch.no_grad():
                u_pinn = model(to_tensor(P)).numpy()
            Xg = to_tensor(P, requires_grad=True)
            u_res = residual(model, spec, Xg, create_graph=False).detach().numpy()
            plots["pinn"] = u_pinn.reshape(shape)
            plots["residual"] = u_res.reshape(shape)
        u_ref = ref(P) if ref is not None else None
        if u_ref is not None:
            plots["reference"] = u_ref.reshape(shape)
            plots["reference_label"] = ref.method
        fb_desc = None
        u_final = u_pinn
        if fallback is not None:
            fb_desc = {k: v for k, v in fallback.items() if k != "solution"}
            if fallback.get("available") and final_method in ("numerical_fallback", "numerical"):
                u_final = fallback["solution"](P)
                plots["numerical"] = u_final.reshape(shape)
        if u_final is not None:
            plots["final"] = u_final.reshape(shape)
            plots["final_label"] = METHOD_TEXT[final_method]
            if u_ref is not None and not (final_method != "pinn" and ref is not None and ref.kind == "numerical"):
                plots["abs_error"] = np.abs(u_final - u_ref).reshape(shape)

        # budget / effort summary
        total_adam = sum(a["training"]["adam_iterations"] for a in attempts)
        total_lbfgs = sum(a["training"]["lbfgs_iterations"] for a in attempts)
        final_points = attempts[-1]["training"]["n_collocation_points"] if attempts else 0

        report = {
            "run_id": run_id,
            "created": created,
            "mode": mode,
            "mode_name": MODE_NAMES[mode],
            "status": status,
            "status_text": STATUS_TEXT.get(status, status),
            "final_method": final_method,
            "final_method_text": METHOD_TEXT[final_method],
            "problem": problem.to_dict(),
            "parsed": spec.to_dict(),
            "strategy": {
                "reasons": cfg.reasons,
                "rules_fired": cfg.rules_fired,
                "config": cfg.to_dict(),
                "config_fingerprint": fp,
                "ablations": ablate,
                "rule_table": RULE_TABLE,
            },
            "architecture": model.describe() if model is not None else None,
            "reference": ref.describe() if ref is not None else None,
            "validation": final_val.to_dict() if final_val is not None else None,
            "attempts": attempts,
            "best_attempt": best_attempt,
            "gpinn": gpinn,
            "fallback": fb_desc,
            "budget": {
                "time_budget_s": cfg.max_time_s if mode != "E" else None,
                "training_time_s": timings.get("training_s"),
                "adam_iterations": total_adam,
                "lbfgs_iterations": total_lbfgs,
                "collocation_points_final": final_points,
                "collocation_point_budget": cfg.max_points if mode != "E" else None,
                "attempts": len(attempts),
                "max_attempts": cfg.max_attempts if mode != "E" else None,
            },
            "timings": timings,
            "warnings": warnings,
            "limitations": LIMITATIONS,
            "reproducibility": {
                "run_id": run_id,
                "seed": cfg.seed,
                "validation_seed": cfg.seed + 1_000_003,
                "config_fingerprint": fp,
                "torch_threads": torch.get_num_threads(),
                "precision": "float64 (CPU)",
                "versions": versions(),
            },
            "loss_history": _downsample(history, 400),
            "plots": plots,
            "files": [],
        }
        report = _clean(report)
        if self.runs_dir:
            report["files"] = self.save_run(report, history, ref, P, u_ref)
        return report

    # ------------------------------------------------------------------
    def save_run(self, report: dict, history: List[dict], ref, P, u_ref) -> List[str]:
        d = os.path.join(self.runs_dir, report["run_id"])
        os.makedirs(d, exist_ok=True)
        files = []

        def dump(name, obj):
            with open(os.path.join(d, name), "w") as f:
                json.dump(_clean(obj), f, indent=2)
            files.append(name)

        dump("problem.json", {"input": report["problem"], "parsed": report["parsed"]})  # D1
        if history:  # D3
            keys = sorted({k for h in history for k in h})
            with open(os.path.join(d, "training_log.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for h in history:
                    w.writerow(h)
            files.append("training_log.csv")
        if ref is not None:  # D4
            dump("reference.json", {"reference": ref.describe(), "points": P, "values": u_ref})
        files.extend(self.plots(report, d))
        with open(os.path.join(d, "report.md"), "w") as f:
            f.write(self.markdown(report))
        files.append("report.md")
        files.append("report.json")
        light = {k: v for k, v in report.items() if k not in ("plots",)}
        light["files"] = files
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump(_clean(light), f, indent=2)
        return files

    # ------------------------------------------------------------------
    def plots(self, report: dict, d: str) -> List[str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = report["plots"]
        out = []
        var = p["variables"]
        if p["dim"] == 1:
            x = np.asarray(p["axes"]["x"])
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            if "reference" in p:
                ax[0].plot(x, p["reference"], "k-", lw=2.5, alpha=0.35, label=f"Reference ({p['reference_label']})")
            if "pinn" in p:
                ax[0].plot(x, p["pinn"], "C0--", lw=1.5, label="PINN")
            if "numerical" in p:
                ax[0].plot(x, p["numerical"], "C3:", lw=1.5, label="Numerical solution")
            ax[0].set_xlabel(var[0]); ax[0].set_ylabel(report["parsed"]["dependent"]); ax[0].legend(fontsize=8)
            ax[0].set_title("Solution")
            if "abs_error" in p:
                ax[1].semilogy(x, np.maximum(np.asarray(p["abs_error"], dtype=float), 1e-17), "C1", label="|final - reference|")
            if "residual" in p:
                ax[1].semilogy(x, np.maximum(np.abs(np.asarray(p["residual"], dtype=float)), 1e-17), "C2", alpha=0.7, label="|PINN residual|")
            ax[1].set_xlabel(var[0]); ax[1].legend(fontsize=8); ax[1].set_title("Error and residual")
            fig.tight_layout()
            fig.savefig(os.path.join(d, "solution.png"), dpi=130)
            plt.close(fig)
            out.append("solution.png")
        else:
            xs, ts = np.asarray(p["axes"]["x"]), np.asarray(p["axes"]["t"])
            panels = [(k, t) for k, t in [("final", "Final solution"), ("reference", "Reference"), ("abs_error", "|final - reference|"), ("residual", "|PINN residual|")] if k in p]
            fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
            axs = np.atleast_1d(axs)
            for ax, (k, title) in zip(axs, panels):
                Z = np.asarray(p[k], dtype=float)
                if k == "residual":
                    Z = np.abs(Z)
                im = ax.pcolormesh(ts, xs, Z, shading="auto", cmap="viridis" if k in ("final", "reference") else "magma")
                fig.colorbar(im, ax=ax)
                ax.set_xlabel("t"); ax.set_ylabel(var[0]); ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(d, "solution.png"), dpi=130)
            plt.close(fig)
            out.append("solution.png")
        hist = report.get("loss_history") or []
        if hist:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            its = list(range(len(hist)))
            for key, lab in [("loss_total", "total"), ("loss_residual", "residual"), ("loss_initial", "initial"), ("loss_boundary", "boundary"), ("loss_gpinn", "gPINN")]:
                ys = [h.get(key) for h in hist]
                if any(y is not None for y in ys):
                    ax.semilogy(its, [y if y is not None and y > 0 else np.nan for y in ys], label=lab)
            ax.set_xlabel("log entry (every 50 iterations; attempts concatenated)")
            ax.set_ylabel("loss"); ax.legend(fontsize=8); ax.set_title("Training loss")
            fig.tight_layout()
            fig.savefig(os.path.join(d, "loss.png"), dpi=130)
            plt.close(fig)
            out.append("loss.png")
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def markdown(r: dict) -> str:
        def f(v):
            return "n/a" if v is None else (f"{v:.3e}" if isinstance(v, float) else str(v))

        L = [f"# Solver report: {r['problem'].get('name') or r['parsed']['equation']}", ""]
        L += [f"**Status:** {r['status_text']}", f"**Solution produced by:** {r['final_method_text']}",
              f"**Mode:** {r['mode']} ({r['mode_name']})  ·  **Run:** `{r['run_id']}`", ""]
        p = r["parsed"]
        L += ["## Problem as parsed", f"- Class: {p['kind']} ({p['subtype']})", f"- Equation: `{p['equation']}`  (explicit: `{p['explicit_form']}`)",
              f"- Domain: {p['domain']}", "- Conditions: " + "; ".join(f"`{c['text']}` ({c['role']}, {c['type']})" for c in p["conditions"]),
              f"- Parameters: {p['parameters'] or 'none'}", f"- Linear: {p['linear']}", f"- Features: {p['features']}", ""]
        L += ["## Strategy"] + [f"- {x}" for x in r["strategy"]["reasons"]] + [""]
        if r.get("architecture"):
            L += [f"Network: {r['architecture']}", ""]
        if r.get("reference"):
            ref = r["reference"]
            L += ["## Reference", f"- {ref['method']} ({ref['kind']}); estimated max error {f(ref['estimated_max_error'])}", f"- Settings: {ref['settings']}"] + [f"- {n}" for n in ref["notes"]] + [""]
        if r.get("validation"):
            v = r["validation"]
            L += ["## Independent validation (fresh points)", f"Criterion: {v['criterion']}; passed: **{v['passed']}**", "",
                  "| metric | value | threshold |", "|---|---|---|"]
            th = v["thresholds"]
            for k in ("rel_l2", "max_abs_error", "residual_rms", "residual_max", "condition_max"):
                L.append(f"| {k} | {f(v['metrics'].get(k))} | {f(th.get(k))} |")
            L += [""] + [f"- Condition `{c['condition']}`: max error {f(c['max_abs_error'])} ({'built in' if c['built_in'] else 'penalty'})" for c in v["conditions"]]
            L += [f"- FAIL: {x}" for x in v["failures"]] + [f"- Note: {x}" for x in v["notes"]] + [""]
        if r.get("attempts"):
            L += ["## Attempts", "| # | action | rel L2 | RMS residual | passed | time (s) |", "|---|---|---|---|---|---|"]
            for a in r["attempts"]:
                m = a["validation"]["metrics"]
                L.append(f"| {a['attempt']} | {a['action']} | {f(m.get('rel_l2'))} | {f(m.get('residual_rms'))} | {a['validation']['passed']} | {a['time_s']} |")
            L.append("")
        if r.get("gpinn"):
            L += ["## gPINN gate", f"{r['gpinn']}", ""]
        if r.get("fallback"):
            L += ["## Numerical solution / fallback", f"{r['fallback'].get('note', r['fallback'].get('error'))}", ""]
        L += ["## Budget", f"{r['budget']}", f"Timings: {r['timings']}", ""]
        if r["warnings"]:
            L += ["## Warnings"] + [f"- {w}" for w in r["warnings"]] + [""]
        L += ["## Limitations"] + [f"- {x}" for x in r["limitations"]] + [""]
        L += ["## Reproducibility", f"{r['reproducibility']}", ""]
        return "\n".join(L)


def _downsample(history: List[dict], n: int) -> List[dict]:
    if len(history) <= n:
        return history
    idx = np.unique(np.linspace(0, len(history) - 1, n).round().astype(int))
    return [history[i] for i in idx]
