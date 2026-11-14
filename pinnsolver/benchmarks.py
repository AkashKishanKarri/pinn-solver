"""Benchmark suite (report Table 4.2) and the multi-seed experiment runner (Objective 2).

Every exact solution below is re-verified by the parser (it must satisfy the
equation and all conditions) before it is used as a reference.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional

PI = math.pi
# IC/BC-error limit that goes with each relative-L2 limit (strategy.DEFAULT_THRESHOLDS: ODE/BVP 1e-3/1e-4, PDE 1e-2/1e-3)
CONDITION_LIMIT = {1e-3: 1e-4, 1e-2: 1e-3}

BENCHMARKS: List[Dict[str, object]] = [
    {
        "id": "decay", "title": "Exponential decay", "class": "First-order ODE IVP", "advanced": False, "budget_s": 30,
        "description": "u' = -2u, u(0) = 1 on [0, 2]; exact u = exp(-2t).",
        "problem": {"name": "Exponential decay", "equation": "u' = -2*u", "variables": ["t"], "domain": {"t": [0, 2]},
                    "conditions": ["u(0) = 1"], "parameters": {}, "exact": "exp(-2*t)"},
    },
    {
        "id": "logistic", "title": "Logistic growth (nonlinear)", "class": "First-order ODE IVP", "advanced": False, "budget_s": 30,
        "description": "u' = r u (1 - u), u(0) = 0.1 on [0, 10]; exact u = 1/(1 + 9 exp(-r t)).",
        "problem": {"name": "Logistic growth", "equation": "u' = r*u*(1 - u)", "variables": ["t"], "domain": {"t": [0, 10]},
                    "conditions": ["u(0) = 0.1"], "parameters": {"r": 1.0}, "exact": "1/(1 + 9*exp(-r*t))"},
    },
    {
        "id": "oscillator", "title": "Harmonic oscillator", "class": "Second-order ODE IVP", "advanced": False, "budget_s": 120,
        "description": "u'' + w^2 u = 0, u(0) = 1, u'(0) = 0 on [0, 10], w = 2; exact u = cos(2t) (about 3.2 cycles).",
        "problem": {"name": "Harmonic oscillator", "equation": "u'' + w^2*u = 0", "variables": ["t"], "domain": {"t": [0, 10]},
                    "conditions": ["u(0) = 1", "u'(0) = 0"], "parameters": {"w": 2.0}, "exact": "cos(2*t)"},
    },
    {
        "id": "bvp", "title": "Linear two-point BVP", "class": "Boundary value problem", "advanced": False, "budget_s": 30,
        "description": "u'' = u, u(0) = 0, u(1) = sinh(1); exact u = sinh(x).",
        "problem": {"name": "Linear BVP", "equation": "u'' = u", "variables": ["x"], "domain": {"x": [0, 1]},
                    "conditions": ["u(0) = 0", "u(1) = sinh(1)"], "parameters": {}, "exact": "sinh(x)"},
    },
    {
        "id": "bvp_neumann", "title": "BVP with a Neumann end", "class": "Boundary value problem", "advanced": False, "budget_s": 30,
        "description": "u'' = u, u(0) = 1, u'(1) = sinh(1); exact u = cosh(x). The Neumann condition is imposed as a penalty.",
        "problem": {"name": "BVP with Neumann condition", "equation": "u'' = u", "variables": ["x"], "domain": {"x": [0, 1]},
                    "conditions": ["u(0) = 1", "u'(1) = sinh(1)"], "parameters": {}, "exact": "cosh(x)"},
    },
    {
        "id": "poisson", "title": "1D Poisson, two scales", "class": "1D Poisson (Dirichlet)", "advanced": False, "budget_s": 60,
        "description": "-u'' = pi^2 sin(pi x) + 6.4 pi^2 sin(8 pi x) on [0, 1], u(0) = u(1) = 0; exact u = sin(pi x) + 0.1 sin(8 pi x).",
        "problem": {"name": "1D Poisson (manufactured)", "equation": "-u'' = pi^2*sin(pi*x) + 6.4*pi^2*sin(8*pi*x)", "variables": ["x"],
                    "domain": {"x": [0, 1]}, "conditions": ["u(0) = 0", "u(1) = 0"], "parameters": {},
                    "exact": "sin(pi*x) + 0.1*sin(8*pi*x)"},
    },
    {
        "id": "heat", "title": "1D heat equation", "class": "1D heat / diffusion", "advanced": False, "budget_s": 120,
        "description": "u_t = k u_xx on [0, 1] x [0, 0.5], k = 0.1, u(x, 0) = sin(pi x), u = 0 at both ends; exact u = exp(-k pi^2 t) sin(pi x).",
        "problem": {"name": "1D heat equation", "equation": "u_t = k*u_xx", "variables": ["x", "t"], "domain": {"x": [0, 1], "t": [0, 0.5]},
                    "conditions": ["u(x, 0) = sin(pi*x)", "u(0, t) = 0", "u(1, t) = 0"], "parameters": {"k": 0.1},
                    "exact": "exp(-k*pi^2*t)*sin(pi*x)"},
    },
    {
        "id": "stiff", "title": "Stiff relaxation (advanced)", "class": "Stress test: stiff ODE", "advanced": True, "budget_s": 30,
        "description": "u' = -50 (u - cos t), u(0) = 0 on [0, 2]; exact solution has an e^(-50 t) transient.",
        "problem": {"name": "Stiff relaxation", "equation": "u' = -a*(u - cos(t))", "variables": ["t"], "domain": {"t": [0, 2]},
                    "conditions": ["u(0) = 0"], "parameters": {"a": 50.0},
                    "exact": "(2500*cos(t) + 50*sin(t))/2501 - 2500/2501*exp(-50*t)"},
    },
    {
        "id": "burgers", "title": "Viscous Burgers' equation (advanced)", "class": "Stress test: steep gradient", "advanced": True, "budget_s": 240,
        "description": "u_t + u u_x = nu u_xx, nu = 0.01/pi, on [-1, 1] x [0, 1], u(x, 0) = -sin(pi x); a steep front forms at x = 0. Reference: fine-grid finite differences.",
        "problem": {"name": "Viscous Burgers' equation", "equation": "u_t + u*u_x = nu*u_xx", "variables": ["x", "t"],
                    "domain": {"x": [-1, 1], "t": [0, 1]}, "conditions": ["u(x, 0) = -sin(pi*x)", "u(-1, t) = 0", "u(1, t) = 0"],
                    "parameters": {"nu": 0.01 / PI}, "exact": None},
    },
]
BY_ID = {b["id"]: b for b in BENCHMARKS}


def get(benchmark_id: str) -> dict:
    if benchmark_id not in BY_ID:
        raise KeyError(f"Unknown benchmark {benchmark_id!r}; available: {', '.join(BY_ID)}")
    return BY_ID[benchmark_id]


# ----------------------------------------------------------------------
# experiment runner
# ----------------------------------------------------------------------
ROW_FIELDS = [
    "problem", "mode", "seed", "status", "final_method", "pinn_passed", "rel_l2", "max_abs_error",
    "residual_rms", "residual_max", "condition_max", "threshold_rel_l2", "training_time_s", "time_budget_s",
    "adam_iterations", "lbfgs_iterations", "collocation_points", "point_budget", "attempts", "gpinn_kept", "error",
]


def run_one(bid: str, mode: str, seed: int, time_budget: Optional[float] = None, ablate=None, threads: int = 4) -> dict:
    from .solver import solve

    b = get(bid)
    budget = float(time_budget if time_budget is not None else b["budget_s"])
    row = {k: None for k in ROW_FIELDS}
    row.update(problem=bid, mode=mode + ("-" + "-".join(ablate) if ablate else ""), seed=seed, time_budget_s=budget)
    try:
        r = solve(b["problem"], mode=mode, seed=seed, time_budget=budget, ablate=ablate, save=False, threads=threads)
    except Exception as exc:  # recorded, never hidden
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return row
    v = r.get("validation") or (r.get("fallback") or {}).get("validation") or {}
    m = v.get("metrics", {}) if v else {}
    row.update(
        status=r["status"], final_method=r["final_method"],
        pinn_passed=(r["status"] in ("passed", "passed_residual_only") and r["final_method"] == "pinn"),
        rel_l2=m.get("rel_l2"), max_abs_error=m.get("max_abs_error"), residual_rms=m.get("residual_rms"),
        residual_max=m.get("residual_max"), condition_max=m.get("condition_max"),
        threshold_rel_l2=(v.get("thresholds") or {}).get("rel_l2") if v else None,
        training_time_s=r["timings"].get("training_s"), adam_iterations=r["budget"]["adam_iterations"],
        lbfgs_iterations=r["budget"]["lbfgs_iterations"], collocation_points=r["budget"]["collocation_points_final"],
        point_budget=r["budget"]["collocation_point_budget"], attempts=r["budget"]["attempts"],
        gpinn_kept=(r.get("gpinn") or {}).get("kept"),
    )
    if mode == "E":
        row["training_time_s"] = r["timings"].get("numerical_s")
    return row


def _fmt(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.2e}"
    return f"{statistics.mean(vals):.2e} ± {statistics.stdev(vals):.1e}"


def summarize(rows: List[dict]) -> str:
    """Markdown summary: mean ± sample std over seeds, success rates, and every failed run."""
    out = ["| problem | mode | runs | PINN met threshold | fallback used | rel. L2 error (mean ± sd) | RMS residual | training time s (mean ± sd) | points | attempts |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    order = {b["id"]: i for i, b in enumerate(BENCHMARKS)}
    rank = lambda prob, mode: (order.get(prob, len(order)), mode)
    rows = sorted(rows, key=lambda r: (*rank(r["problem"], r["mode"]), int(r["seed"])))
    keys = []
    for r in rows:
        k = (r["problem"], r["mode"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rs = [r for r in rows if (r["problem"], r["mode"]) == k]
        n = len(rs)
        ok = sum(1 for r in rs if r["pinn_passed"])
        fb = sum(1 for r in rs if r["final_method"] == "numerical_fallback")
        pts = sorted({r["collocation_points"] for r in rs if r["collocation_points"] is not None})
        att = [r["attempts"] for r in rs if r["attempts"] is not None]
        pinn_col = "n/a (no PINN)" if k[1] == "E" else f"{ok}/{n}"
        out.append(
            f"| {k[0]} | {k[1]} | {n} | {pinn_col} | {fb}/{n} | {_fmt([r['rel_l2'] for r in rs])} | "
            f"{_fmt([r['residual_rms'] for r in rs])} | {_fmt([r['training_time_s'] for r in rs])} | "
            f"{', '.join(str(p) for p in pts) or '-'} | {_fmt([float(a) for a in att]) if att else '-'} |"
        )
    failed = [r for r in rows if r["mode"] != "E" and not r["pinn_passed"]]
    out += ["", f"### Runs in which the PINN did not meet its thresholds ({len(failed)})", "",
            "A run passes only if BOTH the relative L2 error and the maximum IC/BC error are within their limits; the reason column says which limit was missed."]
    if failed:
        out += ["", "| problem | mode | seed | status | final method | rel. L2 (limit) | max IC/BC error (limit) | reason |", "|---|---|---|---|---|---|---|---|"]
        for r in failed:
            thr = r["threshold_rel_l2"]
            cthr = CONDITION_LIMIT.get(thr)
            rl = f"{r['rel_l2']:.2e}" if r["rel_l2"] is not None else "n/a"
            cm = f"{r['condition_max']:.1e}" if r["condition_max"] is not None else "n/a"
            reasons = []
            if r["rel_l2"] is not None and thr is not None and r["rel_l2"] > thr:
                reasons.append("relative L2 error")
            if r["condition_max"] is not None and cthr is not None and r["condition_max"] > cthr:
                reasons.append("IC/BC error")
            reason = " and ".join(reasons) or (r["error"] or "see report")
            out.append(f"| {r['problem']} | {r['mode']} | {r['seed']} | {r['status']} | {r['final_method']} | "
                       f"{rl} ({thr:.0e}) | {cm} ({cthr:.0e}) | {reason} |" if thr is not None and cthr is not None else
                       f"| {r['problem']} | {r['mode']} | {r['seed']} | {r['status']} | {r['final_method']} | {rl} | {cm} | {reason} |")
    else:
        out.append("None.")
    return "\n".join(out)


def run_suite(problems: List[str], modes: List[str], seeds: List[int], out_dir: str, ablations: Optional[List[str]] = None,
              threads: int = 4, log=print) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    rows: List[dict] = []
    if os.path.exists(csv_path):  # resume
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(_parse_row(r))
    done = {(r["problem"], r["mode"], int(r["seed"])) for r in rows}
    jobs = [(p, m, None, s) for p in problems for m in modes for s in seeds]
    jobs += [(p, "D", a, s) for p in problems for a in (ablations or []) for s in seeds]
    for p, m, a, s in jobs:
        label = m + ("-" + a if a else "")
        if (p, label, s) in done:
            continue
        t = time.time()
        row = run_one(p, m, s, ablate=[a] if a else None, threads=threads)
        rows.append(row)
        log(f"{p:12s} {label:14s} seed={s} status={row['status']:22s} rel_l2={row['rel_l2']} time={time.time() - t:.1f}s")
        write_new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
            if write_new:
                w.writeheader()
            w.writerow(row)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def _parse_row(r: dict) -> dict:
    out = {}
    for k in ROW_FIELDS:
        v = r.get(k)
        if v in (None, "", "None"):
            out[k] = None
        elif k in ("seed", "adam_iterations", "lbfgs_iterations", "collocation_points", "point_budget", "attempts"):
            out[k] = int(float(v))
        elif k in ("pinn_passed", "gpinn_kept"):
            out[k] = v == "True"
        elif k in ("rel_l2", "max_abs_error", "residual_rms", "residual_max", "condition_max", "threshold_rel_l2", "training_time_s", "time_budget_s"):
            out[k] = float(v)
        else:
            out[k] = v
    return out
