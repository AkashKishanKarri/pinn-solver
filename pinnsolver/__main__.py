"""Command-line use:  python -m pinnsolver problem.json [--mode D] [--seed 0] [--budget 60]
                      python -m pinnsolver --benchmark decay
Prints a short summary and writes the full report to runs/<run_id>/.
"""
import argparse
import json
import sys

from . import benchmarks
from .parser import ValidationError
from .solver import solve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pinnsolver", description="Solve a supported ODE/PDE with the validation-driven PINN solver.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("problem", nargs="?", help="JSON file with equation, variables, domain, conditions[, parameters, exact]")
    src.add_argument("--benchmark", choices=list(benchmarks.BY_ID))
    ap.add_argument("--mode", default="D", choices=list("ABCDE"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="time budget in seconds")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    if a.benchmark:
        problem = benchmarks.get(a.benchmark)["problem"]
    else:
        try:
            with open(a.problem) as f:
                problem = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"Cannot read problem file {a.problem!r}: {exc}", file=sys.stderr)
            return 2
    try:
        r = solve(problem, mode=a.mode, seed=a.seed, time_budget=a.budget, runs_dir=a.runs_dir,
                  progress=lambda e: print(f"  [{e['stage']}] {e.get('description', '')}", file=sys.stderr) if e.get("stage") in ("attempt", "fallback") else None)
    except (ValidationError, ValueError) as exc:  # ValueError: malformed input, bad budget/mode
        print(f"Problem rejected: {exc}", file=sys.stderr)
        return 2
    print(r["status_text"])
    print(f"Solution produced by: {r['final_method_text']}")
    v = r.get("validation") or (r.get("fallback") or {}).get("validation")
    if v:
        print("Metrics:", {k: (f"{x:.3e}" if isinstance(x, float) else x) for k, x in v["metrics"].items()})
    for w in r["warnings"]:
        print("Warning:", w)
    print(f"Report: {a.runs_dir}/{r['run_id']}/report.md")
    return 0 if r["status"] in ("passed", "passed_residual_only") else 1


if __name__ == "__main__":
    sys.exit(main())
