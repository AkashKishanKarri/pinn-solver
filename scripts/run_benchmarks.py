#!/usr/bin/env python3
"""Run the benchmark suite: configurations A-E x problems x seeds (report Section 6.8).

Every configuration of one problem gets the same wall-clock budget and the same
collocation-point budget. All runs - including failures - are written to
results.csv / results.json, and summary.md reports mean ± sample std over seeds.

Example:
    python scripts/run_benchmarks.py --seeds 0 1 2 --out benchmark_results
    python scripts/run_benchmarks.py --problems decay heat --modes A D --ablations no_adaptive no_gpinn
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnsolver import benchmarks  # noqa: E402
from pinnsolver.report import versions  # noqa: E402
from pinnsolver.solver import ABLATIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=[b["id"] for b in benchmarks.BENCHMARKS], choices=list(benchmarks.BY_ID))
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C", "D", "E"], choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ablations", nargs="*", default=[], choices=list(ABLATIONS), help="extra runs of D with one component removed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="benchmark_results")
    a = ap.parse_args()

    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    csv_path = os.path.join(a.out, "results.csv")
    if os.path.exists(csv_path):
        started += " (resumed: rows already in results.csv were kept, not re-run)"
    rows = benchmarks.run_suite(a.problems, a.modes, a.seeds, a.out, ablations=a.ablations, threads=a.threads)
    v = versions()
    budgets = ", ".join(f"{p}: {benchmarks.get(p)['budget_s']} s" for p in a.problems)
    text = [
        "# Benchmark results", "",
        f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} (started {started}).",
        f"Hardware/software: {v['platform']}, {v['machine']}; Python {v['python']}, PyTorch {v['torch']}, NumPy {v['numpy']}, SciPy {v['scipy']}, SymPy {v['sympy']}; {a.threads} torch threads, float64 on CPU.",
        f"Seeds: {a.seeds}. Equal wall-clock budget per problem for every configuration ({budgets}) and equal collocation-point budget (A/B use it uniformly from the start; C/D grow towards it adaptively).",
        "Thresholds were fixed before the runs: relative L2 <= 1e-3 (ODEs/BVPs) and <= 1e-2 (PDEs); IC/BC error <= 1e-4 (ODE/BVP) and <= 1e-3 (PDE).",
        "Error metrics are for the PINN solution (in D, the best attempt) even when D then displayed the numerical fallback; 'fallback used' counts those runs.",
        "For E the rows report the numerical solver compared with the exact solution where one exists (no PINN is trained).",
        "", benchmarks.summarize(rows), "",
    ]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
