"""pytest plugin: records every input the solver receives during the test suite, with the real outcome.

Used by docs/build_testcases.py:  PYTHONPATH=docs pytest -p pinn_recorder   (output: $PINN_RECORD_OUT, JSON lines)
"""
import json
import math
import os
import sys
import traceback

OUT = os.environ.get("PINN_RECORD_OUT", "recorded.jsonl")


def _test_id():
    t = os.environ.get("PYTEST_CURRENT_TEST", "")
    return t.rsplit(" (", 1)[0]


def _caller_is(fragment, depth=12):
    """True if a frame within `depth` levels above the wrapper comes from a file whose path contains fragment."""
    f = sys._getframe(2)
    for _ in range(depth):
        if f is None:
            return False
        if fragment in f.f_code.co_filename.replace("\\", "/"):
            return True
        f = f.f_back
    return False


def _direct_from_tests():
    f = sys._getframe(2)
    return f is not None and "/tests/" in f.f_code.co_filename.replace("\\", "/")


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def _write(rec):
    rec["test"] = _test_id()
    with open(OUT, "a") as fh:
        fh.write(json.dumps(_clean(rec)) + "\n")


def _problem(p):
    return {"name": p.name, "equation": p.equation, "variables": list(p.variables), "dependent": p.dependent,
            "domain": {k: list(v) for k, v in dict(p.domain).items()}, "conditions": list(p.conditions),
            "parameters": dict(p.parameters or {}), "exact": p.exact}


def pytest_configure(config):
    from pinnsolver import expr, parser, solver, spec

    orig_validate = parser.Parser.validate

    def validate(self, *a, **k):
        if _caller_is("pinnsolver/solver.py"):  # recorded by the Solver wrapper instead
            return orig_validate(self, *a, **k)
        rec = {"kind": "validate", "problem": _problem(self.problem)}
        try:
            s = orig_validate(self, *a, **k)
            rec.update(outcome="accepted", detail=f"{s.kind}: {s.subtype}", warnings=list(getattr(s, "warnings", []) or []))
            return s
        except Exception as exc:
            rec.update(outcome="rejected", detail=str(exc), error=type(exc).__name__)
            raise
        finally:
            _write(rec)

    parser.Parser.validate = validate

    orig_from_dict = spec.ProblemInput.from_dict  # a staticmethod: plain function on access

    @staticmethod
    def from_dict(d, *a, **k):
        try:
            return orig_from_dict(d, *a, **k)
        except Exception as exc:
            if not _caller_is("pinnsolver/api.py"):
                _write({"kind": "problem_dict", "raw": d if isinstance(d, dict) else repr(d),
                        "outcome": "rejected", "detail": str(exc), "error": type(exc).__name__})
            raise

    spec.ProblemInput.from_dict = from_dict

    orig_parse = expr.parse_expression

    def parse_expression(text, *a, **k):
        if not _direct_from_tests():
            return orig_parse(text, *a, **k)
        rec = {"kind": "expression", "text": text}
        try:
            e = orig_parse(text, *a, **k)
            rec.update(outcome="accepted", detail=str(e))
            return e
        except Exception as exc:
            rec.update(outcome="rejected", detail=str(exc), error=type(exc).__name__)
            raise
        finally:
            _write(rec)

    expr.parse_expression = parse_expression
    for mod in list(sys.modules.values()):  # tests may have imported the name directly
        if getattr(mod, "__name__", "").startswith("tests") and getattr(mod, "parse_expression", None) is orig_parse:
            mod.parse_expression = parse_expression

    orig_solve = solver.Solver.solve

    def solve(self, problem, mode="D", seed=0, time_budget=None, thresholds=None, ablate=None, **k):
        rec = {"kind": "solve", "problem": _problem(problem), "mode": mode, "seed": seed, "time_budget": time_budget,
               "thresholds": thresholds, "ablate": ablate}
        try:
            r = orig_solve(self, problem, mode=mode, seed=seed, time_budget=time_budget, thresholds=thresholds, ablate=ablate, **k)
            m = (r.get("validation") or {}).get("metrics") or {}
            rec.update(outcome=r.get("status"), detail=r.get("status_text"), final_method=r.get("final_method"),
                       rel_l2=m.get("rel_l2"), condition_max=m.get("condition_max"),
                       training_s=(r.get("timings") or {}).get("training_s"), attempts=len(r.get("attempts") or []))
            return r
        except Exception as exc:
            rec.update(outcome="rejected" if type(exc).__name__ == "ValidationError" else "error",
                       detail=str(exc), error=type(exc).__name__)
            raise
        finally:
            _write(rec)

    solver.Solver.solve = solve
