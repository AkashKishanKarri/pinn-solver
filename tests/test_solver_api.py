import time

import numpy as np
import pytest

from pinnsolver.benchmarks import get, run_one, summarize
from pinnsolver.solver import solve
from pinnsolver.strategy import StrategySelector
from pinnsolver.parser import parse_problem
from pinnsolver.spec import ProblemInput


def test_strategy_rules_and_modes():
    sel = StrategySelector()
    burgers = parse_problem(ProblemInput.from_dict(get("burgers")["problem"]))
    d = sel.select(burgers, "D")
    assert "R7" in d.rules_fired and d.adaptive and d.gpinn_candidate and d.fallback and d.max_attempts == 4
    assert d.stop_on_stall and not sel.select(burgers, "A").stop_on_stall
    a = sel.select(burgers, "A")
    assert not a.hard_constraints and not a.adaptive and a.lbfgs_iters == 0 and not a.fallback
    b, c = sel.select(burgers, "B"), sel.select(burgers, "C")
    assert b.hard_constraints and not b.adaptive and c.adaptive and c.hard_constraints
    # equal point budgets across A-D
    assert a.max_points == b.max_points == c.max_points == d.max_points
    assert a.n_collocation == a.max_points and c.n_collocation < c.max_points
    stiff = parse_problem(ProblemInput.from_dict(get("stiff")["problem"]))
    assert "R5" in sel.select(stiff, "D").rules_fired


def test_decay_proposed_solver_passes(tmp_path):
    r = solve(get("decay")["problem"], mode="D", seed=0, time_budget=30, runs_dir=str(tmp_path))
    assert r["status"] == "passed" and r["final_method"] == "pinn"
    assert r["validation"]["metrics"]["rel_l2"] < 1e-3
    assert r["validation"]["metrics"]["condition_max"] == 0.0  # hard IC
    for f in ("problem.json", "training_log.csv", "reference.json", "report.json", "report.md", "solution.png"):
        assert (tmp_path / r["run_id"] / f).exists()
    assert r["reproducibility"]["seed"] == 0 and r["reproducibility"]["versions"]["torch"]


def test_seed_reproducibility():
    # generous budget so both runs end by iteration count / convergence, not by the clock
    r1 = solve(get("decay")["problem"], mode="D", seed=7, time_budget=120, save=False)
    r2 = solve(get("decay")["problem"], mode="D", seed=7, time_budget=120, save=False)
    assert r1["budget"]["adam_iterations"] == r2["budget"]["adam_iterations"]
    assert r1["budget"]["lbfgs_iterations"] == r2["budget"]["lbfgs_iterations"]
    assert r1["validation"]["metrics"]["rel_l2"] == pytest.approx(r2["validation"]["metrics"]["rel_l2"], rel=1e-9)
    assert r1["strategy"]["config_fingerprint"] == r2["strategy"]["config_fingerprint"]


def test_time_marching_rescues_logistic():
    r = solve(get("logistic")["problem"], mode="D", seed=0, time_budget=30, save=False)
    actions = [a["action"] for a in r["attempts"]]
    assert r["status"] == "passed" and r["final_method"] == "pinn"
    assert "time_marching" in actions


def test_tiny_budget_triggers_labelled_fallback():
    r = solve(get("oscillator")["problem"], mode="D", seed=0, time_budget=1.0, save=False)
    assert r["status"] == "fallback" and r["final_method"] == "numerical_fallback"
    assert r["fallback"]["available"] and r["fallback"]["validated_against_exact"]
    assert r["fallback"]["validation"]["passed"]
    assert "NOT the PINN" in r["final_method_text"]


def test_mode_e_and_runner_rows():
    r = solve(get("heat")["problem"], mode="E", save=False)
    assert r["status"] == "passed" and r["final_method"] == "numerical" and r["attempts"] == []
    row = run_one("decay", "E", 0)
    assert row["status"] == "passed" and row["rel_l2"] < 1e-8
    md = summarize([row])
    assert "decay" in md and "n/a (no PINN)" in md


def test_api_end_to_end():
    from fastapi.testclient import TestClient
    from pinnsolver.api import app

    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    assert len(c.get("/api/benchmarks").json()) >= 9
    assert c.get("/api/info").json()["rule_table"][0]["id"] == "R1"
    bad = c.post("/api/validate", json={"equation": "u' = os.system(1)", "variables": ["t"], "domain": {"t": [0, 1]}, "conditions": ["u(0)=1"]})
    assert bad.status_code == 422
    ok = c.post("/api/validate", json=get("decay")["problem"])
    assert ok.status_code == 200 and ok.json()["kind"] == "ode_ivp"
    assert c.post("/api/jobs", json={"problem": get("decay")["problem"], "mode": "Z"}).status_code == 422
    assert c.post("/api/jobs", json={"problem": get("decay")["problem"], "time_budget": 5000}).status_code == 422
    j = c.post("/api/jobs", json={"problem": get("decay")["problem"], "mode": "E"}).json()
    for _ in range(100):
        s = c.get(f"/api/jobs/{j['job_id']}").json()
        if s["state"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert s["state"] == "done", s
    rep = c.get(f"/api/jobs/{j['job_id']}/report").json()
    assert rep["final_method"] == "numerical"
    assert c.get(f"/api/runs/{rep['run_id']}/report.json").status_code == 200
    assert c.get(f"/api/runs/{rep['run_id']}/secret.txt").status_code == 404
    assert c.get("/api/jobs/doesnotexist").status_code == 404


def test_api_cancel():
    from fastapi.testclient import TestClient
    from pinnsolver.api import app

    c = TestClient(app)
    j = c.post("/api/jobs", json={"problem": get("heat")["problem"], "mode": "A", "time_budget": 60}).json()
    time.sleep(1.0)
    c.post(f"/api/jobs/{j['job_id']}/cancel")
    for _ in range(200):
        s = c.get(f"/api/jobs/{j['job_id']}").json()
        if s["state"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert s["state"] == "cancelled"
