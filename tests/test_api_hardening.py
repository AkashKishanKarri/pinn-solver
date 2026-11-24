"""Regression tests for request-level edge cases found in the final review."""
import json

from fastapi.testclient import TestClient

from pinnsolver.api import app

client = TestClient(app)
BASE = {"equation": "u' = -u", "variables": ["t"], "domain": {"t": [0, 1]}, "conditions": ["u(0) = 1"]}


def _post(path, body):
    # allow_nan: Python's JSON parser (and therefore FastAPI) accepts NaN/Infinity literals.
    return client.post(path, content=json.dumps(body, allow_nan=True), headers={"Content-Type": "application/json"})


def test_non_finite_time_budget_is_a_422_not_a_500():
    for bad in (float("nan"), float("inf")):
        r = _post("/api/jobs", {"problem": BASE, "time_budget": bad})
        assert r.status_code == 422, r.text
        assert "input" not in r.json()["detail"][0]


def test_thresholds_must_be_known_finite_and_positive():
    for thr in ({"rel_l2": float("inf")}, {"rel_l2": float("nan")}, {"rel_l2": -1}, {"rel_l2": 0}, {"foo": 1e-3}):
        r = _post("/api/jobs", {"problem": BASE, "thresholds": thr})
        assert r.status_code == 422, (thr, r.text)


def test_conflicting_conditions_are_named_as_such():
    r = _post("/api/validate", dict(BASE, conditions=["u(0) = 1", "u(0) = 2"]))
    assert r.status_code == 422 and "conflicting" in r.json()["detail"]
    pde = {"equation": "u_t = u_xx", "variables": ["x", "t"], "domain": {"x": [0, 1], "t": [0, 1]},
           "conditions": ["u(x, 0) = sin(pi*x)", "u(0, t) = 0", "u(0, t) = 1", "u(1, t) = 0"]}
    r = _post("/api/validate", pde)
    assert r.status_code == 422 and "conflicting" in r.json()["detail"]


def test_non_finite_problem_numbers_are_rejected():
    for body in (dict(BASE, domain={"t": [0, float("nan")]}), dict(BASE, equation="u' = -k*u", parameters={"k": float("inf")})):
        assert _post("/api/validate", body).status_code == 422
