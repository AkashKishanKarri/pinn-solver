"""FastAPI backend (report Section 7; brief Section 9).

The numerical core is separate from the API: this module only validates
requests, runs solver jobs asynchronously in a bounded worker pool, and
serves reports. Limits: request sizes (pydantic), expression grammar and
complexity (parser), time budget (<= 900 s per job), point budget, one
training job at a time with at most ``MAX_QUEUED`` waiting.

Run:  uvicorn pinnsolver.api:app --port 8000
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Dict, List, Literal, Optional

import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import benchmarks
from .expr import CONSTANTS, FUNCTIONS, MAX_EXPR_CHARS, MAX_NODES
from .parser import Parser, ValidationError
from .report import _clean
from .solver import ABLATIONS, Solver
from .spec import ProblemInput
from .strategy import DEFAULT_THRESHOLDS, DEFAULT_TIME_BUDGET, MAX_POINTS, MAX_TIME_BUDGET, MODE_NAMES, RULE_TABLE
from .training import Cancelled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("PINN_RUNS_DIR", os.path.join(ROOT, "runs"))
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")
MAX_CONCURRENT = int(os.environ.get("PINN_MAX_CONCURRENT", "1"))
MAX_QUEUED = int(os.environ.get("PINN_MAX_QUEUED", "8"))
MAX_JOBS_KEPT = 50
MAX_RUNS_KEPT = int(os.environ.get("PINN_MAX_RUNS_KEPT", "200"))  # saved run folders on disk
THRESHOLD_KEYS = frozenset(k for d in DEFAULT_THRESHOLDS.values() for k in d)
TRAINING_TRACE_KEPT = 600


# ----------------------------------------------------------------------
# request models
# ----------------------------------------------------------------------
class ProblemModel(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    equation: str = Field(min_length=1, max_length=MAX_EXPR_CHARS)
    variables: List[str] = Field(min_length=1, max_length=2)
    dependent: str = Field(default="u", min_length=1, max_length=1)
    domain: Dict[str, List[float]]
    conditions: List[str] = Field(min_length=1, max_length=6)
    parameters: Dict[str, float] = Field(default_factory=dict)
    exact: Optional[str] = Field(default=None, max_length=MAX_EXPR_CHARS)

    @field_validator("domain")
    @classmethod
    def _domain(cls, v):
        if not 1 <= len(v) <= 2:
            raise ValueError("domain must have one or two variables")
        for k, iv in v.items():
            if len(iv) != 2:
                raise ValueError(f"domain of {k} must be [lower, upper]")
        return v

    @field_validator("conditions")
    @classmethod
    def _conds(cls, v):
        for c in v:
            if len(c) > MAX_EXPR_CHARS:
                raise ValueError("condition too long")
        return v

    @field_validator("parameters")
    @classmethod
    def _params(cls, v):
        if len(v) > 10:
            raise ValueError("at most 10 parameters")
        return v

    def to_input(self) -> ProblemInput:
        return ProblemInput.from_dict(self.model_dump())


class JobRequest(BaseModel):
    problem: ProblemModel
    mode: Literal["A", "B", "C", "D", "E"] = "D"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    time_budget: Optional[float] = Field(default=None, gt=0, le=MAX_TIME_BUDGET, allow_inf_nan=False)
    thresholds: Optional[Dict[str, float]] = None
    ablate: List[str] = Field(default_factory=list)

    @field_validator("thresholds")
    @classmethod
    def _thresholds(cls, v):
        # A non-finite or non-positive threshold would make every result "pass" (or none);
        # an unknown key would be silently ignored. Both are rejected.
        if v is None:
            return v
        bad = [k for k in v if k not in THRESHOLD_KEYS]
        if bad:
            raise ValueError(f"unknown threshold(s) {bad}; allowed: {sorted(THRESHOLD_KEYS)}")
        for k, x in v.items():
            if not (math.isfinite(x) and 0 < x <= 1e3):
                raise ValueError(f"threshold {k!r} must be a finite number in (0, 1000]")
        return v

    @field_validator("ablate")
    @classmethod
    def _ablate(cls, v):
        bad = [a for a in v if a not in ABLATIONS]
        if bad:
            raise ValueError(f"unknown ablation(s) {bad}")
        return v


# ----------------------------------------------------------------------
# job manager
# ----------------------------------------------------------------------
class Job:
    def __init__(self, req: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.req = req
        self.state = "queued"  # queued | running | done | error | cancelled
        self.stage = "queued"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.error: Optional[str] = None
        self.report: Optional[dict] = None
        self.events: Deque[dict] = deque(maxlen=50)
        self.trace: Deque[dict] = deque(maxlen=TRAINING_TRACE_KEPT)
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def progress(self, event: dict) -> None:
        if self.cancel.is_set():
            raise Cancelled("Cancelled by the user.")
        with self.lock:
            stage = event.get("stage", self.stage)
            self.stage = stage
            if stage == "training":
                self.trace.append({k: event.get(k) for k in ("attempt", "phase", "iteration", "time_s", "n_points", "loss_total", "loss_residual")})
            else:
                self.events.append({"t": round(time.time() - (self.started or self.created), 2), **_jsonable(event)})

    def summary(self) -> dict:
        with self.lock:
            p = self.req.problem
            return {
                "job_id": self.id, "state": self.state, "stage": self.stage, "mode": self.req.mode, "seed": self.req.seed,
                "problem_name": p.name or p.equation, "created": self.created,
                "elapsed_s": round((self.finished or time.time()) - (self.started or time.time()), 2) if self.started else 0.0,
                "error": self.error, "report_available": self.report is not None,
                "status": self.report.get("status") if self.report else None,
            }

    def detail(self) -> dict:
        d = self.summary()
        with self.lock:
            d["events"] = list(self.events)
            d["trace"] = list(self.trace)
        return d


def _jsonable(e: dict) -> dict:
    out = {}
    for k, v in e.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool)) or x is None else str(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {kk: (vv if isinstance(vv, (str, int, float, bool)) or vv is None else str(vv)) for kk, vv in v.items()}
        else:
            out[k] = str(v)
    return out


def _prune_runs() -> None:
    """Keep at most MAX_RUNS_KEPT saved run folders so a long-lived deployment cannot fill its disk."""
    try:
        runs = [os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)]
        runs = sorted((r for r in runs if os.path.isdir(r)), key=os.path.getmtime)
        for r in runs[:max(0, len(runs) - MAX_RUNS_KEPT)]:
            shutil.rmtree(r, ignore_errors=True)
    except OSError:
        pass


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.order: Deque[str] = deque()
        self.pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="pinn-job")
        self.lock = threading.Lock()

    def pending(self) -> int:
        return sum(1 for j in self.jobs.values() if j.state in ("queued", "running"))

    def submit(self, req: JobRequest) -> Job:
        with self.lock:
            if self.pending() >= MAX_QUEUED + MAX_CONCURRENT:
                raise HTTPException(status_code=429, detail=f"Too many jobs in progress (limit {MAX_QUEUED + MAX_CONCURRENT}). Try again later.")
            job = Job(req)
            self.jobs[job.id] = job
            self.order.append(job.id)
            if len(self.order) > MAX_JOBS_KEPT:  # forget the oldest finished jobs, never in-progress ones
                excess = len(self.order) - MAX_JOBS_KEPT
                for old in list(self.order):
                    if excess == 0:
                        break
                    if self.jobs[old].state not in ("queued", "running"):
                        self.order.remove(old)
                        self.jobs.pop(old, None)
                        excess -= 1
        self.pool.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        if job.cancel.is_set():
            job.state, job.finished = "cancelled", time.time()
            return
        job.state, job.started = "running", time.time()
        try:
            solver = Solver(progress=job.progress, runs_dir=RUNS_DIR, save=True)
            report = solver.solve(job.req.problem.to_input(), mode=job.req.mode, seed=job.req.seed,
                                  time_budget=job.req.time_budget, thresholds=job.req.thresholds, ablate=job.req.ablate)
            with job.lock:
                job.report = report
                job.state, job.stage = "done", "done"
        except Cancelled:
            job.state, job.stage = "cancelled", "cancelled"
        except ValidationError as exc:
            job.state, job.error = "error", str(exc)
        except Exception as exc:  # reported to the user, never hidden
            job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished = time.time()
            _prune_runs()

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job


manager = JobManager()
app = FastAPI(title="Intelligent Differential Equation Solver (PINN)", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def _request_validation_error(request: Request, exc: RequestValidationError):
    # The default handler echoes the offending input back; a NaN/Infinity input (accepted by the
    # JSON parser) cannot be serialised and turned a 422 into a 500. Drop the raw input instead.
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx", "url")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


# ----------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "jobs_in_progress": manager.pending(), "max_concurrent": MAX_CONCURRENT, "max_queued": MAX_QUEUED}


@app.get("/api/benchmarks")
def list_benchmarks():
    return [{k: b[k] for k in ("id", "title", "class", "advanced", "description", "problem", "budget_s")} for b in benchmarks.BENCHMARKS]


@app.get("/api/info")
def info():
    return {
        "modes": MODE_NAMES,
        "rule_table": RULE_TABLE,
        "thresholds": DEFAULT_THRESHOLDS,
        "default_time_budget_s": DEFAULT_TIME_BUDGET,
        "limits": {"max_time_budget_s": MAX_TIME_BUDGET, "max_points": MAX_POINTS, "max_expression_chars": MAX_EXPR_CHARS,
                   "max_expression_nodes": MAX_NODES, "max_derivative_order": 2, "max_concurrent_jobs": MAX_CONCURRENT,
                   "max_queued_jobs": MAX_QUEUED},
        "grammar": {"functions": sorted(FUNCTIONS), "constants": sorted(CONSTANTS), "operators": ["+", "-", "*", "/", "^", "**", "( )"],
                    "derivatives": ["u'", "u''", "u_t", "u_x", "u_xx"]},
        "ablations": list(ABLATIONS),
    }


@app.post("/api/validate")
def validate(problem: ProblemModel):
    try:
        spec = Parser(problem.to_input()).validate()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _clean(spec.to_dict())  # NaN/Inf are not valid JSON


@app.post("/api/jobs", status_code=202)
def create_job(req: JobRequest):
    try:  # fail early, synchronously, on invalid problems
        Parser(req.problem.to_input()).validate()
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    job = manager.submit(req)
    return job.summary()


@app.get("/api/jobs")
def list_jobs():
    with manager.lock:  # submit() mutates the deque from other request threads
        jobs = [manager.jobs[j] for j in reversed(manager.order) if j in manager.jobs]
    return [j.summary() for j in jobs]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return manager.get(job_id).detail()


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    job = manager.get(job_id)
    if job.report is None:
        raise HTTPException(status_code=409, detail=f"Report not available (job state: {job.state}).")
    return JSONResponse(job.report)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = manager.get(job_id)
    job.cancel.set()
    if job.state == "queued":
        job.state = "cancelled"
    return job.summary()


@app.get("/api/runs/{run_id}/{filename}")
def run_file(run_id: str, filename: str):
    allowed = {"report.json", "report.md", "problem.json", "reference.json", "training_log.csv", "solution.png", "loss.png"}
    if filename not in allowed or not all(c.isalnum() or c in "-_" for c in run_id):
        raise HTTPException(status_code=404, detail="Not found.")
    path = os.path.realpath(os.path.join(RUNS_DIR, run_id, filename))
    if not path.startswith(os.path.realpath(RUNS_DIR) + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path)


if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
