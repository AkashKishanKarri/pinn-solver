"""Builds the test-case catalogue: docs/PINN_Solver_Test_Cases.pdf and docs/PINN_Solver_Test_Cases.xlsx.

Every input is captured automatically, never copied by hand:
  1. The full pytest suite runs with docs/pinn_recorder.py, which records each problem, expression and
     solve the solver receives, together with the solver's actual response.
  2. The API edge-case requests from the final review are replayed against the FastAPI app.
  3. The benchmark suite (definitions + measured results) is read from pinnsolver/benchmarks.py and
     benchmark_results/results.csv.

Usage:  .venv/bin/python docs/build_testcases.py
"""
from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict
from datetime import date
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))

import build_docs as bd  # noqa: E402  (fonts, colours, paragraph styles)
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from pinnsolver import benchmarks  # noqa: E402

PDF_OUT = os.path.join(ROOT, "docs", "PINN_Solver_Test_Cases.pdf")
XLSX_OUT = os.path.join(ROOT, "docs", "PINN_Solver_Test_Cases.xlsx")
PAGE = landscape(A4)
MARGIN = 1.4 * cm
W = PAGE[0] - 2 * MARGIN

FILE_TITLES = OrderedDict([
    ("tests/test_expr_parser.py", "Grammar, classification and rejection reasons"),
    ("tests/test_friendly_input.py", "Friendly notation, well-posedness checks and scaling"),
    ("tests/test_edge_cases.py", "Edge cases: constants, limits, references, budgets"),
    ("tests/test_numerics.py", "Numerical correctness"),
    ("tests/test_solver_api.py", "Strategy, solver and REST API"),
    ("tests/test_api_hardening.py", "API hardening (final review)"),
    ("(background API job)", "Jobs finished by the API worker thread"),
])

# API requests sent during the final review (Section 8 of the documentation), replayed here live.
BASE = {"equation": "u' = -u", "variables": ["t"], "domain": {"t": [0, 1]}, "conditions": ["u(0) = 1"]}
NAN, INF = float("nan"), float("inf")


def _m(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


API_PROBES = [
    ("Valid first-order IVP", "/api/validate", BASE),
    ("Domain end is NaN", "/api/validate", _m(domain={"t": [NAN, 1]})),
    ("Domain end is Infinity", "/api/validate", _m(domain={"t": [0, INF]})),
    ("Domain reversed", "/api/validate", _m(domain={"t": [1, 0]})),
    ("Domain empty (lower = upper)", "/api/validate", _m(domain={"t": [1, 1]})),
    ("Domain too large", "/api/validate", _m(domain={"t": [0, 1e300]})),
    ("Domain too narrow", "/api/validate", _m(domain={"t": [0, 1e-300]})),
    ("Domain variable does not match", "/api/validate", _m(domain={"x": [0, 1]})),
    ("Parameter is NaN", "/api/validate", _m(equation="u' = -k*u", parameters={"k": NAN})),
    ("Parameter is Infinity", "/api/validate", _m(equation="u' = -k*u", parameters={"k": INF})),
    ("Parameter named like the unknown", "/api/validate", _m(parameters={"u": 1.0})),
    ("Parameter named like the variable", "/api/validate", _m(parameters={"t": 1.0})),
    ("Parameter named pi", "/api/validate", _m(equation="u' = -pi*u", parameters={"pi": 2.0})),
    ("Unused multi-letter parameter", "/api/validate", _m(parameters={"kk": 2.0})),
    ("Duplicate variables", "/api/validate", _m(variables=["t", "t"])),
    ("Variable is not a letter", "/api/validate", _m(variables=["1"])),
    ("Unknown named like the variable", "/api/validate", _m(dependent="t")),
    ("Unknown is a digit", "/api/validate", _m(dependent="1")),
    ("Empty condition", "/api/validate", _m(conditions=[""])),
    ("Duplicate condition", "/api/validate", _m(conditions=["u(0)=1", "u(0)=1"])),
    ("Conflicting conditions", "/api/validate", _m(conditions=["u(0)=1", "u(0)=2"])),
    ("Condition outside the domain", "/api/validate", _m(conditions=["u(5)=1"])),
    ("Condition at the end of an IVP", "/api/validate", _m(conditions=["u(1)=1"])),
    ("Control character in equation", "/api/validate", _m(equation="u' = -u\u0000")),
    ("Emoji in equation", "/api/validate", _m(equation="u' = \U0001F600")),
    ("150 nested parentheses", "/api/validate", _m(equation="u' = " + "(" * 150 + "u" + ")" * 150)),
    ("Third-order derivative", "/api/validate", _m(equation="u''' = -u")),
    ("Name is null", "/api/validate", _m(name=None)),
    ("Wrong exact solution supplied", "/api/validate", _m(exact="sin(t)")),
    ("Correct exact solution supplied", "/api/validate", _m(exact="exp(-t)")),
    ("Nonlinear in the highest derivative", "/api/validate", _m(equation="u'^2 = u")),
    ("PDE without time variable t", "/api/validate",
     {"equation": "u_xx = 0", "variables": ["x", "y"], "domain": {"x": [0, 1], "y": [0, 1]}, "conditions": ["u(0,y)=0"]}),
    ("Threshold is Infinity", "/api/jobs", {"problem": BASE, "time_budget": 5, "thresholds": {"rel_l2": INF}}),
    ("Threshold is negative", "/api/jobs", {"problem": BASE, "time_budget": 5, "thresholds": {"rel_l2": -1}}),
    ("Unknown threshold key", "/api/jobs", {"problem": BASE, "time_budget": 5, "thresholds": {"foo": 1}}),
    ("Time budget is NaN", "/api/jobs", {"problem": BASE, "time_budget": NAN}),
    ("Time budget is 0", "/api/jobs", {"problem": BASE, "time_budget": 0}),
    ("Time budget above 900 s", "/api/jobs", {"problem": BASE, "time_budget": 901}),
    ("Unknown mode", "/api/jobs", {"problem": BASE, "time_budget": 5, "mode": "Z"}),
    ("Unknown ablation", "/api/jobs", {"problem": BASE, "time_budget": 5, "ablate": ["x"]}),
    ("Negative seed", "/api/jobs", {"problem": BASE, "time_budget": 5, "seed": -1}),
    ("Unknown job id", "GET /api/jobs/nope", None),
    ("Path traversal in run files", "GET /api/runs/..%2F..%2Fetc/report.json", None),
    ("Run file not on the whitelist", "GET /api/runs/x/passwd", None),
    ("Cancel unknown job", "POST /api/jobs/zzz/cancel", None),
]


# ---------------------------------------------------------------- data collection
def record_test_suite(tmp: str) -> tuple[list, str]:
    out = os.path.join(tmp, "recorded.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "docs"), PINN_RECORD_OUT=out, PINN_RUNS_DIR=os.path.join(tmp, "runs"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "pinn_recorder"], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no output"
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        raise SystemExit(f"pytest failed: {summary}")
    recs, seen = [], set()
    for line in open(out):
        r = json.loads(line)
        f, _, name = r["test"].partition("::")
        r["file"], r["test_name"] = (f or "(background API job)"), (name or "API worker thread")
        key = json.dumps({k: v for k, v in r.items() if k not in ("training_s",)}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            recs.append(r)
    return recs, re.sub(r"\s+in [\d.]+s.*", "", summary)


def replay_api_probes(tmp: str) -> list:
    os.environ["PINN_RUNS_DIR"] = os.path.join(tmp, "probe_runs")
    from fastapi.testclient import TestClient

    from pinnsolver.api import app, manager
    client = TestClient(app)
    rows = []
    for title, path, body in API_PROBES:
        if body is None:
            method, url = path.split(" ", 1)
            r = client.request(method, url)
            shown = f"{method} {url}"
        else:
            r = client.post(path, content=json.dumps(body, allow_nan=True), headers={"Content-Type": "application/json"})
            shown = json.dumps(body, allow_nan=True, ensure_ascii=False)
        try:
            d = r.json()
        except ValueError:
            d = r.text
        if isinstance(d, dict) and "detail" in d:
            det = d["detail"]
            msg = det if isinstance(det, str) else "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in det)
        elif isinstance(d, dict) and "kind" in d:
            msg = f"accepted: {d['kind']} ({d.get('subtype')})" + (f"; warnings: {' / '.join(d['warnings'])}" if d.get("warnings") else "")
        elif isinstance(d, dict) and "job_id" in d:
            msg = f"job queued ({d['state']})"
            client.post(f"/api/jobs/{d['job_id']}/cancel")
        else:
            msg = str(d)[:200]
        rows.append({"title": title, "endpoint": path.split(" ")[0] if body is not None else path.split(" ")[0],
                     "request": shown, "status": r.status_code, "response": msg})
    for j in list(manager.jobs.values()):
        j.cancel.set()
    return rows


def benchmark_rows():
    with open(os.path.join(ROOT, "benchmark_results", "results.csv")) as f:
        res = list(csv.DictReader(f))
    out = []
    for b in benchmarks.BENCHMARKS:
        p = b["problem"]
        per_mode = {}
        for m in "ABCDE":
            rs = [r for r in res if r["problem"] == b["id"] and r["mode"] == m]
            if not rs:
                continue
            if m == "E":
                per_mode[m] = "numerical only"
            else:
                med = statistics.median(float(r["rel_l2"]) for r in rs)
                per_mode[m] = f"{sum(r['pinn_passed'] == 'True' for r in rs)}/{len(rs)} · {med:.1e}"
        out.append({"id": b["id"], "title": b["title"], "class": b["class"], "budget": b["budget_s"], "problem": p, "modes": per_mode})
    return out, len(res)


# ---------------------------------------------------------------- formatting helpers
def fmt_domain(d):
    return ", ".join(f"{k} ∈ [{v[0]:g}, {v[1]:g}]" if all(isinstance(x, (int, float)) for x in v) else f"{k} ∈ {v}" for k, v in d.items())


def fmt_extra(p):
    bits = []
    if p.get("parameters"):
        bits.append(", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in p["parameters"].items()))
    if p.get("dependent") and p["dependent"] != "u":
        bits.append(f"unknown: {p['dependent']}")
    if p.get("exact"):
        bits.append(f"exact: {p['exact']}")
    return "; ".join(bits)


def short_test(name):
    return name.replace("test_", "", 1).replace("_", " ")


def outcome_label(r):
    o = r["outcome"]
    return {"accepted": "Accepted", "rejected": "Rejected", "passed": "PASSED", "failed": "FAILED (expected)",
            "fallback": "Fallback (labelled)", "numerical_unvalidated": "Numerical, flagged",
            "error": "Cancelled" if "Cancelled" in (r.get("detail") or "") else "Error"}.get(o, o)


def clip(s, n=260):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- PDF
CELL = ParagraphStyle("c", parent=bd.CELL, fontSize=7.2, leading=9.2)
CELLM = ParagraphStyle("cm", parent=CELL, fontName="DVM", fontSize=6.9)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="DV-B")
OK_C, BAD_C, INFO_C = colors.HexColor("#e7f4ea"), colors.HexColor("#fbeaea"), colors.HexColor("#eef2f8")


def para(text, style=CELL):
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def grid(header, rows, widths, shade_col=None):
    data = [[Paragraph(escape(h), CELLB) for h in header]]
    styles = []
    for i, r in enumerate(rows, start=1):
        data.append([c if isinstance(c, Paragraph) else para(c, CELLM if j in (1, 2) and header[j] in ("Equation", "Input", "Request", "Expression") else CELL)
                     for j, c in enumerate(r)])
        if shade_col is not None:
            v = str(r[shade_col])
            col = OK_C if v.startswith(("Accepted", "PASSED", "200", "202", "Cancelled", "Numerical")) else \
                BAD_C if v.startswith(("Rejected", "4", "FAILED", "Fallback", "Error")) else INFO_C
            styles.append(("BACKGROUND", (shade_col, i), (shade_col, i), col))
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, bd.RULE),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, bd.ACCENT),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)] + styles))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(bd.MUTED)
    canvas.drawString(MARGIN, 0.8 * cm, "PINN Solver — Test-case catalogue (generated by docs/build_testcases.py)")
    canvas.drawRightString(PAGE[0] - MARGIN, 0.8 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(recs, suite_summary, probes, bench, n_bench_runs):
    H1 = ParagraphStyle("h1", parent=bd.H1, fontSize=15, leading=19)
    story = []
    n_val = [r for r in recs if r["kind"] == "validate"]
    n_expr = [r for r in recs if r["kind"] == "expression"]
    n_dict = [r for r in recs if r["kind"] == "problem_dict"]
    n_solve = [r for r in recs if r["kind"] == "solve"]
    total = len(recs) + len(probes) + len(bench)

    story += [Paragraph("PINN Solver — Test-Case Catalogue", bd.TITLE), Spacer(1, 6),
              Paragraph("Every input given to the solver by the automated tests, the API edge-case checks and the benchmark "
                        "suite, with the solver's actual response.", bd.SUB), Spacer(1, 14)]
    summary = [["Generated", date.today().strftime("%d %B %Y")],
               ["Test suite", f"{suite_summary} (all assertions hold)"],
               ["Environment", f"Python {platform.python_version()}, {platform.system()} {platform.machine()}"],
               ["Inputs catalogued", f"{total} in total"],
               ["  A. Problem validation", f"{len(n_val)} problems ({sum(r['outcome'] == 'accepted' for r in n_val)} accepted, "
                                          f"{sum(r['outcome'] == 'rejected' for r in n_val)} rejected with a reason)"],
               ["  B. Expressions", f"{len(n_expr)} expressions given directly to the expression parser"],
               ["  C. Malformed problem data", f"{len(n_dict)} malformed problem dictionaries"],
               ["  D. End-to-end solves", f"{len(n_solve)} solves run by the tests (training + validation)"],
               ["  E. API requests", f"{len(probes)} HTTP requests replayed against the REST API"],
               ["  F. Benchmarks", f"{len(bench)} benchmark problems ({n_bench_runs} recorded runs: 5 configurations × 3 seeds)"]]
    st = Table([[para(a, CELLB), para(b)] for a, b in summary], colWidths=[0.2 * W, 0.8 * W])
    st.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, bd.RULE), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [st, Spacer(1, 12),
              Paragraph("<b>How this catalogue is made.</b> Nothing here is typed by hand. "
                        "<font face='DVM'>docs/build_testcases.py</font> runs the full pytest suite with a recording plugin "
                        "(<font face='DVM'>docs/pinn_recorder.py</font>). The plugin logs each problem, expression and solve "
                        "the solver receives, and the response it actually gave. Because every test passed, each response "
                        "shown is the one its test asserts. Rejections show the solver's exact message. The API requests are "
                        "replayed live, and the benchmark figures come from "
                        "<font face='DVM'>benchmark_results/results.csv</font>. The same data is in the Excel workbook "
                        "<font face='DVM'>docs/PINN_Solver_Test_Cases.xlsx</font>.", bd.BODY),
              Paragraph("Colour key for the result column: green = accepted, passed or handled as intended; red = rejected "
                        "with a reason, or an intentional failure (for example a 1-second budget); blue = other.", bd.SMALL),
              PageBreak()]

    # A. validation, grouped by test file
    story.append(Paragraph(f"A. Problem validation ({len(n_val)} inputs)", H1))
    story.append(Paragraph("Each row is one problem passed to <font face='DVM'>Parser.validate</font>, which checks the "
                           "equation, classifies it, and checks well-posedness before any training.", bd.BODY))
    by_file = defaultdict(list)
    for r in n_val:
        by_file[r["file"]].append(r)
    k = 0
    for f, title in FILE_TITLES.items():
        rs = by_file.get(f)
        if not rs:
            continue
        story.append(Paragraph(f"{title} <font color='#5b6470' size='8'>({f}, {len(rs)})</font>", bd.H3))
        rows = []
        for r in rs:
            k += 1
            p = r["problem"]
            resp = r["detail"] + (("\nWarning: " + " / ".join(r["warnings"])) if r.get("warnings") else "")
            rows.append([f"A{k}", p["equation"], "\n".join(p["conditions"]), fmt_domain(p["domain"]) + ("\n" + fmt_extra(p) if fmt_extra(p) else ""),
                         short_test(r["test_name"]), outcome_label(r), clip(resp, 300)])
        story.append(grid(["#", "Equation", "Conditions", "Domain / extras", "Test", "Result", "Solver response"],
                          rows, [0.04, 0.17, 0.13, 0.15, 0.14, 0.07, 0.30], shade_col=5))
    story.append(PageBreak())

    # B. expressions
    story.append(Paragraph(f"B. Expression parser ({len(n_expr)} inputs)", H1))
    story.append(Paragraph("Expressions given directly to the restricted grammar (no <font face='DVM'>eval</font>). "
                           "Accepted expressions show the parsed form.", bd.BODY))
    story.append(grid(["#", "Expression", "Test", "Result", "Parsed form / reason"],
                      [[f"B{i}", repr(r["text"]), short_test(r["test_name"]), outcome_label(r), clip(r["detail"], 200)]
                       for i, r in enumerate(n_expr, 1)], [0.05, 0.30, 0.18, 0.08, 0.39], shade_col=3))

    # C. malformed dicts
    story.append(Paragraph(f"C. Malformed problem data ({len(n_dict)} inputs)", H1))
    story.append(grid(["#", "Input", "Test", "Result", "Reason"],
                      [[f"C{i}", clip(json.dumps(r["raw"]) if not isinstance(r["raw"], str) else r["raw"], 200), short_test(r["test_name"]),
                        outcome_label(r), r["detail"]] for i, r in enumerate(n_dict, 1)], [0.05, 0.35, 0.18, 0.08, 0.34], shade_col=3))
    story.append(PageBreak())

    # D. solves
    story.append(Paragraph(f"D. End-to-end solves ({len(n_solve)} runs)", H1))
    story.append(Paragraph("Full runs of <font face='DVM'>Solver.solve</font>: training, validation, retries and fallback. "
                           "Several runs are deliberately given a 1-second budget, to check that the solver reports failure "
                           "or a labelled fallback and never crashes.", bd.BODY))
    rows = []
    for i, r in enumerate(n_solve, 1):
        p = r["problem"]
        rl = r.get("rel_l2")
        res = (f"rel. L2 {rl:.2e}" if isinstance(rl, (int, float)) else "") + \
              (f"; {r['attempts']} attempt(s); {r['training_s']:.1f} s" if r.get("attempts") is not None and r.get("training_s") is not None else "")
        rows.append([f"D{i}", p["equation"], "\n".join(p["conditions"]), fmt_domain(p["domain"]) + ("\n" + fmt_extra(p) if fmt_extra(p) else ""),
                     f"{r['mode']} / {r['time_budget'] if r['time_budget'] is not None else 'default'}",
                     short_test(r["test_name"]), outcome_label(r), clip((r.get("detail") or "") + ("\n" + res if res else ""), 220)])
    story.append(grid(["#", "Equation", "Conditions", "Domain / extras", "Mode / budget (s)", "Test", "Result", "Details"],
                      rows, [0.04, 0.15, 0.12, 0.14, 0.08, 0.14, 0.08, 0.25], shade_col=6))
    story.append(PageBreak())

    # E. API
    story.append(Paragraph(f"E. REST API requests ({len(probes)} requests)", H1))
    story.append(Paragraph("Requests replayed against the FastAPI app. NaN and Infinity are sent as the JSON literals "
                           "<font face='DVM'>NaN</font> and <font face='DVM'>Infinity</font>, which Python's JSON parser accepts. "
                           "Every malformed request gets a 4xx answer with a reason; none causes a server error.", bd.BODY))
    story.append(grid(["#", "Case", "Request", "HTTP", "Response"],
                      [[f"E{i}", p["title"], clip(p["request"], 230), p["status"], clip(p["response"], 260)] for i, p in enumerate(probes, 1)],
                      [0.04, 0.17, 0.38, 0.05, 0.36], shade_col=3))
    story.append(PageBreak())

    # F. benchmarks
    story.append(Paragraph(f"F. Benchmark suite ({len(bench)} problems, {n_bench_runs} runs)", H1))
    story.append(Paragraph("Each cell gives the number of seeds in which the PINN met every threshold, then the median "
                           "relative L2 error. A–C are fixed baseline recipes, D is the proposed solver, and E is the "
                           "classical numerical solver alone.", bd.BODY))
    rows = []
    for i, b in enumerate(bench, 1):
        p = b["problem"]
        rows.append([f"F{i}", f"{b['title']}\n({b['class']})", p["equation"], "\n".join(p["conditions"]),
                     fmt_domain(p["domain"]) + ("\n" + fmt_extra(p) if fmt_extra(p) else ""), f"{b['budget']} s"] +
                    [b["modes"].get(m, "") for m in "ABCD"])
    story.append(grid(["#", "Benchmark", "Equation", "Conditions", "Domain / extras", "Budget", "A", "B", "C", "D (proposed)"],
                      rows, [0.03, 0.13, 0.14, 0.12, 0.17, 0.05, 0.08, 0.08, 0.08, 0.12]))

    doc = SimpleDocTemplate(PDF_OUT, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=1.3 * cm,
                            title="PINN Solver — Test-Case Catalogue", author="Akash Kishan Karri")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------- Excel
def build_xlsx(recs, suite_summary, probes, bench, n_bench_runs):
    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="E6ECF5")
    ok_fill, bad_fill = PatternFill("solid", fgColor="E7F4EA"), PatternFill("solid", fgColor="FBEAEA")

    def sheet(title, header, rows, widths, result_col=None):
        ws = wb.create_sheet(title)
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill = Font(bold=True), head_fill
        for r in rows:
            ws.append([str(x) if isinstance(x, (list, dict)) else x for x in r])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
            if result_col is not None:
                v = str(row[result_col].value)
                row[result_col].fill = ok_fill if v.startswith(("Accepted", "PASSED", "200", "202", "Cancelled", "Numerical")) else bad_fill
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        return ws

    ws = wb.active
    ws.title = "Summary"
    for row in [["PINN Solver — Test-Case Catalogue"], ["Generated", date.today().isoformat()], ["Test suite", suite_summary],
                ["Source", "docs/build_testcases.py (recorded automatically from the test suite, API replay and benchmark results)"],
                [], ["Sheet", "Inputs"],
                ["Validation", sum(r["kind"] == "validate" for r in recs)], ["Expressions", sum(r["kind"] == "expression" for r in recs)],
                ["Malformed data", sum(r["kind"] == "problem_dict" for r in recs)], ["Solves", sum(r["kind"] == "solve" for r in recs)],
                ["API requests", len(probes)], ["Benchmarks", f"{len(bench)} problems, {n_bench_runs} runs"]]:
        ws.append(row)
    ws["A1"].font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 90

    val = [r for r in recs if r["kind"] == "validate"]
    sheet("Validation", ["#", "Test file", "Test", "Equation", "Variables", "Unknown", "Domain", "Conditions", "Parameters", "Exact",
                         "Result", "Solver response", "Warnings"],
          [[f"A{i}", r["file"], r["test_name"], r["problem"]["equation"], ", ".join(r["problem"]["variables"]), r["problem"]["dependent"],
            fmt_domain(r["problem"]["domain"]), "\n".join(r["problem"]["conditions"]),
            ", ".join(f"{k}={v}" for k, v in r["problem"]["parameters"].items()), r["problem"]["exact"] or "",
            outcome_label(r), r["detail"], " / ".join(r.get("warnings") or [])] for i, r in enumerate(val, 1)],
          [6, 24, 36, 30, 10, 8, 20, 26, 14, 18, 11, 60, 30], result_col=10)
    sheet("Expressions", ["#", "Test", "Expression", "Result", "Parsed form / reason"],
          [[f"B{i}", r["test_name"], r["text"], outcome_label(r), r["detail"]] for i, r in enumerate([r for r in recs if r["kind"] == "expression"], 1)],
          [6, 36, 40, 11, 60], result_col=3)
    sheet("Malformed data", ["#", "Test", "Input", "Result", "Reason"],
          [[f"C{i}", r["test_name"], json.dumps(r["raw"]) if not isinstance(r["raw"], str) else r["raw"], outcome_label(r), r["detail"]]
           for i, r in enumerate([r for r in recs if r["kind"] == "problem_dict"], 1)], [6, 36, 60, 11, 60], result_col=3)
    sheet("Solves", ["#", "Test file", "Test", "Equation", "Domain", "Conditions", "Parameters / exact", "Mode", "Time budget (s)",
                     "Status", "Final method", "Rel. L2", "Max IC/BC error", "Attempts", "Training (s)", "Details"],
          [[f"D{i}", r["file"], r["test_name"], r["problem"]["equation"], fmt_domain(r["problem"]["domain"]), "\n".join(r["problem"]["conditions"]),
            fmt_extra(r["problem"]), r["mode"], r["time_budget"] if r["time_budget"] is not None else "default", outcome_label(r),
            r.get("final_method"), r.get("rel_l2"), r.get("condition_max"), r.get("attempts"), r.get("training_s"), r.get("detail")]
           for i, r in enumerate([r for r in recs if r["kind"] == "solve"], 1)],
          [6, 22, 34, 26, 20, 26, 20, 6, 10, 14, 12, 11, 11, 8, 10, 60], result_col=9)
    sheet("API requests", ["#", "Case", "Endpoint", "Request body", "HTTP status", "Response"],
          [[f"E{i}", p["title"], p["endpoint"], p["request"], p["status"], p["response"]] for i, p in enumerate(probes, 1)],
          [6, 30, 22, 70, 10, 70], result_col=4)
    sheet("Benchmarks", ["#", "ID", "Title", "Class", "Equation", "Domain", "Conditions", "Parameters / exact", "Budget (s)",
                         "A passed · median rel. L2", "B", "C", "D (proposed)", "E"],
          [[f"F{i}", b["id"], b["title"], b["class"], b["problem"]["equation"], fmt_domain(b["problem"]["domain"]),
            "\n".join(b["problem"]["conditions"]), fmt_extra(b["problem"]), b["budget"]] + [b["modes"].get(m, "") for m in "ABCDE"]
           for i, b in enumerate(bench, 1)], [6, 12, 26, 22, 26, 20, 26, 26, 9, 16, 16, 16, 16, 14])
    wb.save(XLSX_OUT)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("Running the test suite with the recorder...")
        recs, suite = record_test_suite(tmp)
        print(f"  {suite}; {len(recs)} unique inputs recorded")
        print("Replaying API requests...")
        probes = replay_api_probes(tmp)
        bench, n_runs = benchmark_rows()
        build_pdf(recs, suite, probes, bench, n_runs)
        build_xlsx(recs, suite, probes, bench, n_runs)
    print(PDF_OUT)
    print(XLSX_OUT)


if __name__ == "__main__":
    main()
