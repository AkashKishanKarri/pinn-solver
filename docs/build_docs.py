"""Builds docs/PINN_Solver_Documentation.pdf.

Usage:
    .venv/bin/pip install -r docs/requirements-docs.txt
    .venv/bin/python docs/build_docs.py [--runs DIR]

--runs points at a directory containing run folders produced by
`python -m pinnsolver --benchmark <id> --runs-dir DIR/<id>`; their solution and
loss plots are embedded as example output (optional).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pinnsolver import benchmarks  # noqa: E402
from pinnsolver.strategy import DEFAULT_THRESHOLDS, DEFAULT_TIME_BUDGET, MODE_NAMES, RULE_TABLE  # noqa: E402

OUT = os.path.join(ROOT, "docs", "PINN_Solver_Documentation.pdf")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
LIVE_URL = os.environ.get("PINN_LIVE_URL", "")
REPO_URL = "https://github.com/AkashKishanKarri/pinn-solver"

# ---------------------------------------------------------------- fonts & styles
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#5b6470")
ACCENT = colors.HexColor("#2457a6")
RULE = colors.HexColor("#d5dae1")
PANEL = colors.HexColor("#f3f5f8")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.6, leading=14, textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=11, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.0, leading=10.4, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=17, leading=22, spaceBefore=4, spaceAfter=10, textColor=ACCENT)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=12, leading=16, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=10, leading=14, spaceBefore=6, spaceAfter=3)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CODE = ParagraphStyle("code", parent=BODY, fontName="DVM", fontSize=7.8, leading=10.2, backColor=PANEL,
                      borderPadding=(6, 6, 6, 6), leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
CAPTION = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=26, leading=32, textColor=INK)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=13, leading=18, textColor=MUTED)

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BUL):
    return [Paragraph(t, style, bulletText="•") for t in items]


def code(text):
    return Preformatted(text.strip("\n"), CODE)


def table(rows, widths, header=True, font=CELL, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if (header and i == 0) else font) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafbfc")))
    t.setStyle(TableStyle(st))
    return t


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            level = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.2 * cm, "Intelligent Differential Equation Solver Using PINNs — Technical Documentation")
    canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------- figures
def load_results():
    with open(os.path.join(ROOT, "benchmark_results", "results.csv")) as f:
        return list(csv.DictReader(f))


def benchmark_chart(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    probs = [b["id"] for b in benchmarks.BENCHMARKS]
    modes = ["A", "B", "C", "D"]
    palette = {"A": "#9aa5b1", "B": "#e0a458", "C": "#6aa6d8", "D": "#2457a6"}
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=200)
    w = 0.2
    for j, m in enumerate(modes):
        vals = []
        for p in probs:
            xs = [float(r["rel_l2"]) for r in rows if r["problem"] == p and r["mode"] == m and r["rel_l2"] not in ("", "nan")]
            vals.append(statistics.median(xs) if xs else float("nan"))
        ax.bar([i + (j - 1.5) * w for i in range(len(probs))], vals, w, label=f"{m}", color=palette[m])
    ax.set_yscale("log")
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(probs, fontsize=8)
    ax.set_ylabel("median relative L2 error", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(1e-3, color="#c0392b", lw=0.8, ls="--")
    ax.text(4.5, 1.25e-3, "ODE/BVP threshold 1e-3", fontsize=6.5, color="#c0392b", ha="center")
    ax.legend(title="configuration", fontsize=7, title_fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e3e6ea", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "benchmark_rel_l2.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def pass_table(rows):
    probs = [b["id"] for b in benchmarks.BENCHMARKS]
    out = [["Problem", "A", "B", "C", "D", "D: median rel. L2", "D: mean time (s)"]]
    for p in probs:
        line = [p]
        for m in "ABCD":
            rs = [r for r in rows if r["problem"] == p and r["mode"] == m]
            line.append(f"{sum(r['pinn_passed'] == 'True' for r in rs)}/{len(rs)}")
        d = [r for r in rows if r["problem"] == p and r["mode"] == "D"]
        line.append(f"{statistics.median(float(r['rel_l2']) for r in d):.2e}")
        line.append(f"{statistics.mean(float(r['training_time_s']) for r in d):.1f}")
        out.append(line)
    tot = ["<b>Total</b>"]
    for m in "ABCD":
        rs = [r for r in rows if r["mode"] == m]
        tot.append(f"<b>{sum(r['pinn_passed'] == 'True' for r in rs)}/{len(rs)}</b>")
    out.append(tot + ["", ""])
    return out


def architecture_diagram():
    d = Drawing(CONTENT_W, 205)
    box_w, box_h = 98, 34

    def box(x, y, title, sub, fill="#e6ecf5"):
        d.add(Rect(x, y, box_w, box_h, rx=4, ry=4, fillColor=colors.HexColor(fill), strokeColor=ACCENT, strokeWidth=0.8))
        d.add(String(x + box_w / 2, y + 20, title, fontName="DV-B", fontSize=7.6, textAnchor="middle", fillColor=INK))
        d.add(String(x + box_w / 2, y + 8, sub, fontName="DV", fontSize=6.4, textAnchor="middle", fillColor=MUTED))
        return (x, y)

    def arrow(x1, y1, x2, y2, label=None):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.9))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        s = 5
        d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                       x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)], fillColor=MUTED, strokeColor=MUTED))
        if label:
            d.add(String((x1 + x2) / 2 + 3, (y1 + y2) / 2 + 3, label, fontName="DV", fontSize=6.2, fillColor=MUTED))

    gx = (CONTENT_W - 4 * box_w) / 3
    xs = [i * (box_w + gx) for i in range(4)]
    box(xs[0], 160, "React UI", "Vite · KaTeX · Plotly", "#fdf3e1")
    box(xs[0], 105, "CLI", "python -m pinnsolver", "#fdf3e1")
    box(xs[1], 160, "FastAPI (api.py)", "validation · job queue")
    box(xs[1], 105, "Solver.solve", "control loop (solver.py)")
    box(xs[2], 160, "Parser / expr.py", "M1–M3 · safe grammar")
    box(xs[2], 105, "StrategySelector", "M4 · rules R1–R9")
    box(xs[3], 160, "References", "M8 · SciPy / FD / exact")
    box(xs[3], 105, "Trainer + PINN", "M5–M6 · Adam→L-BFGS, RAR")
    box(xs[3], 50, "Validator", "M7 · fresh points")
    box(xs[2], 50, "ReportGenerator", "M9 · json/md/csv/png")
    box(xs[1], 0, "runs/<id>/", "stores D1–D5", "#eef6ee")
    arrow(xs[0] + box_w, 177, xs[1], 177, "REST")
    arrow(xs[0] + box_w, 122, xs[1], 122)
    arrow(xs[1] + box_w / 2, 160, xs[1] + box_w / 2, 139)
    arrow(xs[1] + box_w, 130, xs[2], 168, "1")
    arrow(xs[1] + box_w, 122, xs[2], 122, "2")
    arrow(xs[1] + box_w, 114, xs[3], 170, "")
    arrow(xs[2] + box_w, 122, xs[3], 122, "3")
    arrow(xs[3] + box_w / 2, 105, xs[3] + box_w / 2, 84)
    d.add(Line(xs[3] + box_w, 67, xs[3] + box_w + 6, 67, strokeColor=ACCENT))
    d.add(Line(xs[3] + box_w + 6, 67, xs[3] + box_w + 6, 122, strokeColor=ACCENT))
    arrow(xs[3] + box_w + 6, 122, xs[3] + box_w, 122)
    d.add(String(xs[3] + box_w + 9, 92, "retry", fontName="DV", fontSize=6.2, fillColor=ACCENT))
    arrow(xs[3], 67, xs[2] + box_w, 67, "4")
    arrow(xs[2] + box_w / 2, 50, xs[1] + box_w, 17)
    return d


def run_figures(runs_dir):
    found = []
    if not runs_dir:
        return found
    for bid in ("heat", "logistic", "oscillator"):
        reps = sorted(glob.glob(os.path.join(runs_dir, bid, "*", "report.json")), key=os.path.getmtime)
        if not reps:
            continue
        folder = os.path.dirname(reps[-1])
        with open(reps[-1]) as f:
            rep = json.load(f)
        found.append((bid, folder, rep))
    return found


def img(path, width):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=width, height=width * ih / iw)


# ---------------------------------------------------------------- content
def build(runs_dir):
    rows = load_results()
    story = []

    # Title page
    story += [Spacer(1, 4.2 * cm),
              P("Intelligent Differential Equation Solver Using Physics-Informed Neural Networks", TITLE), Spacer(1, 10),
              P("Technical documentation · design, mathematics, validation, edge-case handling, API and deployment", SUB),
              Spacer(1, 1.2 * cm)]
    meta = [["Version", "1.0.0"], ["Date", date.today().strftime("%d %B %Y")],
            ["Source code", f'<link href="{REPO_URL}" color="#2457a6">{REPO_URL}</link>']]
    if LIVE_URL:
        meta.append(["Live demo", f'<link href="{LIVE_URL}" color="#2457a6">{LIVE_URL}</link>'])
    meta += [["Stack", "Python 3.10 · PyTorch (float64) · NumPy · SciPy · FastAPI · React 18 · Vite · KaTeX · Plotly"],
             ["Quality", "138 automated tests · 9 benchmark problems × 5 configurations × 3 seeds"]]
    story.append(table(meta, [0.2, 0.8], header=False, zebra=False))
    story.append(PageBreak())

    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", parent=BODY, fontName="DV-B", fontSize=10, leading=15, leftIndent=0),
                       ParagraphStyle("t1", parent=BODY, fontSize=8.8, leading=12, leftIndent=16, textColor=MUTED)]
    story += [P("Contents", ParagraphStyle("toch", parent=H1)), toc, PageBreak()]

    # 1 Summary
    story += [P("1. Executive summary", H1),
              P("This project is a working solver for ordinary and partial differential equations based on "
                "<b>physics-informed neural networks (PINNs)</b>. A PINN represents the unknown solution "
                "u(x, t) by a neural network and trains it so that the differential equation and its initial and boundary "
                "conditions hold at many sample points. Derivatives are computed exactly by automatic differentiation."),
              P("Plain PINNs are known to be fragile. Training can stall, converge to a wrong solution, or report a small "
                "loss while the solution is far from correct. This solver is built around that fact. It is "
                "<b>equation-aware</b>: it reads the equation and picks a training recipe from a visible rule table. "
                "It is also <b>validation-driven</b>: no result is shown as correct until it has been checked against "
                "an exact or independent numerical reference, on points never used in training, against thresholds "
                "fixed in advance. When the PINN fails, the solver refines, retries, and finally shows a numerical "
                "solution that is clearly labelled as <i>not</i> the PINN."),
              P("Main results (from <font face='DVM'>benchmark_results/summary.md</font>):", H3)]
    story += bullets([
        "The proposed configuration D met its thresholds in <b>26 of 27</b> benchmark runs, against 7–18 of 27 for "
        "the fixed-recipe baselines A–C. The remaining oscillator run was labelled as a numerical fallback.",
        "On the nonlinear logistic equation every baseline fails at least two of three seeds, because the network drifts "
        "to the wrong equilibrium u ≡ 0. D's causal time-marching rule (R9) fixes this in 3 of 3 runs.",
        "D usually needs a fraction of the time budget. For example, the heat equation takes about 60 s out of 120 s, "
        "the decay ODE 5 s out of 30 s, and the stiff relaxation 6 s out of 30 s.",
        "Inputs are parsed by a purpose-written restricted grammar with no use of <font face='DVM'>eval</font>. "
        "Unsupported or ill-posed problems are rejected before training, with a reason.",
    ])

    # 2 Problem
    story += [P("2. Problem statement and scope", H1),
              P("Engineers and students often need a quick, trustworthy solution to a small differential equation. "
                "Classical solvers are fast and accurate but must be set up for each problem class. PINNs are mesh-free and "
                "general, but their accuracy is hard to judge from the training loss alone. The goal of this project "
                "is a tool that combines the generality of PINNs with the trustworthiness of classical "
                "numerics. It decides <i>how</i> to train, verifies <i>whether</i> training worked, and is honest when it did not."),
              P("2.1 Supported problem classes", H2),
              table([["Class", "Form", "Conditions", "Reference used for validation"],
                     ["ODE initial value problem", "order 1 or 2, explicit in the highest derivative, e.g. u'' + ω²u = 0",
                      "u(a)=…, u'(a)=… at the start", "exact (if given and verified) or SciPy solve_ivp (DOP853; Radau when stiff)"],
                     ["ODE boundary value problem", "order 2, e.g. −u'' = f(x)", "one per end, u or u', at least one on u",
                      "exact or SciPy solve_bvp"],
                     ["1D parabolic PDE", "u_t = F(x, t, u, u_x, u_xx) with positive u_xx coefficient (heat, reaction–diffusion, Burgers)",
                      "u(x,t0)=g(x), Dirichlet u(a,t), u(b,t)", "exact or finite differences (method of lines + BDF)"]],
                    [0.2, 0.32, 0.22, 0.26]),
              Spacer(1, 6),
              P("Anything outside this list is rejected with an explanation. The solver never pretends to solve "
                "it. Not implemented by design: systems of equations, 2D/3D spatial domains, irregular geometries, inverse "
                "problems, Neumann/Robin conditions for PDEs, Fourier features and domain decomposition."),
              P("2.2 Input syntax", H2)]
    story += bullets([
        "Derivatives: <font face='DVM'>u'</font>, <font face='DVM'>u''</font> (ODEs) or <font face='DVM'>u_t, u_x, u_xx</font> (PDEs). "
        "Also accepted and rewritten: <font face='DVM'>du/dt</font>, <font face='DVM'>d^2u/dx^2</font>, ∂u/∂t, "
        "<font face='DVM'>u'(t)</font>, <font face='DVM'>u(x, t)</font>, typographic characters (′ − · × π ²).",
        "Operators <font face='DVM'>+ - * / ^ **</font> and parentheses. Functions: sin, cos, tan, exp, log/ln, sqrt, sinh, cosh, tanh, abs. "
        "Constants: pi, e.",
        "Implicit multiplication: <font face='DVM'>2u</font>, <font face='DVM'>(1 - u)u</font>, <font face='DVM'>k u</font>. "
        "Parameters are named numbers, e.g. k = 0.1. The unknown may be renamed (e.g. y).",
        "An equation without '=' is read as '… = 0', and a warning says so.",
    ])
    story.append(P("Example problem file (CLI and API):", H3))
    story.append(code('''{"name": "Heat", "equation": "u_t = k*u_xx", "variables": ["x", "t"],
 "domain": {"x": [0, 1], "t": [0, 0.5]},
 "conditions": ["u(x, 0) = sin(pi*x)", "u(0, t) = 0", "u(1, t) = 0"],
 "parameters": {"k": 0.1}, "exact": "exp(-k*pi^2*t)*sin(pi*x)"}'''))

    # 3 Architecture
    story += [Spacer(1, 4), P("3. System architecture", H1),
              P("The numerical core (package <font face='DVM'>pinnsolver</font>) has no dependency on the web layer. "
                "The FastAPI app and the CLI are thin shells around <font face='DVM'>Solver.solve</font>. The numbers on "
                "the arrows give the order of the steps in one solve."),
              architecture_diagram(), P("Figure 1. Components and data flow of one solve.", CAPTION),
              table([["Module", "Responsibility", "Code"],
                     ["M1 Problem input", "Problem as typed by the user (equation, variables, domain, conditions, parameters, optional exact solution)", "spec.py ProblemInput"],
                     ["M2 Parser & validator", "Restricted grammar, own recursive-descent parser, classification, well-posedness checks", "expr.py, parser.py"],
                     ["M3 Representation", "Typed equation and conditions, features (stiffness, oscillation, Péclet number)", "spec.py EquationSpec, Condition"],
                     ["M4 Strategy selector", "Applies rules R1–R9 and produces a TrainingConfig, with the reasons for each choice", "strategy.py"],
                     ["M5 PINN training", "tanh MLP in float64, hard-constraint trial solutions, Adam→L-BFGS, time-marching, gPINN", "pinn.py, training.py"],
                     ["M6 Adaptive collocation", "RAR: add the highest-residual points from fresh Sobol candidates", "training.py AdaptiveSampler"],
                     ["M7 Validation", "Fresh-point residuals, condition errors, relative L2 against the reference, thresholds", "validation.py"],
                     ["M8 Reference solvers", "Analytical, solve_ivp, solve_bvp, finite differences, each with its own error estimate", "references.py"],
                     ["M9 Report", "report.json/.md, plots, training log, provenance", "report.py"],
                     ["Control loop", "Attempts, refinement, retries, fallback within the time budget", "solver.py"],
                     ["Interfaces", "REST API + job queue, static UI hosting; CLI; React UI", "api.py, __main__.py, frontend/"]],
                    [0.22, 0.53, 0.25])]

    # 4 Method
    story += [PageBreak(), P("4. Mathematical method", H1),
              P("4.1 Network and derivatives", H2),
              P("The solution is approximated by a fully connected tanh network N<sub>θ</sub> in double precision. Inputs are "
                "scaled to [−1, 1] inside the network. Derivatives are taken by autograd with respect to the "
                "<i>physical</i> coordinates, so the chain-rule factors from the scaling are exact. Default sizes: 3×32 for "
                "ODEs, 4×32 for PDEs, and 4×64 for oscillatory problems."),
              P("4.2 Loss function", H2),
              P("With residual r = (left side − right side) of the equation as entered, the loss is"),
              P("<b>L(θ) = λ<sub>r</sub>·mean(r²) + λ<sub>b</sub>·L<sub>bc</sub> + λ<sub>i</sub>·L<sub>ic</sub> "
                "[+ λ<sub>g</sub>·mean|∇r|²]</b>", ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER)),
              P("The condition terms appear only for conditions that are <i>not</i> built into the trial solution. The "
                "gradient-enhanced term (gPINN) is used only on a retry for steep-gradient problems. It is kept only if "
                "validation improves."),
              P("4.3 Hard constraints (Lagaris et al.)", H2),
              P("These are used only where the construction is mathematically exact:")]
    story += bullets([
        "IVP: u = u<sub>0</sub> + v<sub>0</sub>(t − t<sub>0</sub>) + ((t − t<sub>0</sub>)/T)<super>n</super> N, with n the order.",
        "Dirichlet BVP: linear interpolant of the end values + s(1 − s) N, where s is the normalised coordinate.",
        "Parabolic PDE: transfinite interpolation A(x, t) + τ s(1 − s) N. This is used only when the initial and boundary data agree at "
        "the corners. Otherwise penalty terms are used and the report says so.",
    ])
    story += [P("4.4 Solution scaling", H2),
              P("For a linear equation whose data are clearly not O(1) (outside [0.1, 10]), the network output is "
                "multiplied by the data size U. Residuals and condition errors in the loss are divided by U, so the problem "
                "trains exactly like its O(1) counterpart. Absolute thresholds are scaled by U as well. All nine benchmarks "
                "have O(1) data and are unaffected; a test checks this."),
              P("4.5 Optimisation", H2)]
    story += bullets([
        "Adam with exponential learning-rate decay, then L-BFGS with a strong-Wolfe line search.",
        "A stall detector ends Adam when the best loss improves by less than 1% over 1000 iterations.",
        "NaN/Inf guards restore the last finite checkpoint. Each attempt has a time cap, and Adam gets at most 60% of it when L-BFGS follows.",
    ])
    story += [P("4.6 Adaptive collocation (RAR)", H2),
              P("|r| is evaluated on fresh Sobol candidates over the whole domain. The top-k points are <i>added</i> to the "
                "retained set, so coverage elsewhere is kept. Growth stops at the point budget (at most 20 000)."),
              P("4.7 Causal time-marching (R9)", H2),
              P("For time-dependent problems that still fail after refinement, the solver starts again from a fresh "
                "initialisation and trains over four growing time windows [t<sub>0</sub>, t<sub>0</sub> + kT/4]. The "
                "network then learns the early-time behaviour before the late-time behaviour. This rescues the logistic "
                "benchmark, where a plain PINN drifts to the trivial equilibrium u ≡ 0.")]

    # 5 Strategy
    story += [PageBreak(), P("5. Strategy selection: rule table", H1),
              P("The selector computes features of the parsed equation, such as order, linearity, a stiffness index, the "
                "number of oscillation cycles, and the Péclet number. It then applies the rules below. Each applied rule is "
                "written into the report together with the reason it fired."),
              table([["Rule", "Feature", "Strategy", "Caution"]] +
                    [[r["id"], r["feature"], r["strategy"], r["caution"]] for r in RULE_TABLE], [0.07, 0.23, 0.42, 0.28]),
              P("5.1 Configurations A–E (ablation study)", H2),
              table([["Mode", "Name", "Recipe"],
                     ["A", MODE_NAMES["A"], "Plain PINN, all conditions as penalties, Adam only, fixed uniform points, one attempt"],
                     ["B", MODE_NAMES["B"], "A + hard constraints where exact"],
                     ["C", MODE_NAMES["C"], "B + adaptive collocation"],
                     ["D", MODE_NAMES["D"], "Proposed solver: rule-based strategy, Adam→L-BFGS, validation, refine/retry (R8/R9), labelled fallback"],
                     ["E", MODE_NAMES["E"], "Numerical solver only (no PINN); the classical baseline"]], [0.08, 0.32, 0.6]),
              P("Ablations of D can be requested through <font face='DVM'>ablate=[…]</font> to switch off individual "
                "components and measure what each contributes.", SMALL)]

    # 6 Validation & control loop
    story += [P("6. Validation and the control loop", H1),
              P("6.1 Independent validation", H2)]
    story += bullets([
        "Residual RMS and maximum are computed on fresh points from a separate random stream that training never uses.",
        "IC/BC errors are reported per condition.",
        "Relative L2 error (ε = 1e-12) and maximum error are computed on a uniform grid: 1001 points in 1D, 101×101 in 2D.",
        "If the reference is identically zero, the absolute RMS error is used instead of the relative error.",
        "If no reference exists, the result is labelled “residual-only; true error not established”.",
    ])
    th = DEFAULT_THRESHOLDS
    story.append(table([["Problem type", "Relative L2", "IC/BC error", "Residual RMS", "Default budget"]] +
                       [[k.replace("_", " "), f"≤ {v['rel_l2']:g}", f"≤ {v['condition_max']:g}", f"≤ {v['residual_rms']:g}",
                         f"{DEFAULT_TIME_BUDGET[k]:g} s"] for k, v in th.items()], [0.28, 0.18, 0.18, 0.18, 0.18]))
    story += [Spacer(1, 6), P("6.2 Reference error", H2),
              P("A reference is only as good as its own accuracy, so each one estimates its own error. SciPy references are "
                "re-solved with tolerances 100× looser. The finite-difference reference uses grid halving (Richardson), "
                "with the cell Péclet number kept ≤ 0.25. A reference whose estimated error exceeds 1% of the solution "
                "size is flagged. The Burgers reference agrees with an independent Cole–Hopf quadrature solution to a "
                "relative L2 error of about 1.5e-4."),
              P("6.3 Mode D control loop (R8/R9)", H2)]
    story += bullets([
        "<b>Attempt 1</b> may use at most 60% of the budget.",
        "<b>Refine.</b> On failure, high-residual points are added and training continues. This repeats only while each "
        "refinement cuts the error by at least 20%.",
        "<b>Time-marching (R9)</b> for IVPs and parabolic PDEs.",
        "<b>gPINN (R7)</b> for steep-gradient problems if affordable. It is kept only if validation improves.",
        "Otherwise a <b>wider network</b> is tried. There are at most 4 attempts, and the best one is kept.",
        "If the thresholds are still not met, the numerical solution is shown, labelled "
        "<b>“Numerical fallback solver (NOT the PINN)”</b>, together with the PINN's own error.",
    ])

    # 7 Benchmarks
    story += [PageBreak(), P("7. Benchmark results", H1),
              P("Nine problems were solved with configurations A–D and three seeds each. All configurations of a problem "
                "get the same wall-clock budget and point budget. Thresholds were fixed before the runs. The figures "
                "below are computed directly from <font face='DVM'>benchmark_results/results.csv</font>. They are the "
                "only performance numbers this project reports."),
              img(benchmark_chart(rows), CONTENT_W),
              P("Figure 2. Median relative L2 error of the PINN (log scale; lower is better). Dashed line: ODE/BVP "
                "threshold. The PDE threshold is 1e-2.", CAPTION),
              P("Runs in which the PINN met every threshold (out of 3 seeds):", H3),
              table(pass_table(rows), [0.18, 0.09, 0.09, 0.09, 0.09, 0.24, 0.22]),
              Spacer(1, 6),
              P("Configuration E (numerical solver only) reaches errors of 1e-10 to 1e-13 in well under a second on these "
                "small problems. The PINN is <i>not</i> claimed to be faster than classical solvers in 1D. Its value is "
                "being mesh-free and general, and the contribution here is making it <i>reliable and honest</i>.",
                SMALL)]

    figs = run_figures(runs_dir)
    if figs:
        story += [PageBreak(), P("7.1 Example output", H2),
                  P("These plots come from fresh runs of the proposed solver (mode D) on three benchmarks. They are the "
                    "files the solver writes to <font face='DVM'>runs/&lt;id&gt;/</font> and also shows in the web UI.")]
        for bid, folder, rep in figs:
            v = rep.get("validation", {}).get("metrics", {})
            cap = (f"Figure: <b>{bid}</b>. Status: {rep.get('status')}; final method: {rep.get('final_method')}; "
                   f"rel. L2 = {v.get('rel_l2', float('nan')):.2e}; training {rep.get('timings', {}).get('training_s', 0):.1f} s; "
                   f"attempts: {len(rep.get('attempts', []))}.")
            from reportlab.lib.utils import ImageReader
            sol, loss = os.path.join(folder, "solution.png"), os.path.join(folder, "loss.png")
            block = []
            if os.path.isfile(sol):
                iw, ih = ImageReader(sol).getSize()
                if iw / ih > 2.2:  # wide multi-panel PDE figure: full width, loss plot below
                    block.append(img(sol, CONTENT_W))
                    if os.path.isfile(loss):
                        block.append(img(loss, CONTENT_W * 0.55))
                    t = None
                else:
                    parts = [img(sol, CONTENT_W * 0.58)] + ([img(loss, CONTENT_W * 0.4)] if os.path.isfile(loss) else [])
                    t = Table([parts], colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4][:len(parts)])
                    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
                    block.append(t)
            if block:
                story.append(KeepTogether(block + [P(cap, CAPTION)]))

    # 8 Edge cases
    story += [PageBreak(), P("8. Edge-case handling", H1),
              P("Every input is treated as untrusted. The table below lists the edge cases handled, grouped by layer. Each "
                "one is covered by an automated test (files under <font face='DVM'>tests/</font>). Items marked "
                "<b>[fixed in final review]</b> were found by probing the live API during the final review and now have "
                "regression tests in <font face='DVM'>tests/test_api_hardening.py</font>."),
              table([["Category", "Input / situation", "Behaviour"],
                     ["Parsing safety", "eval-style payloads, attribute access, unknown functions, emoji or control characters",
                      "Rejected by the own grammar. eval/parse_expr/lambdify are never called on user text"],
                     ["Size limits", "> 400 characters, > 300 tree nodes, nesting deeper than the limit, expansion > 5000 terms",
                      "422 with a reason, returned in milliseconds"],
                     ["Constants", "sqrt(−1), exp(1000), 9^9^9, or non-finite only after a parameter is substituted", "Rejected: not a finite real number"],
                     ["Domain", "NaN/∞ ends, lower ≥ upper, |x| > 10⁴, width too small to resolve, variables not matching the domain",
                      "Rejected with the offending interval"],
                     ["Names", "Parameter named like a variable, function, u or pi; duplicate or multi-letter variables; t as unknown",
                      "Rejected, with a suggestion where possible (e.g. ku → k*u)"],
                     ["Order / form", "Third-order derivative, nonlinear in the highest derivative, PDE without t, non-parabolic PDE",
                      "Rejected: outside the supported classes"],
                     ["Conditions", "Empty, duplicate or <b>conflicting</b> (u(0)=1 and u(0)=2), interior point, terminal-value IVP, too few or too many",
                      "Rejected; conflicting conditions are named as such <b>[fixed in final review]</b>"],
                     ["Well-posedness", "Equation undefined in the domain (1/(t−0.5)), blow-up (u' = u², u(0)=1 on [0,2]), "
                      "non-unique BVP (u'' = −u, u(0)=u(π)=0), IC where the equation is undefined",
                      "Rejected before training. A nearly non-unique BVP is accepted with a warning"],
                     ["Exact solution", "Wrong formula, NaN at an end point, does not satisfy a condition", "Ignored with a warning, never trusted"],
                     ["Scaling", "Very small or large data (u(0) = 0.001)", "Output scaling; trains like the O(1) problem"],
                     ["Zero solution", "Reference identically 0", "Absolute RMS error replaces relative error"],
                     ["Reference failure", "FD blow-up or front too steep for the grid", "Solved anyway, labelled residual-only"],
                     ["Training", "NaN/Inf loss, tiny budget, stall", "Checkpoint restore; tiny budget gives a labelled fallback, never a crash"],
                     ["Request numbers", "NaN/Infinity literals in JSON (accepted by the parser) in time_budget or thresholds",
                      "422, never 500 <b>[fixed in final review]</b>"],
                     ["Thresholds", "Infinite, NaN, zero, negative or unknown keys (∞ would make every run “pass”)",
                      "422 with the allowed keys and range <b>[fixed in final review]</b>"],
                     ["Server resources", "Many submissions; long-lived deployment", "429 beyond 1 running + 8 queued; last 50 jobs in "
                      "memory; old run folders pruned to 200 <b>[fixed in final review]</b>"],
                     ["Files", "Path traversal in /api/runs/{id}/{file}", "Whitelisted names, realpath check, 404"],
                     ["Cancellation", "Cancel queued, running, finished or unknown job", "Queued/running stop; finished unchanged; unknown is a 404"],
                     ["Frontend", "Backend unreachable or waking up", "Clear message instead of a raw network error"]],
                    [0.17, 0.43, 0.40])]

    # 9 Web & API
    story += [PageBreak(), P("9. Web application and REST API", H1),
              P("The React UI checks the problem as the user types (<font face='DVM'>POST /api/validate</font>, debounced). "
                "It shows the equation and conditions typeset with KaTeX, or the reason they cannot be read yet. Solve is "
                "enabled once the problem is accepted. During training the UI polls the job, plots the live loss, and "
                "lists the stages. The finished report shows the status banner, the strategy and its reasons, every "
                "attempt, the error metrics, and interactive Plotly charts: solution vs. reference, pointwise error, and "
                "for PDEs a heat map with time slices. Plotly (about 4.5 MB) is lazy-loaded, so the form appears after "
                "about 440 KB of JavaScript. Benchmarks can be loaded as presets. Every job has a shareable "
                "<font face='DVM'>#job=&lt;id&gt;</font> URL."),
              table([["Method", "Path", "Purpose"],
                     ["GET", "/api/health", "Liveness and number of jobs in progress"],
                     ["GET", "/api/info", "Modes, rule table, thresholds, limits, grammar"],
                     ["GET", "/api/benchmarks", "The nine built-in problems"],
                     ["POST", "/api/validate", "Parse and classify without training (422 + reason if rejected)"],
                     ["POST", "/api/jobs", "Start a solve → 202 with job id; 422 invalid; 429 queue full"],
                     ["GET", "/api/jobs", "Recent jobs"],
                     ["GET", "/api/jobs/{id}", "State, stage, events, live loss trace"],
                     ["GET", "/api/jobs/{id}/report", "Full report (409 until done)"],
                     ["POST", "/api/jobs/{id}/cancel", "Cancel a job"],
                     ["GET", "/api/runs/{run_id}/{file}", "report.json/.md, problem.json, reference.json, training_log.csv, solution.png, loss.png"]],
                    [0.1, 0.32, 0.58]),
              Spacer(1, 6), P("Example:", H3),
              code('''curl -X POST $HOST/api/jobs -H 'Content-Type: application/json' -d '{
  "problem": {"equation": "u'' + 4u = 0", "variables": ["t"], "domain": {"t": [0, 3]},
              "conditions": ["u(0) = 1", "u'(0) = 0"]},
  "mode": "D", "seed": 0, "time_budget": 60}'
# -> {"job_id": "3f2a9c1b7d4e", "state": "queued", ...}
curl $HOST/api/jobs/3f2a9c1b7d4e          # poll until state == "done"
curl $HOST/api/jobs/3f2a9c1b7d4e/report   # status, validation, attempts, plots'''),
              P("Interactive OpenAPI documentation is served at <font face='DVM'>/docs</font> on every running server.", SMALL),
              P("9.1 Command line", H2),
              code('''python -m pinnsolver --benchmark heat              # built-in benchmark
python -m pinnsolver my_problem.json --mode D       # your own problem
python scripts/run_benchmarks.py --seeds 0 1 2      # configurations A-E, all problems''')]

    # 10 Deployment
    story += [P("10. Installation and deployment", H1),
              P("10.1 Local", H2),
              code('''git clone https://github.com/AkashKishanKarri/pinn-solver && cd pinn-solver
./run.sh            # creates .venv, builds the UI, serves http://127.0.0.1:8000
# or manually:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd frontend && npm ci && npm run build)
.venv/bin/uvicorn pinnsolver.api:app --port 8000'''),
              P("10.2 Docker", H2),
              P("The multi-stage Dockerfile builds the UI with Node 20. It then installs the CPU build of the pinned "
                "PyTorch version (about 200 MB instead of about 2.5 GB with CUDA) on python:3.10-slim. It runs as a "
                "non-root user (uid 1000) and serves on <font face='DVM'>$PORT</font> (default 7860). A health check "
                "calls <font face='DVM'>/api/health</font>."),
              code('''docker build -t pinn-solver . && docker run -p 7860:7860 pinn-solver'''),
              P("10.3 Hosting", H2),
              P("The live demo runs on <b>Render</b> (free plan) from the Blueprint in <font face='DVM'>render.yaml</font>, "
                "which redeploys on every push to main. The free plan has 512 MB RAM and 0.1 CPU. The blueprint "
                "therefore caps collocation points at 6000 (the measured peak memory of a PDE solve is about 400 MB) "
                "and uses one thread. Training there is roughly 10× slower than on a laptop, so within the default "
                "budgets more problems end in the labelled numerical fallback. The benchmark numbers in Section 7 "
                "were measured on a laptop. Docker Spaces on Hugging Face need a PRO subscription; "
                "<font face='DVM'>scripts/deploy_hf_space.sh</font> deploys there. Railway and Fly.io work from the same "
                "Dockerfile. Serverless platforms (Vercel, AWS Lambda) are <i>not</i> suitable: a solve trains for up to "
                "15 minutes in a background thread and job state is kept in memory between requests."),
              table([["Environment variable", "Default", "Meaning"],
                     ["PORT", "7860 (Docker)", "HTTP port"],
                     ["PINN_RUNS_DIR", "runs/ (/tmp/pinn-runs in Docker)", "Where run artefacts are written"],
                     ["PINN_MAX_CONCURRENT", "1", "Training jobs in parallel"],
                     ["PINN_MAX_QUEUED", "8", "Jobs allowed to wait"],
                     ["PINN_MAX_RUNS_KEPT", "200", "Run folders kept on disk"],
                     ["PINN_MAX_POINTS", "20000", "Collocation-point cap (512–20000) for small-memory hosts"]], [0.3, 0.3, 0.4])]

    # 11 Testing
    story += [PageBreak(), P("11. Testing and quality assurance", H1),
              table([["Test file", "What it covers"],
                     ["test_expr_parser.py", "Grammar, rejection of unsafe input, NumPy/Torch evaluation agreement, classification, rejection reasons, exact-solution verification, corner compatibility"],
                     ["test_friendly_input.py", "Alternative notations (Leibniz, ∂, Unicode), missing '=', hints, undefined equations, blow-up, non-unique BVPs, scaling, zero reference"],
                     ["test_edge_cases.py", "Huge constants and expansions, complex constants, abs(u), NaN exact solutions, non-finite references, failed PDE references, malformed dicts, tiny budgets"],
                     ["test_numerics.py", "All benchmark exact solutions verify; numerical references match exact; Burgers vs. Cole–Hopf; hard constraints exact for random weights; autograd residual vanishes on the exact solution; validator independence"],
                     ["test_solver_api.py", "Strategy rules and modes, end-to-end solves, seed reproducibility, time-marching rescue, labelled fallback, API end-to-end, cancellation"],
                     ["test_api_hardening.py", "Regression tests from the final review: NaN/∞ request numbers, threshold validation, conflicting conditions"]],
                    [0.25, 0.75]),
              Spacer(1, 6),
              code('''.venv/bin/python -m pytest -q        # 138 passed'''),
              P("In addition, the production server was tested end to end. A real job (u' = −u on [0, 1]) was submitted "
                "over HTTP, polled to completion, and passed with relative L2 = 1.2e-5 in 4 s. Its plots and report "
                "files were served correctly.", BODY)]

    # 12 Limitations
    story += [P("12. Limitations and future work", H1)]
    story += bullets([
        "A small residual is not a proof of correctness. This is why a reference comparison is required for a “passed” verdict.",
        "Numerical references carry their own, estimated, error.",
        "Results from a single seed vary, and wall-clock budgets make results depend somewhat on the hardware. Use the benchmark runner for statistics.",
        "In 1D, classical solvers are faster and more accurate. The PINN's advantages (mesh-free, differentiable, extensible to inverse problems) show in settings outside the current scope.",
        "Not for safety-critical use without domain-specific validation and expert review.",
        "Future work: systems of ODEs, 2D domains, Neumann/Robin PDE conditions, Fourier features for high-frequency "
        "solutions (R4), inverse problems (parameter identification), and persistent job storage for multi-instance hosting.",
    ])
    story += [P("References", H2)]
    story += bullets([
        "M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. J. Comput. Phys. 378 (2019) 686–707.",
        "I. E. Lagaris, A. Likas, D. I. Fotiadis. Artificial neural networks for solving ordinary and partial differential equations. IEEE TNN 9 (1998).",
        "L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. DeepXDE: a deep learning library for solving differential equations. SIAM Review 63 (2021). (Residual-based adaptive refinement.)",
        "J. Yu, L. Lu, X. Meng, G. E. Karniadakis. Gradient-enhanced PINNs (gPINN). CMAME 393 (2022).",
        "S. Wang, S. Sankaran, P. Perdikaris. Respecting causality for training PINNs. CMAME 421 (2024).",
        "P. Virtanen et al. SciPy 1.0. Nature Methods 17 (2020).",
    ], ParagraphStyle("ref", parent=BUL, fontSize=8.4, leading=11.5))

    doc = Doc(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=2.2 * cm,
              title="Intelligent Differential Equation Solver Using PINNs — Technical Documentation",
              author="Akash Kishan Karri", subject="PINN solver documentation")
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="directory with benchmark run folders to embed as example output")
    build(ap.parse_args().runs)
