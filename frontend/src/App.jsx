import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import Plot from './Plot.jsx'
import katex from 'katex'
import 'katex/dist/katex.min.css'

// Typeset mathematics (LaTeX produced by the backend from the parsed problem).
function Tex({ tex, block = false }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { displayMode: block, throwOnError: false })
    } catch {
      return null
    }
  }, [tex, block])
  if (html === null) return <span className="mono">{tex}</span>
  return <span className={block ? 'tex block' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />
}

const EMPTY_FORM = {
  name: '',
  equation: '',
  variables: 't',
  dependent: 'u',
  domain: { t: ['0', '1'] },
  conditions: '',
  parameters: '',
  exact: '',
}

const STATUS_STYLE = {
  passed: { cls: 'ok', label: 'Passed' },
  passed_residual_only: { cls: 'warn', label: 'Passed (residual only)' },
  fallback: { cls: 'warn', label: 'PINN failed: numerical fallback shown' },
  failed: { cls: 'bad', label: 'Failed' },
  numerical_unvalidated: { cls: 'info', label: 'Numerical (not independently validated)' },
}

const STAGE_TEXT = {
  queued: 'Waiting in queue',
  parse: 'Parsing and validating the problem',
  strategy: 'Selecting a strategy',
  reference: 'Computing the reference solution',
  attempt: 'Starting a training attempt',
  training: 'Training the PINN',
  validated: 'Validating on fresh points',
  fallback: 'Running the numerical fallback',
  numerical: 'Running the numerical solver',
  report: 'Writing the report',
  done: 'Finished',
  cancelled: 'Cancelled',
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined) return 'n/a'
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v)
  const a = Math.abs(v)
  if (a >= 1e-3 && a < 1e4) {
    const t = v.toPrecision(digits)
    return t.includes('.') && !t.includes('e') ? t.replace(/0+$/, '').replace(/\.$/, '') : t
  }
  return v.toExponential(digits - 1)
}

function parseVariables(text) {
  return text
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

// Domain end points may be written with pi and simple arithmetic, e.g. "pi", "2*pi", "-pi/2", "1e-3".
// A small recursive-descent evaluator: the text is never executed as code.
function parseNumber(text) {
  const src = String(text).replace(/\s+/g, '').replace(/π/g, 'pi').replace(/[−–]/g, '-').replace(/[·×]/g, '*')
  if (!src) return NaN
  let i = 0
  const atom = () => {
    if (src[i] === '(') {
      i += 1
      const v = sum()
      if (src[i] !== ')') throw new Error('missing )')
      i += 1
      return v
    }
    if (src.startsWith('pi', i)) { i += 2; return Math.PI }
    const m = /^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?/i.exec(src.slice(i))
    if (!m) throw new Error('not a number')
    i += m[0].length
    return Number(m[0])
  }
  const unary = () => {
    if (src[i] === '-') { i += 1; return -unary() }
    if (src[i] === '+') { i += 1; return unary() }
    return atom()
  }
  const product = () => {
    let v = unary()
    for (;;) {
      if (src[i] === '*') { i += 1; v *= unary() }
      else if (src[i] === '/') { i += 1; v /= unary() }
      else if (src.startsWith('pi', i) || src[i] === '(') v *= atom() // "2pi", "2(…)"
      else return v
    }
  }
  const sum = () => {
    let v = product()
    for (;;) {
      if (src[i] === '+') { i += 1; v += product() }
      else if (src[i] === '-') { i += 1; v -= product() }
      else return v
    }
  }
  try {
    const v = sum()
    return i === src.length ? v : NaN
  } catch {
    return NaN
  }
}

function formToProblem(form) {
  const variables = parseVariables(form.variables)
  const domain = {}
  for (const v of variables) {
    const [lo, hi] = form.domain[v] || ['', '']
    const a = parseNumber(lo)
    const b = parseNumber(hi)
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw new Error(`Domain of ${v}: enter two numbers (pi is allowed, e.g. 2*pi).`)
    domain[v] = [a, b]
  }
  const parameters = {}
  for (const raw of form.parameters.split('\n')) {
    const line = raw.trim()
    if (!line) continue
    const m = line.match(/^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\S+)$/)
    if (!m || !Number.isFinite(Number(m[2]))) throw new Error(`Parameter line "${line}" must look like  name = number`)
    parameters[m[1]] = Number(m[2])
  }
  return {
    name: form.name || null,
    equation: form.equation,
    variables,
    dependent: (form.dependent || '').trim() || 'u',
    domain,
    conditions: form.conditions.split('\n').map((s) => s.trim()).filter(Boolean),
    parameters,
    exact: form.exact.trim() || null,
  }
}

function problemToForm(p) {
  return {
    name: p.name || '',
    equation: p.equation,
    variables: p.variables.join(', '),
    dependent: p.dependent || 'u',
    domain: Object.fromEntries(Object.entries(p.domain).map(([k, v]) => [k, v.map((x) => String(Number(x.toPrecision(12))))])),
    conditions: p.conditions.join('\n'),
    parameters: Object.entries(p.parameters || {}).map(([k, v]) => `${k} = ${Number(v.toPrecision(15))}`).join('\n'),
    exact: p.exact || '',
  }
}

// ---------------------------------------------------------------------------
export default function App() {
  const [info, setInfo] = useState(null)
  const [benchmarks, setBenchmarks] = useState([])
  const [preset, setPreset] = useState('')
  const [form, setForm] = useState(EMPTY_FORM)
  const [mode, setMode] = useState('D')
  const [seed, setSeed] = useState('0')
  const [budget, setBudget] = useState('')
  const [thrL2, setThrL2] = useState('')
  const [thrCond, setThrCond] = useState('')
  const [ablate, setAblate] = useState([])
  const [parsed, setParsed] = useState(null)
  const [liveError, setLiveError] = useState('')
  const [checking, setChecking] = useState(false)
  const checkSeq = useRef(0)
  const [formError, setFormError] = useState('')
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState(null)
  const [report, setReport] = useState(null)
  const [jobs, setJobs] = useState([])
  const pollRef = useRef(null)

  useEffect(() => {
    api.info().then(setInfo).catch((e) => setFormError(`Cannot reach the solver API: ${e.message}`))
    api.benchmarks().then((b) => {
      setBenchmarks(b)
      if (b.length && !/^#job=/.test(window.location.hash)) choosePreset(b[0], true)
    }).catch(() => {})
    api.jobs().then(setJobs).catch(() => {})
    const m = window.location.hash.match(/^#job=([0-9a-f]+)$/)
    if (m) openJob(m[1])
    return () => clearInterval(pollRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const choosePreset = (b, initial = false) => {
    setPreset(b.id)
    setForm(problemToForm(b.problem))
    setBudget(String(b.budget_s))
    setFormError('')
    if (!initial) setReport(null)
  }

  const variables = useMemo(() => parseVariables(form.variables), [form.variables])

  const update = (patch) => {
    setForm((f) => ({ ...f, ...patch }))
    setPreset('')
  }

  const setDomain = (v, i, value) => {
    setForm((f) => {
      const d = { ...f.domain, [v]: [...(f.domain[v] || ['', ''])] }
      d[v][i] = value
      return { ...f, domain: d }
    })
    setPreset('')
  }

  // Live check: a short pause after the last keystroke, the problem is validated by the backend and
  // shown typeset, or the reason it cannot be read yet is shown. Out-of-date answers are ignored.
  useEffect(() => {
    const id = ++checkSeq.current
    if (!form.equation.trim()) {
      setParsed(null)
      setLiveError('')
      setChecking(false)
      return undefined
    }
    setChecking(true)
    const timer = setTimeout(async () => {
      let spec = null
      let error = ''
      try {
        const p = formToProblem(form)
        if (!p.conditions.length) throw new Error(`Add at least one initial or boundary condition, e.g. ${p.dependent}(0) = 1.`)
        spec = await api.validate(p)
      } catch (e) {
        error = e.message
      }
      if (checkSeq.current !== id) return
      setParsed(spec)
      setLiveError(error)
      setChecking(false)
    }, 450)
    return () => clearTimeout(timer)
  }, [form])

  const poll = useCallback((id) => {
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.job(id)
        setJob(j)
        if (['done', 'error', 'cancelled'].includes(j.state)) {
          clearInterval(pollRef.current)
          if (j.report_available) setReport(await api.report(id))
          api.jobs().then(setJobs).catch(() => {})
        }
      } catch (e) {
        clearInterval(pollRef.current)
        setFormError(e.message)
      }
    }, 1000)
  }, [])

  const solve = async () => {
    setFormError('')
    try {
      const problem = formToProblem(form)
      const thresholds = {}
      if (thrL2) thresholds.rel_l2 = Number(thrL2)
      if (thrCond) thresholds.condition_max = Number(thrCond)
      const body = {
        problem,
        mode,
        seed: Number(seed) || 0,
        time_budget: budget ? Number(budget) : null,
        thresholds: Object.keys(thresholds).length ? thresholds : null,
        ablate: mode === 'D' ? ablate : [],
      }
      setBusy(true)
      const j = await api.submit(body)
      setJob(j)
      setReport(null)
      window.history.replaceState(null, '', `#job=${j.job_id}`)
      poll(j.job_id)
      api.jobs().then(setJobs).catch(() => {})
    } catch (e) {
      setFormError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const openJob = async (id) => {
    try {
      const j = await api.job(id)
      setJob(j)
      const rep = j.report_available ? await api.report(id) : null
      setReport(rep)
      if (rep) {
        // show the problem and settings that produced this report
        setForm(problemToForm(rep.problem))
        setPreset('')
        setMode(rep.mode)
        setSeed(String(rep.reproducibility.seed))
        setBudget(rep.budget.time_budget_s != null ? String(rep.budget.time_budget_s) : '')
      }
      window.history.replaceState(null, '', `#job=${id}`)
      if (['queued', 'running'].includes(j.state)) poll(id)
    } catch (e) {
      setFormError(e.message)
    }
  }

  const running = job && ['queued', 'running'].includes(job.state)

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Intelligent Differential Equation Solver</h1>
          <p className="sub">Physics-informed neural networks with rule-based strategy selection, independent validation and a transparent numerical fallback.</p>
        </div>
      </header>

      <main className="layout">
        <section className="panel form">
          <h2>Problem</h2>
          <label className="field">
            <span>Example problem</span>
            <select value={preset} onChange={(e) => { const b = benchmarks.find((x) => x.id === e.target.value); if (b) choosePreset(b) }}>
              <option value="">Custom problem</option>
              {benchmarks.map((b) => (
                <option key={b.id} value={b.id}>{b.title}: {b.class}</option>
              ))}
            </select>
          </label>
          {preset && <p className="hint">{benchmarks.find((b) => b.id === preset)?.description}</p>}

          <label className="field">
            <span>Equation</span>
            <input className="mono" value={form.equation} onChange={(e) => update({ equation: e.target.value })} placeholder="u' = -2*u     or     du/dt = -2u" spellCheck={false} />
          </label>
          <div className="row">
            <label className="field narrow">
              <span>Unknown function</span>
              <input className="mono" value={form.dependent} maxLength={1} onChange={(e) => update({ dependent: e.target.value })} placeholder="u" spellCheck={false} />
            </label>
            <label className="field">
              <span>Independent variable(s)</span>
              <input className="mono" value={form.variables} onChange={(e) => update({ variables: e.target.value })} placeholder="t   or   x, t" spellCheck={false} />
            </label>
          </div>
          {variables.map((v) => (
            <div className="row" key={v}>
              <label className="field">
                <span>{v} from</span>
                <input className="mono" value={form.domain[v]?.[0] ?? ''} onChange={(e) => setDomain(v, 0, e.target.value)} placeholder="0" spellCheck={false} />
              </label>
              <label className="field">
                <span>{v} to</span>
                <input className="mono" value={form.domain[v]?.[1] ?? ''} onChange={(e) => setDomain(v, 1, e.target.value)} placeholder="1  or  2*pi" spellCheck={false} />
              </label>
            </div>
          ))}
          <label className="field">
            <span>Initial / boundary conditions (one per line)</span>
            <textarea className="mono" rows={3} value={form.conditions} onChange={(e) => update({ conditions: e.target.value })} placeholder={"u(0) = 1\nu'(0) = 0"} spellCheck={false} />
          </label>
          <ProblemPreview spec={parsed} error={liveError} checking={checking} empty={!form.equation.trim()} />
          <div className="row">
            <label className="field">
              <span>Parameters (name = value)</span>
              <textarea className="mono" rows={2} value={form.parameters} onChange={(e) => update({ parameters: e.target.value })} placeholder="k = 0.1" spellCheck={false} />
            </label>
            <label className="field">
              <span>Exact solution (optional)</span>
              <textarea className="mono" rows={2} value={form.exact} onChange={(e) => update({ exact: e.target.value })} placeholder="exp(-2*t)" spellCheck={false} />
            </label>
          </div>
          <label className="field">
            <span>Name (optional)</span>
            <input value={form.name} onChange={(e) => update({ name: e.target.value })} placeholder="Shown in the report and recent jobs" />
          </label>
          <details className="syntax">
            <summary>Input syntax and supported problems</summary>
            <ul>
              <li>Derivatives: <code>u'</code>, <code>u''</code> (ODEs) or <code>u_t</code>, <code>u_x</code>, <code>u_xx</code>; <code>du/dt</code>, <code>d^2u/dx^2</code> and <code>u'(t)</code> are also understood.</li>
              <li>Operators <code>+ - * / ^</code>; functions {info ? info.grammar.functions.join(', ') : 'sin, cos, exp, ...'}; constants <code>pi</code> and <code>e</code>.</li>
              <li>Multiplication: <code>2u</code>, <code>k*u</code> or <code>k u</code> (with a space). <code>ku</code> is read as one name. An equation without <code>=</code> is read as <code>… = 0</code>.</li>
              <li>If your unknown is <code>y</code>, set "Unknown function" to <code>y</code> and write <code>y' = …</code>, <code>y(0) = …</code>. Domain limits may use <code>pi</code>, e.g. <code>2*pi</code>.</li>
              <li>ODE initial value problems of order 1 or 2: conditions <code>u(a)</code> and <code>u'(a)</code> at the start.</li>
              <li>Second-order boundary value problems: one condition at each end, <code>u(a)</code> or <code>u'(a)</code>, at least one on <code>u</code>.</li>
              <li>1D parabolic PDEs <code>u_t = F(x, t, u, u_x, u_xx)</code> with a positive <code>u_xx</code> coefficient; conditions <code>u(x, 0)</code>, <code>u(a, t)</code>, <code>u(b, t)</code>.</li>
              <li>Anything else is rejected with an explanation. Input is parsed as mathematics and never executed as code.</li>
            </ul>
          </details>

          <h2>Solver</h2>
          <label className="field">
            <span>Configuration</span>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {info && Object.entries(info.modes).map(([k, v]) => (
                <option key={k} value={k}>{k}: {v}</option>
              ))}
            </select>
          </label>
          <p className="hint">D is the full method and the recommended choice. A–C are simpler PINN baselines for comparison; E uses only a classical numerical solver.</p>
          <div className="row">
            <label className="field">
              <span>Seed</span>
              <input className="mono" value={seed} onChange={(e) => setSeed(e.target.value)} inputMode="numeric" />
            </label>
            <label className="field">
              <span>Time budget (s)</span>
              <input className="mono" value={budget} onChange={(e) => setBudget(e.target.value)} placeholder="default" inputMode="decimal" />
            </label>
          </div>
          <details className="syntax">
            <summary>Advanced: thresholds and ablations</summary>
            <div className="row">
              <label className="field">
                <span>Relative L2 threshold</span>
                <input className="mono" value={thrL2} onChange={(e) => setThrL2(e.target.value)} placeholder="default by class" />
              </label>
              <label className="field">
                <span>Condition error threshold</span>
                <input className="mono" value={thrCond} onChange={(e) => setThrCond(e.target.value)} placeholder="default by class" />
              </label>
            </div>
            {mode === 'D' && info && (
              <div className="checks">
                <span>Remove one component of D (ablation):</span>
                {info.ablations.map((a) => (
                  <label key={a} className="check">
                    <input type="checkbox" checked={ablate.includes(a)} onChange={(e) => setAblate((s) => (e.target.checked ? [...s, a] : s.filter((x) => x !== a)))} />
                    {a.replace('no_', 'no ')}
                  </label>
                ))}
              </div>
            )}
          </details>

          <div className="actions">
            <button className="primary" onClick={solve} disabled={busy || running || checking || !parsed} title={!parsed && !checking ? 'Fix the problem shown above first' : undefined}>{running ? 'Solving…' : 'Solve'}</button>
          </div>
          {formError && <div className="alert bad">{formError}</div>}
          {parsed && (
            <details className="syntax">
              <summary>How the problem was read (class, features)</summary>
              <ParsedView spec={parsed} />
            </details>
          )}

          {jobs.length > 0 && (
            <>
              <h2>Recent jobs</h2>
              <ul className="jobs">
                {jobs.slice(0, 8).map((j) => (
                  <li key={j.job_id}>
                    <button className="link" onClick={() => openJob(j.job_id)}>
                      <span className={`dot ${STATUS_STYLE[j.status]?.cls || (j.state === 'error' ? 'bad' : 'muted')}`} />
                      {j.problem_name} · {j.mode} · seed {j.seed}
                    </button>
                    <span className="muted small">{j.state}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        <section className="results">
          {!job && !report && <Welcome info={info} />}
          {job && <JobView job={job} onCancel={() => api.cancel(job.job_id).then(setJob).catch((e) => setFormError(e.message))} />}
          {report && <ReportView r={report} />}
        </section>
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
function Welcome({ info }) {
  return (
    <div className="panel">
      <h2>How it works</h2>
      <ol className="steps">
        <li><b>Parse and validate.</b> The equation is read with a restricted grammar, classified, and rejected with a reason if unsupported.</li>
        <li><b>Select a strategy.</b> A visible rule table picks the network, collocation points, built-in conditions and options.</li>
        <li><b>Train the PINN.</b> Adam then L-BFGS, within a time and point budget; adaptive points are added where the residual is large.</li>
        <li><b>Validate independently.</b> Fresh points, condition errors, and comparison with an exact or numerical reference against pre-declared thresholds.</li>
        <li><b>Refine, retry or fall back.</b> If the thresholds are not met, the solver refines and retries, then shows a clearly labelled numerical solution.</li>
      </ol>
      <p className="small muted">Pick an example on the left (or type your own equation) and press <b>Solve</b>. Results appear here.</p>
      {info && (
        <details className="syntax">
          <summary>Rule table: how the strategy is chosen ({info.rule_table.length} rules)</summary>
          <div className="tablewrap">
            <table>
              <thead><tr><th>Rule</th><th>Problem feature</th><th>Strategy tried first</th><th>Caution</th></tr></thead>
              <tbody>
                {info.rule_table.map((r) => (
                  <tr key={r.id}><td>{r.id}</td><td>{r.feature}</td><td>{r.strategy}</td><td>{r.caution}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  )
}

function ProblemPreview({ spec, error, checking, empty }) {
  if (empty) return <div className="preview muted small">Type an equation and it will be shown here as mathematics and checked as you type.</div>
  if (!spec && !error) return <div className="preview muted small">Checking…</div>
  if (!spec) {
    return (
      <div className={`preview ${checking ? 'stale' : ''}`}>
        <div className="alert warn">{error}</div>
      </div>
    )
  }
  return (
    <div className={`preview ${checking ? 'stale' : ''}`}>
      <Tex tex={spec.equation_latex} block />
      <div className="preview-conds">
        {(spec.conditions_latex || []).map((c) => <Tex key={c} tex={c} />)}
      </div>
      {spec.exact_latex && <div className="small muted">Exact solution (verified): <Tex tex={spec.exact_latex} /></div>}
      <div className="small okc">✓ Understood as a {spec.subtype || spec.kind.replace(/_/g, ' ')}{spec.linear ? ' (linear)' : ' (nonlinear)'}</div>
      {spec.warnings.map((w) => <div key={w} className="alert warn">{w}</div>)}
    </div>
  )
}

function ParsedView({ spec }) {
  return (
    <div className="parsed">
      <dl>
        <dt>Equation</dt><dd className="mono">{spec.equation}</dd>
        <dt>Explicit form</dt><dd className="mono">{spec.explicit_form}</dd>
        <dt>Conditions</dt>
        <dd>{spec.conditions.map((c) => <div key={c.text} className="mono">{c.text} <span className="muted">({c.role}, {c.type})</span></div>)}</dd>
        <dt>Linear</dt><dd>{spec.linear ? 'yes' : 'no'}</dd>
        {spec.exact_solution && (<><dt>Exact solution</dt><dd className="mono">{spec.exact_solution} <span className="muted">(verified)</span></dd></>)}
        <dt>Features</dt>
        <dd className="small">{Object.entries(spec.features).map(([k, v]) => <span key={k} className="chip">{k.replace(/_/g, ' ')}: {fmt(v)}</span>)}</dd>
      </dl>
    </div>
  )
}

function JobView({ job, onCancel }) {
  const running = ['queued', 'running'].includes(job.state)
  const trace = job.trace || []
  const lossData = useMemo(() => {
    if (!trace.length) return null
    const byAttempt = {}
    for (const e of trace) {
      if (e.loss_total == null) continue
      const k = e.attempt || 1
      ;(byAttempt[k] ||= { x: [], y: [] })
      byAttempt[k].x.push(e.time_s)
      byAttempt[k].y.push(e.loss_total)
    }
    return Object.entries(byAttempt).map(([k, v]) => ({ x: v.x, y: v.y, mode: 'lines', name: `attempt ${k}` }))
  }, [trace])
  const last = trace[trace.length - 1]
  if (!running && job.state === 'done') return null
  return (
    <div className="panel">
      <div className="jobhead">
        <h2>Job {job.job_id}</h2>
        {running && <button className="secondary" onClick={onCancel}>Cancel</button>}
      </div>
      <p>
        <span className={`dot ${running ? 'info pulse' : job.state === 'error' ? 'bad' : 'muted'}`} />
        {STAGE_TEXT[job.stage] || job.stage} · {fmt(job.elapsed_s)} s
        {last && running ? ` · iteration ${last.iteration} (${last.phase}) · ${last.n_points} points · loss ${fmt(last.loss_total)}` : ''}
      </p>
      {job.error && <div className="alert bad">{job.error}</div>}
      {job.state === 'cancelled' && <div className="alert warn">The job was cancelled.</div>}
      {lossData && lossData.length > 0 && (
        <Plot data={lossData} layout={{ title: 'Training loss (live)', xaxis: { title: 'training time (s)' }, yaxis: { type: 'log', title: 'loss' } }} height={260} />
      )}
      {job.events?.length > 0 && (
        <ul className="events small">
          {job.events.filter((e) => e.stage === 'attempt' || e.stage === 'validated').slice(-8).map((e, i) => (
            <li key={i}>
              {e.stage === 'attempt' ? `Attempt ${e.attempt} (${e.action}): ${e.description}` : `Attempt ${e.attempt} validated: ${e.passed ? 'passed' : 'not passed'}`}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
function ReportView({ r }) {
  const st = STATUS_STYLE[r.status] || { cls: 'info', label: r.status }
  const v = r.validation || r.fallback?.validation
  const files = r.files || []
  return (
    <div className="report">
      <div className={`banner ${st.cls}`}>
        <div className="banner-title">{st.label}</div>
        <div>{r.status_text}</div>
        <div className="small">Solution produced by: <b>{r.final_method_text}</b> · Configuration {r.mode} ({r.mode_name}) · run <span className="mono">{r.run_id}</span></div>
      </div>

      {r.warnings.length > 0 && (
        <div className="panel">
          <h2>Warnings</h2>
          {r.warnings.map((w) => <div key={w} className="alert warn">{w}</div>)}
        </div>
      )}

      <div className="panel">
        <h2>Solution</h2>
        <p className="small muted section-intro">The computed solution plotted against the reference, and how far apart they are at each point.</p>
        <SolutionPlots r={r} />
      </div>

      {v && (
        <div className="panel">
          <h2>Independent validation</h2>
          <p className="small muted section-intro">How the answer was checked. A metric with a threshold must be met for the result to pass; the others are shown for information.</p>
          <p className="small muted">
            {v.n_validation_points ? `${v.n_validation_points} fresh residual points (never used in training); ` : ''}
            errors on a uniform evaluation grid; criterion: {v.criterion === 'reference' ? 'comparison with the reference solution' : 'residual and conditions only'}.
          </p>
          <MetricsTable v={v} />
          {v.conditions?.length > 0 && (
            <div className="tablewrap">
              <table>
                <thead><tr><th>Condition</th><th>Type</th><th>How imposed</th><th>Max error</th><th>RMS error</th></tr></thead>
                <tbody>
                  {v.conditions.map((c) => (
                    <tr key={c.condition}><td className="mono">{c.condition}</td><td>{c.role}, {c.type}</td><td>{c.built_in ? 'built into the trial solution' : 'penalty term'}</td><td>{fmt(c.max_abs_error)}</td><td>{fmt(c.rms_error)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {v.failures.map((f) => <div key={f} className="alert bad">{f}</div>)}
          {v.notes.map((n) => <div key={n} className="note">{n}</div>)}
        </div>
      )}

      {r.fallback && (
        <div className="panel">
          <h2>{r.mode === 'E' ? 'Numerical solver' : 'Numerical fallback'}</h2>
          {r.fallback.available ? (
            <>
              <p><b>{r.fallback.solver.method}</b>: {r.fallback.note}</p>
              <KeyValues obj={r.fallback.solver.settings} />
            </>
          ) : <div className="alert bad">{r.fallback.error}</div>}
        </div>
      )}

      {r.attempts?.length > 0 && (
        <div className="panel">
          <h2>Training attempts</h2>
          <p className="small muted section-intro">Each attempt trains the network once; the solver retries with a refined setup if validation fails. ★ marks the attempt used.</p>
          <div className="tablewrap">
            <table>
              <thead><tr><th>#</th><th>Action</th><th>Network</th><th>Points</th><th>Adam / L-BFGS iters</th><th>Rel. L2</th><th>RMS residual</th><th>Passed</th><th>Time (s)</th></tr></thead>
              <tbody>
                {r.attempts.map((a) => (
                  <tr key={a.attempt} className={a.attempt === r.best_attempt ? 'best' : ''}>
                    <td>{a.attempt}{a.attempt === r.best_attempt ? ' ★' : ''}</td>
                    <td title={a.description}>{a.action}</td>
                    <td>{a.config.hidden.join('×')}{a.config.gpinn ? ' + gPINN' : ''}</td>
                    <td>{a.training.n_collocation_points}</td>
                    <td>{a.training.adam_iterations} / {a.training.lbfgs_iterations}</td>
                    <td>{fmt(a.validation.metrics.rel_l2)}</td>
                    <td>{fmt(a.validation.metrics.residual_rms)}</td>
                    <td>{a.validation.passed ? 'yes' : 'no'}</td>
                    <td>{fmt(a.time_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="small">
            {r.attempts.map((a) => <li key={a.attempt}><b>Attempt {a.attempt}:</b> {a.description}. {a.training.stop_reasons.join('; ')}.</li>)}
          </ul>
          {r.gpinn && (
            <div className="note">
              <b>gPINN gate:</b>{' '}
              {r.gpinn.tried
                ? `tried with λg = ${r.gpinn.lambda_g}; rel. L2 ${fmt(r.gpinn.rel_l2_before)} → ${fmt(r.gpinn.rel_l2_after)}, RMS residual ${fmt(r.gpinn.residual_rms_before)} → ${fmt(r.gpinn.residual_rms_after)}, cost ${fmt(r.gpinn.time_s)} s; ${r.gpinn.kept ? 'kept (validation improved)' : 'discarded (no improvement), previous weights restored'}.`
                : `not tried: ${r.gpinn.reason}.`}
            </div>
          )}
          <LossPlot history={r.loss_history} />
        </div>
      )}

      <h2 className="section-title">Technical details</h2>
      <div className="grid2">
        <div className="panel">
          <h2>Problem as parsed</h2>
          <dl>
            <dt>Class</dt><dd>{r.parsed.kind.replace(/_/g, ' ')}: {r.parsed.subtype}</dd>
            <dt>Equation</dt><dd>{r.parsed.equation_latex ? <Tex tex={r.parsed.equation_latex} /> : <span className="mono">{r.parsed.equation}</span>}</dd>
            <dt>Explicit form</dt><dd className="mono">{r.parsed.explicit_form}</dd>
            <dt>Domain</dt><dd className="mono">{Object.entries(r.parsed.domain).map(([k, d]) => `${k} ∈ [${fmt(d[0])}, ${fmt(d[1])}]`).join(',  ')}</dd>
            <dt>Conditions</dt><dd>{r.parsed.conditions.map((c) => <div key={c.text} className="mono">{c.text}</div>)}</dd>
            {Object.keys(r.parsed.parameters).length > 0 && (<><dt>Parameters</dt><dd className="mono">{Object.entries(r.parsed.parameters).map(([k, x]) => `${k} = ${fmt(x, 6)}`).join(', ')}</dd></>)}
            <dt>Features</dt><dd className="small">{Object.entries(r.parsed.features).map(([k, x]) => <span key={k} className="chip">{k.replace(/_/g, ' ')}: {fmt(x)}</span>)}</dd>
          </dl>
        </div>
        <div className="panel">
          <h2>Strategy and provenance</h2>
          <ul className="small">{r.strategy.reasons.map((x) => <li key={x}>{x}</li>)}</ul>
          {r.architecture && <p className="small">Network: {r.architecture.hidden_layers.join('×')} tanh, {r.architecture.parameters} parameters, inputs scaled to [−1, 1], float64; conditions {r.architecture.hard_constraints ? 'built into the trial solution' : 'imposed as penalty terms'}.</p>}
          {r.reference && (
            <p className="small">
              Reference: <b>{r.reference.method}</b> ({r.reference.kind}); estimated error {fmt(r.reference.estimated_max_error)}. {r.reference.notes.join(' ')}
            </p>
          )}
          <h3>Budget</h3>
          <KeyValues obj={r.budget} />
        </div>
      </div>

      <div className="grid2">
        <div className="panel">
          <h2>Limitations</h2>
          <ul className="small">{r.limitations.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
        <div className="panel">
          <h2>Reproducibility</h2>
          <KeyValues obj={{ seed: r.reproducibility.seed, validation_seed: r.reproducibility.validation_seed, config_fingerprint: r.reproducibility.config_fingerprint, precision: r.reproducibility.precision, torch_threads: r.reproducibility.torch_threads, ...r.reproducibility.versions }} />
          {files.length > 0 && (
            <p className="files small">
              Saved files: {files.map((f) => <a key={f} href={`/api/runs/${r.run_id}/${f}`} target="_blank" rel="noreferrer">{f}</a>)}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function KeyValues({ obj }) {
  return (
    <dl className="kv small">
      {Object.entries(obj || {}).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
        <div key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd className="mono">{fmt(v, 4)}</dd></div>
      ))}
    </dl>
  )
}

function MetricsTable({ v }) {
  const th = v.thresholds || {}
  const gating = v.criterion === 'reference' ? ['rel_l2', 'condition_max'] : ['residual_rms', 'condition_max']
  const rows = [
    ['rel_l2', 'Relative L2 error vs reference'],
    ['max_abs_error', 'Maximum absolute error'],
    ['residual_rms', 'RMS residual (fresh points)'],
    ['residual_max', 'Maximum residual (fresh points)'],
    ['condition_max', 'Maximum IC/BC error'],
  ]
  return (
    <div className="tablewrap">
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Check</th></tr></thead>
        <tbody>
          {rows.map(([k, label]) => {
            const val = v.metrics[k]
            const t = gating.includes(k) ? th[k] : null
            const ok = t == null || val == null ? null : val <= t
            return (
              <tr key={k}>
                <td>{label}</td>
                <td className="mono">{fmt(val)}</td>
                <td className="mono">{t == null ? '-' : `≤ ${fmt(t)}`}</td>
                <td>{ok === null ? <span className="muted">reported</span> : ok ? <span className="okc">✓ met</span> : <span className="badc">✗ not met</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function LossPlot({ history }) {
  const data = useMemo(() => {
    if (!history?.length) return null
    const keys = [['loss_total', 'total'], ['loss_residual', 'residual'], ['loss_initial', 'initial (penalty)'], ['loss_boundary', 'boundary (penalty)'], ['loss_gpinn', 'gPINN']]
    return keys
      .filter(([k]) => history.some((h) => h[k] != null))
      .map(([k, name]) => ({ x: history.map((h) => h.time_s), y: history.map((h) => (h[k] > 0 ? h[k] : null)), mode: 'lines', name }))
  }, [history])
  if (!data) return null
  return <Plot data={data} layout={{ title: 'Training loss', xaxis: { title: 'training time (s)' }, yaxis: { type: 'log' } }} height={280} />
}

function SolutionPlots({ r }) {
  const p = r.plots
  if (!p) return null
  if (p.dim === 1) return <Plots1D p={p} r={r} />
  return <Plots2D p={p} r={r} />
}

function Plots1D({ p, r }) {
  const x = p.axes.x
  const dep = r.parsed.dependent
  const sol = []
  if (p.reference) sol.push({ x, y: p.reference, mode: 'lines', name: `reference (${p.reference_label})`, line: { width: 6, color: 'rgba(120,120,120,0.35)' } })
  if (p.pinn) sol.push({ x, y: p.pinn, mode: 'lines', name: 'PINN', line: { width: 2, dash: 'dash', color: '#2a6fdb' } })
  if (p.numerical) sol.push({ x, y: p.numerical, mode: 'lines', name: 'numerical solution', line: { width: 2, dash: 'dot', color: '#d1495b' } })
  const err = []
  if (p.abs_error) err.push({ x, y: p.abs_error.map((e) => Math.max(e, 1e-17)), mode: 'lines', name: '|final − reference|', line: { color: '#e08a1e' } })
  if (p.residual) err.push({ x, y: p.residual.map((e) => Math.max(Math.abs(e), 1e-17)), mode: 'lines', name: '|PINN residual|', line: { color: '#2a9d8f' } })
  return (
    <div className="grid2 tight">
      <Plot data={sol} layout={{ title: `${dep}(${p.variables[0]})`, xaxis: { title: p.variables[0] }, yaxis: { title: dep } }} />
      {err.length > 0 && <Plot data={err} layout={{ title: 'Pointwise error and residual', xaxis: { title: p.variables[0] }, yaxis: { type: 'log' } }} />}
    </div>
  )
}

function Plots2D({ p, r }) {
  const xs = p.axes.x
  const ts = p.axes.t
  const sx = p.variables[0]
  const [k, setK] = useState(Math.floor((ts.length - 1) / 2))
  const heat = (z, title, colorscale = 'Viridis', abs = false) => (
    <Plot
      key={title}
      data={[{ type: 'heatmap', x: ts, y: xs, z: abs ? z.map((row) => row.map((v) => Math.abs(v))) : z, colorscale, colorbar: { thickness: 12, exponentformat: 'power' } }]}
      layout={{ title, xaxis: { title: 't' }, yaxis: { title: sx }, margin: { l: 50, r: 10, t: 34, b: 44 } }}
      height={300}
    />
  )
  const slice = useMemo(() => {
    const d = []
    if (p.reference) d.push({ x: xs, y: p.reference.map((row) => row[k]), mode: 'lines', name: 'reference', line: { width: 6, color: 'rgba(120,120,120,0.35)' } })
    if (p.final) d.push({ x: xs, y: p.final.map((row) => row[k]), mode: 'lines', name: p.final_label, line: { width: 2, dash: 'dash', color: '#2a6fdb' } })
    return d
  }, [p, xs, k])
  return (
    <>
      <div className="grid2 tight">
        {p.final && heat(p.final, `${p.final_label}`)}
        {p.reference && heat(p.reference, 'Reference')}
        {p.abs_error && heat(p.abs_error, '|final − reference|', 'Hot')}
        {p.residual && heat(p.residual, '|PINN residual|', 'Hot', true)}
      </div>
      <div className="slice">
        <label className="field">
          <span>Profile at t = {fmt(ts[k], 4)}</span>
          <input type="range" min={0} max={ts.length - 1} value={k} onChange={(e) => setK(Number(e.target.value))} />
        </label>
        <Plot data={slice} layout={{ title: `${r.parsed.dependent}(${sx}, t = ${fmt(ts[k], 4)})`, xaxis: { title: sx } }} height={280} />
      </div>
    </>
  )
}
