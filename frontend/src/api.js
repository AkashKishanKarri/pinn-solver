// Thin client for the FastAPI backend.
const BACKEND_DOWN =
  'The solver server is offline right now (it runs on the author\'s laptop). Please try again later. ' +
  'Running locally? Start it with: .venv/bin/uvicorn pinnsolver.api:app --port 8000'

// Every call is short (training runs as a background job), so a request that has not answered
// in 20 s means the server is unreachable: e.g. the laptop hosting it is asleep, and the proxy
// would otherwise keep the request hanging.
const REQUEST_TIMEOUT_MS = 20000

async function request(path, options = {}) {
  let res
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS)
  try {
    res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
      signal: ctrl.signal,
    })
  } catch {
    throw new Error(BACKEND_DOWN)
  } finally {
    clearTimeout(timer) // only the wait for the response headers is limited, not reading a large report
  }
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  // The backend always answers with JSON; a 5xx without a JSON body comes from the
  // dev-server proxy (or a gateway) failing to reach it.
  if (!res.ok && body === null && res.status >= 500) throw new Error(BACKEND_DOWN)
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    if (body && body.detail) {
      msg = Array.isArray(body.detail)
        ? body.detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
        : body.detail
    }
    throw new Error(msg)
  }
  return body
}

export const api = {
  info: () => request('/api/info'),
  benchmarks: () => request('/api/benchmarks'),
  validate: (problem) => request('/api/validate', { method: 'POST', body: JSON.stringify(problem) }),
  submit: (job) => request('/api/jobs', { method: 'POST', body: JSON.stringify(job) }),
  job: (id) => request(`/api/jobs/${id}`),
  jobs: () => request('/api/jobs'),
  report: (id) => request(`/api/jobs/${id}/report`),
  cancel: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' }),
}
