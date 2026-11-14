// Thin client for the FastAPI backend.
async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
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
