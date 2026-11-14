import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'

function themeColors() {
  const css = getComputedStyle(document.documentElement)
  return {
    fg: css.getPropertyValue('--fg').trim() || '#1b1f24',
    muted: css.getPropertyValue('--muted').trim() || '#5b6470',
    grid: css.getPropertyValue('--grid').trim() || '#e3e6ea',
  }
}

// Minimal Plotly wrapper: re-renders when data/layout change, cleans up on unmount.
export default function Plot({ data, layout, height = 320 }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const c = themeColors()
    const base = {
      height,
      margin: { l: 56, r: 16, t: 34, b: 56 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', size: 12, color: c.fg },
      legend: { orientation: 'h', y: -0.3, font: { size: 11 } },
      ...layout,
    }
    for (const ax of ['xaxis', 'yaxis']) {
      base[ax] = { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, exponentformat: 'power', ...(layout?.[ax] || {}) }
    }
    if (base.title && typeof base.title === 'string') base.title = { text: base.title, font: { size: 13 } }
    Plotly.react(el, data, base, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] })
  }, [data, layout, height])
  useEffect(() => {
    const el = ref.current
    return () => {
      if (el) Plotly.purge(el)
    }
  }, [])
  return <div ref={ref} className="plot" />
}
