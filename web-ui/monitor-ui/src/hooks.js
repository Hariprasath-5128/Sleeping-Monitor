import { useState, useEffect, useCallback, useRef } from 'react'

const API = '/api/status'
const SSE = '/stream'
const POLL_MS = 3000

// ─── Color helpers ──────────────────────────────────────────────────────────
export function riskColor(state) {
  if (!state) return 'var(--muted)'
  const s = state.toUpperCase()
  if (s.includes('FALL'))  return 'var(--red)'
  if (s.includes('DRIFT')) return 'var(--orange)'
  return 'var(--green)'
}

export function riskBg(state) {
  if (!state) return 'transparent'
  const s = state.toUpperCase()
  if (s.includes('FALL'))  return 'rgba(239, 68, 68, 0.1)'
  if (s.includes('DRIFT')) return 'rgba(245, 158, 11, 0.1)'
  return 'rgba(16, 185, 129, 0.1)'
}

// ─── Custom hook: live data from /api/status ─────────────────────────────────
export function useLiveData() {
  const [data, setData] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const timerRef = useRef(null)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(API)
      if (!res.ok) return
      const json = await res.json()
      setData(json)
      setLastUpdate(new Date())
    } catch (e) { /* silent */ }
  }, [])

  useEffect(() => {
    fetchData()

    // Try SSE first (pushes when live_predictions.json changes)
    let es
    try {
      es = new EventSource(SSE)
      es.onmessage = () => fetchData()   // re-fetch status on any SSE push
      es.onerror  = () => { es.close(); es = null }
    } catch(e) {}

    // Always keep a polling fallback
    timerRef.current = setInterval(fetchData, POLL_MS)

    return () => {
      clearInterval(timerRef.current)
      if (es) es.close()
    }
  }, [fetchData])

  return { data, lastUpdate, refetch: fetchData }
}
