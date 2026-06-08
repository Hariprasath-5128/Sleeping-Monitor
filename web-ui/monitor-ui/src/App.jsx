import React, { useMemo } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { useLiveData, riskColor } from './hooks'
import './App.css'

// ─── Small helpers ───────────────────────────────────────────────────────────
const fmt = (v, d = 1) => (typeof v === 'number' ? v.toFixed(d) : '—')

function Badge({ label, variant = 'cyan' }) {
  const colors = {
    green:  { bg: 'transparent',  border: 'var(--green)',  color: 'var(--green)' },
    red:    { bg: 'rgba(239, 68, 68, 0.1)',   border: 'var(--red)',   color: 'var(--red)' },
    orange: { bg: 'transparent',   border: 'var(--orange)',   color: 'var(--orange)' },
    cyan:   { bg: 'transparent',   border: 'var(--cyan)',   color: 'var(--cyan)' },
  }
  const c = colors[variant] || colors.cyan
  return (
    <span style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.color,
      borderRadius: 2, padding: '2px 8px', fontSize: 10,
      fontWeight: 600, whiteSpace: 'nowrap', textTransform: 'uppercase'
    }}>{label}</span>
  )
}

// ─── Edge-Warning Banner ─────────────────────────────────────────────────────
function EdgeWarningBanner({ data }) {
  if (!data?.on_edge) return null
  const side = data.edge_side
  return (
    <div className="edge-banner">
      <span className="edge-icon">⚠</span>
      <span>
        <strong>EDGE CONTACT DETECTED</strong>
        &nbsp;— Whitener boundary intersects the <strong>{side}</strong> danger zone!
        &nbsp;(warp_x = {fmt(data.warp_x, 1)}&nbsp;·&nbsp;{side === 'LEFT' ? `${fmt(data.warp_x,1)} < LEFT_Z 60` : `${fmt(data.warp_x,1)} > RIGHT_Z 340`})
      </span>
    </div>
  )
}

// ─── Stat Card ───────────────────────────────────────────────────────────────
function StatCard({ title, value, sub, valueColor = 'var(--text)', highlight }) {
  return (
    <div className={`panel stat-card ${highlight ? 'stat-card--alert' : ''}`}>
      <div className="panel-title">{title}</div>
      <div className="stat-value mono" style={{ color: valueColor }}>{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

// ─── Custom Recharts Tooltip ─────────────────────────────────────────────────
function DarkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="recharts-tooltip">
      <div className="rt-label">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

// ─── Position History Chart ──────────────────────────────────────────────────
function PositionChart({ history = [] }) {
  return (
    <div className="panel chart-panel">
      <div className="panel-title">LIVE POSITION HISTORY (WARP X)</div>
      <ResponsiveContainer width="100%" height={185}>
        <AreaChart data={history} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="var(--cyan)" stopOpacity={0.25} />
              <stop offset="95%" stopColor="var(--cyan)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="index" stroke="var(--muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--muted)" tick={{ fontSize: 10 }} width={40} domain={['auto','auto']} />
          <Tooltip content={<DarkTooltip />} />
          <Area type="monotone" dataKey="warp_x" stroke="var(--cyan)"
            fill="url(#cyanGrad)" strokeWidth={1.5}
            dot={{ r: 0 }}
            activeDot={{ r: 4, fill: 'var(--cyan)', strokeWidth: 0 }}
            name="Warp X" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── System Status ───────────────────────────────────────────────────────────
function SystemStatus({ data }) {
  const cur  = data?.risk_pred || '—'
  const next = data?.next_pred || '—'
  const conf = data?.confidence ?? 0
  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel-title">SYSTEM STATUS</div>

      <div>
        <div className="status-row-label">CURRENT STATE</div>
        <div className="status-state" style={{ color: riskColor(cur) }}>
          <span className="status-dot" style={{ background: riskColor(cur) }} />
          {cur}
        </div>
      </div>

      <div>
        <div className="status-row-label">NEXT PREDICTED</div>
        <div className="status-state" style={{ color: riskColor(next) }}>
          <span className="status-dot" style={{ background: riskColor(next) }} />
          {next}
        </div>
      </div>

      <div>
        <div className="panel-title" style={{ marginBottom: 4 }}>CONFIDENCE</div>
        <div className="conf-value mono">{fmt(conf, 0)}%</div>
        <div className="conf-bar-track">
          <div className="conf-bar-fill"
            style={{ width: `${conf}%`, background: riskColor(cur) }} />
        </div>
      </div>
    </div>
  )
}

// ─── Model Status ─────────────────────────────────────────────────────────────
function ModelStatus({ data }) {
  const weights = data?.weights || {}
  const accuracy = data?.model_accuracy ?? 0
  const drift = data?.drift_intensity ?? 0
  const driftDet = data?.drift_detected

  return (
    <div className="panel">
      <div className="panel-title">MODEL STATUS</div>

      <div className="model-row">
        <span className="model-key">Adaptive Learning</span>
        <Badge label="ACTIVE" variant="green" />
      </div>
      <div className="model-row">
        <span className="model-key">Drift Detected</span>
        <Badge label={driftDet ? 'ALERT' : 'NONE'} variant={driftDet ? 'red' : 'green'} />
      </div>
      <div className="model-row">
        <span className="model-key">Drift Intensity</span>
        <span className="model-val mono" style={driftDet ? { color: 'var(--orange)' } : {}}>
          {fmt(drift, 3)}
        </span>
      </div>
      <div className="model-row">
        <span className="model-key">Model Accuracy</span>
        <span className="model-val mono" style={{ color: 'var(--cyan)' }}>
          {fmt(accuracy, 1)}%
        </span>
      </div>

      <div className="section-sub">ADAPTIVE WEIGHTS</div>

      {[
        { label: 'Velocity',     key: 'velocity',     color: 'var(--cyan)' },
        { label: 'Acceleration', key: 'acceleration', color: 'var(--orange)' },
        { label: 'Position',     key: 'position',     color: 'var(--purple)' },
      ].map(({ label, key, color }) => {
        const val = weights[key] ?? 0
        return (
          <div key={key} className="weight-row">
            <span className="weight-label">{label}</span>
            <div className="weight-track">
              <div className="weight-fill"
                style={{ width: `${(val * 100).toFixed(1)}%`, background: color }} />
            </div>
            <span className="weight-val mono">{val.toFixed(3)}</span>
          </div>
        )
      })}
    </div>
  )
}

// ─── Forecast Chart ───────────────────────────────────────────────────────────
function ForecastChart({ forecast = [] }) {
  return (
    <div className="panel chart-panel">
      <div className="panel-title">5-STEP POSITION FORECAST</div>
      <ResponsiveContainer width="100%" height={195}>
        <LineChart data={forecast} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="label" stroke="var(--muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--muted)" tick={{ fontSize: 10 }} width={40} domain={['auto','auto']} />
          <Tooltip content={<DarkTooltip />} />
          <Line type="monotone" dataKey="value" stroke="var(--orange)" strokeWidth={1.5}
            dot={{ r: 3, fill: 'var(--orange)', strokeWidth: 0 }}
            activeDot={{ r: 5 }} name="Forecast" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Risk Score History ───────────────────────────────────────────────────────
function RiskHistoryChart({ history = [] }) {
  return (
    <div className="panel chart-panel">
      <div className="panel-title">RISK SCORE HISTORY</div>
      <ResponsiveContainer width="100%" height={195}>
        <AreaChart data={history} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="redGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="var(--red)" stopOpacity={0.2} />
              <stop offset="95%" stopColor="var(--red)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="index" stroke="var(--muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--muted)" tick={{ fontSize: 10 }} domain={[0, 1]} width={40} />
          <Tooltip content={<DarkTooltip />} />
          <ReferenceLine y={0.7} stroke="var(--red)" strokeDasharray="4 4" opacity={0.5} />
          <ReferenceLine y={0.4} stroke="var(--orange)" strokeDasharray="4 4" opacity={0.5} />
          <Area type="stepAfter" dataKey="risk_score" stroke="var(--red)"
            fill="url(#redGrad)" strokeWidth={1.5}
            dot={false}
            name="Risk Score" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Edge Zone Visualiser ─────────────────────────────────────────────────────
function EdgeZoneVisualiser({ data }) {
  const wx = data?.warp_x ?? 200
  const LEFT_Z  = 60
  const RIGHT_Z = 340
  const WARP    = 400
  const pct     = Math.max(0, Math.min(100, (wx / WARP) * 100))
  const onEdge  = data?.on_edge

  return (
    <div className="panel">
      <div className="panel-title">EDGE ZONE TRACKING</div>
      <div style={{ marginBottom: 8, fontSize: 11 }}>
        <span style={{ color: 'var(--muted)' }}>Warp Position: </span>
        <span className="mono" style={{ color: onEdge ? 'var(--red)' : 'var(--text)', fontWeight: 600 }}>
          {fmt(wx, 1)}
        </span>
        <span style={{ color: 'var(--muted)' }}> / 400</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style={{ color: 'var(--muted)' }}>Thresholds: </span>
        <span className="mono" style={{ color: 'var(--red)' }}>L=60</span>
        &nbsp;&nbsp;
        <span className="mono" style={{ color: 'var(--red)' }}>R=340</span>
      </div>

      {/* Zone bar */}
      <div className="zone-bar-track">
        <div className="zone-seg zone-danger" style={{ width: `${(LEFT_Z / WARP) * 100}%` }}>LF</div>
        <div className="zone-seg zone-safe" style={{ width: `${((RIGHT_Z - LEFT_Z) / WARP) * 100}%` }}>SAFE ZONE</div>
        <div className="zone-seg zone-danger" style={{ width: `${((WARP - RIGHT_Z) / WARP) * 100}%` }}>RT</div>
        {/* Whitener marker */}
        <div className="whitener-marker" style={{ left: `${pct}%` }} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: 'var(--muted)' }}>
        <span>0</span>
        <span>60</span>
        <span>200</span>
        <span>340</span>
        <span>400</span>
      </div>

      {onEdge && (
        <div className="zone-alert">
          ⚠ Detection intersects {data.edge_side} boundary
        </div>
      )}
    </div>
  )
}

// ─── Per-Class Metrics Table ─────────────────────────────────────────────────
function MetricsTable({ metrics = {} }) {
  const rows = Object.entries(metrics).filter(([k]) =>
    !['accuracy','macro avg','weighted avg'].includes(k)
  )
  const metColor = { 'STABLE': 'var(--green)', 'FALL IMMINENT': 'var(--red)', 'DRIFT WARNING': 'var(--orange)' }

  return (
    <div className="panel">
      <div className="panel-title">PER-CLASS CLASSIFIER METRICS</div>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([cls, m]) => (
            <tr key={cls}>
              <td style={{ color: metColor[cls] || 'var(--cyan)', fontWeight: 600 }}>{cls}</td>
              <td className="mono">{(m.precision * 100).toFixed(1)}%</td>
              <td className="mono">{(m.recall * 100).toFixed(1)}%</td>
              <td className="mono">{(m['f1-score'] * 100).toFixed(1)}%</td>
              <td className="mono">{m.support}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── App Root ─────────────────────────────────────────────────────────────────
export default function App() {
  const { data, lastUpdate } = useLiveData()
  const d = data || {}

  const fallProb   = d.fall_probability ?? null
  const riskScore  = d.risk_score ?? null
  const sleepQ     = d.sleep_quality ?? null
  const timeToFall = d.time_to_fall ?? 'N/A'
  const cur        = d.risk_pred || '—'

  const fallColor = fallProb !== null
    ? fallProb > 60 ? 'var(--red)' : fallProb > 30 ? 'var(--orange)' : 'var(--green)'
    : 'var(--text)'

  const riskScoreColor = riskScore !== null
    ? riskScore > 0.7 ? 'var(--red)' : riskScore > 0.4 ? 'var(--orange)' : 'var(--cyan)'
    : 'var(--text)'

  const tsStr = lastUpdate
    ? lastUpdate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—'

  return (
    <div className="app">
      {/* ══ HEADER ══ */}
      <header className="header">
        <div>
          <h1 className="header-title">
            Patient <span className="accent">Safety Monitor</span>
          </h1>
          {d.n_train && (
            <div className="header-sub">
              RF Active · {d.n_train} train / {d.n_test} test ·&nbsp;
              <span className="accent mono">{fmt(d.model_accuracy, 1)}%</span> acc
            </div>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {d.on_edge && (
            <div className="header-edge-alert">
              BOUNDARY BREACH
            </div>
          )}
          <div className="live-badge">
            <span className="live-dot" />
            LIVE
          </div>
        </div>
      </header>

      {/* ══ EDGE WARNING BANNER ══ */}
      <EdgeWarningBanner data={d} />

      <div className="dashboard">
        {/* ── ROW 1: STATS ── */}
        <div className="row-stats">
          <StatCard
            title="FALL PROBABILITY"
            value={fallProb !== null ? `${fmt(fallProb, 1)}%` : '—'}
            sub="Model predicted risk"
            valueColor={fallColor}
            highlight={fallProb > 60}
          />
          <StatCard
            title="TIME TO FALL"
            value={timeToFall}
            sub="Estimated safety window"
            valueColor={timeToFall !== 'N/A' ? 'var(--red)' : 'var(--text)'}
          />
          <StatCard
            title="RISK SCORE"
            value={riskScore !== null ? fmt(riskScore, 2) : '—'}
            sub="Composite health index"
            valueColor={riskScoreColor}
          />
          <StatCard
            title="SLEEP AGITATION"
            value={sleepQ !== null ? `${fmt(sleepQ, 1)}%` : '—'}
            sub="Movement instability"
            valueColor="var(--orange)"
          />
        </div>

        {/* ── ROW 2: POSITION + STATUS + MODEL ── */}
        <div className="row-mid">
          <PositionChart history={d.history || []} />
          <SystemStatus data={d} />
          <ModelStatus data={d} />
        </div>

        {/* ── ROW 3: EDGE ZONE + METRICS ── */}
        <div className="row-edge">
          <EdgeZoneVisualiser data={d} />
          <MetricsTable metrics={d.per_class_metrics || {}} />
        </div>

        {/* ── ROW 4: FORECAST + RISK HISTORY ── */}
        <div className="row-charts">
          <ForecastChart forecast={d.forecast || []} />
          <RiskHistoryChart history={d.history || []} />
        </div>
      </div>

      {/* ══ FOOTER ══ */}
      <footer className="footer">
        <span>Patient Safety Monitor (Enterprise Edition v1.2)</span>
        <span>Latched to: <strong className="accent mono">{tsStr}</strong></span>
      </footer>
    </div>
  )
}
