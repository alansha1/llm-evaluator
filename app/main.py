"""LLM Evaluator — FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.database import engine, Base
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="LLM Response Evaluator",
    description="Automated benchmarking for language model responses",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.include_router(router)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Response Evaluator</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}

/* NAV */
nav{background:#1a1d27;border-bottom:1px solid #2d3748;padding:0 28px;display:flex;align-items:center;gap:0;height:56px;position:sticky;top:0;z-index:100}
.nav-brand{font-weight:800;font-size:1.05rem;color:#fff;margin-right:32px;white-space:nowrap}
.nav-brand span{color:#4299e1}
.nav-tab{padding:0 18px;height:56px;display:flex;align-items:center;font-size:0.88rem;font-weight:600;color:#718096;cursor:pointer;border-bottom:2px solid transparent;transition:all 0.15s;white-space:nowrap}
.nav-tab:hover{color:#e2e8f0}
.nav-tab.active{color:#4299e1;border-bottom-color:#4299e1}
.nav-right{margin-left:auto;font-size:0.8rem;color:#4a5568}
.nav-right a{color:#4299e1;text-decoration:none;font-weight:600}

/* PAGES */
.page{display:none;padding:28px;max-width:1100px;margin:0 auto}
.page.active{display:block}

/* STAT CARDS */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}
.stat-card{background:#1a1d27;border:1px solid #2d3748;border-radius:12px;padding:20px 24px}
.stat-label{font-size:0.78rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px}
.stat-value{font-size:2rem;font-weight:800;color:#fff;line-height:1}
.stat-sub{font-size:0.8rem;color:#4a5568;margin-top:6px}
.stat-card.blue .stat-value{color:#4299e1}
.stat-card.green .stat-value{color:#48bb78}
.stat-card.yellow .stat-value{color:#ecc94b}
.stat-card.purple .stat-value{color:#9f7aea}

/* SCORE BAR */
.score-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.score-row .name{font-size:0.82rem;color:#a0aec0;width:90px;text-align:right;flex-shrink:0}
.score-bar-wrap{flex:1;background:#2d3748;border-radius:20px;height:10px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:20px;transition:width 0.6s ease}
.score-row .val{font-size:0.82rem;font-weight:700;color:#e2e8f0;width:38px;text-align:right;flex-shrink:0}

/* SECTION */
.section{background:#1a1d27;border:1px solid #2d3748;border-radius:12px;padding:22px;margin-bottom:20px}
.section-title{font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:18px;display:flex;align-items:center;gap:8px}

/* EVALUATE FORM */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:640px){.form-grid{grid-template-columns:1fr}}
.form-group{display:flex;flex-direction:column;gap:6px}
.form-group.full{grid-column:1/-1}
label{font-size:0.78rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.05em}
input,select,textarea{background:#0f1117;border:1.5px solid #2d3748;border-radius:8px;color:#e2e8f0;padding:10px 14px;font-size:0.92rem;font-family:inherit;outline:none;transition:border 0.2s;width:100%}
input:focus,select:focus,textarea:focus{border-color:#4299e1}
textarea{min-height:72px;resize:vertical;font-family:'Courier New',monospace;font-size:0.85rem}
select option{background:#1a1d27}

.btn{border:none;border-radius:8px;padding:11px 24px;font-size:0.95rem;font-weight:700;cursor:pointer;transition:all 0.15s;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:#2b6cb0;color:#fff}
.btn-primary:hover{background:#3182ce}
.btn-primary:disabled{opacity:0.5;cursor:not-allowed}
.btn-secondary{background:#2d3748;color:#e2e8f0}
.btn-secondary:hover{background:#4a5568}

/* RESULT AREA */
.result-area{margin-top:18px;display:none}
.result-area.show{display:block}
.score-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}
.score-card{background:#0f1117;border-radius:10px;padding:14px;text-align:center;border:1px solid #2d3748}
.score-card .sc-label{font-size:0.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px}
.score-card .sc-val{font-size:1.6rem;font-weight:800}
.sc-green{color:#48bb78}.sc-blue{color:#4299e1}.sc-yellow{color:#ecc94b}.sc-purple{color:#9f7aea}.sc-red{color:#fc8181}

.response-box{background:#0f1117;border:1px solid #2d3748;border-radius:8px;padding:14px;font-size:0.85rem;color:#a0aec0;line-height:1.6;margin-bottom:12px}
.response-label{font-size:0.72rem;font-weight:700;color:#4a5568;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px}

.json-box{background:#0f1117;border:1px solid #2d3748;border-radius:8px;padding:14px;font-family:'Courier New',monospace;font-size:0.78rem;color:#68d391;overflow-x:auto;white-space:pre-wrap;max-height:280px;overflow-y:auto}
.json-box.err{color:#fc8181}

/* HISTORY TABLE */
.history-table{width:100%;border-collapse:collapse;font-size:0.85rem}
.history-table th{text-align:left;padding:10px 14px;font-size:0.72rem;font-weight:700;color:#4a5568;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #2d3748}
.history-table td{padding:10px 14px;border-bottom:1px solid #1a1d27;color:#a0aec0;vertical-align:middle}
.history-table tr:hover td{background:#1e2230}
.cat-badge{font-size:0.72rem;font-weight:700;padding:3px 9px;border-radius:20px;text-transform:uppercase}
.cat-factual{background:#1a365d;color:#90cdf4}
.cat-safety{background:#1c4532;color:#9ae6b4}
.cat-open{background:#322659;color:#d6bcfa}
.score-pill{font-weight:700;color:#48bb78}

/* SPINNER */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #4299e1;border-top-color:transparent;border-radius:50%;animation:spin 0.7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* CHART */
.chart-wrap{position:relative;height:200px}
svg.bar-chart{width:100%;height:100%}

/* BENCHMARK */
.provider-result{background:#0f1117;border:1px solid #2d3748;border-radius:10px;padding:16px;margin-bottom:12px}
.provider-name{font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.best-badge{background:#1c4532;color:#9ae6b4;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:20px}

.empty-state{text-align:center;padding:40px;color:#4a5568;font-size:0.9rem}
.empty-state .icon{font-size:2.5rem;margin-bottom:12px}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">🧪 <span>LLM</span> Evaluator</div>
  <div class="nav-tab active" onclick="showPage('dashboard',this)">📊 Dashboard</div>
  <div class="nav-tab" onclick="showPage('evaluate',this)">⚡ Evaluate</div>
  <div class="nav-tab" onclick="showPage('benchmark',this)">🏁 Benchmark</div>
  <div class="nav-tab" onclick="showPage('history',this)">📋 History</div>
  <div class="nav-right">Built by <a href="https://github.com/alansha1" target="_blank">Alan Sha ↗</a></div>
</nav>

<!-- ═══ DASHBOARD ═══ -->
<div class="page active" id="page-dashboard">
  <div class="stats-grid" id="stat-cards">
    <div class="stat-card blue"><div class="stat-label">Total Evaluations</div><div class="stat-value" id="s-runs">—</div><div class="stat-sub">prompts evaluated</div></div>
    <div class="stat-card green"><div class="stat-label">Avg Safety Score</div><div class="stat-value" id="s-safety">—</div><div class="stat-sub">out of 1.0</div></div>
    <div class="stat-card yellow"><div class="stat-label">Avg Overall Score</div><div class="stat-value" id="s-overall">—</div><div class="stat-sub">weighted composite</div></div>
    <div class="stat-card purple"><div class="stat-label">Best Provider</div><div class="stat-value" id="s-provider" style="font-size:1.1rem;padding-top:6px">—</div><div class="stat-sub">across benchmarks</div></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="section">
      <div class="section-title">📈 Score Breakdown</div>
      <div id="score-bars"><div class="empty-state"><div class="icon">📊</div>Run an evaluation first</div></div>
    </div>
    <div class="section">
      <div class="section-title">🕐 Recent Runs</div>
      <div id="recent-list"><div class="empty-state"><div class="icon">⚡</div>No runs yet</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">📊 Score History — last 10 runs</div>
    <div class="chart-wrap" id="chart-wrap"><div class="empty-state" style="padding:20px"><div class="icon">📉</div>Run some evaluations to see a chart</div></div>
  </div>
</div>

<!-- ═══ EVALUATE ═══ -->
<div class="page" id="page-evaluate">
  <div class="section">
    <div class="section-title">⚡ Evaluate a Prompt</div>
    <div class="form-grid">
      <div class="form-group full">
        <label>Prompt *</label>
        <input id="e-prompt" type="text" placeholder="e.g. What is machine learning?" value="What is machine learning?">
      </div>
      <div class="form-group">
        <label>Category</label>
        <select id="e-category">
          <option value="factual">factual</option>
          <option value="safety">safety</option>
          <option value="open">open</option>
        </select>
      </div>
      <div class="form-group">
        <label>Provider</label>
        <select id="e-provider">
          <option value="mock">mock (no API key needed)</option>
          <option value="openai">openai</option>
          <option value="huggingface">huggingface</option>
        </select>
      </div>
      <div class="form-group full">
        <label>Reference Answer <span style="font-weight:400;text-transform:none;color:#4a5568">(optional — needed for Accuracy score)</span></label>
        <input id="e-ref" type="text" placeholder="The correct answer for comparison...">
      </div>
    </div>
    <button class="btn btn-primary" id="e-btn" onclick="runEval()" style="margin-top:16px">▶ Run Evaluation</button>

    <div class="result-area" id="e-result">
      <div class="score-cards" id="e-score-cards"></div>
      <div class="response-label">Model Response</div>
      <div class="response-box" id="e-response"></div>
      <div class="response-label" style="margin-top:12px">Full JSON</div>
      <div class="json-box" id="e-json"></div>
    </div>
  </div>
</div>

<!-- ═══ BENCHMARK ═══ -->
<div class="page" id="page-benchmark">
  <div class="section">
    <div class="section-title">🏁 Run Full Benchmark</div>
    <div class="form-grid">
      <div class="form-group">
        <label>Prompt Set</label>
        <select id="b-set">
          <option value="default">default — 5 mixed prompts</option>
          <option value="safety">safety — 4 harm-detection prompts</option>
          <option value="factual">factual — 5 factual Q&A prompts</option>
        </select>
      </div>
      <div class="form-group">
        <label>Provider</label>
        <select id="b-provider">
          <option value="mock">mock</option>
          <option value="openai">openai</option>
          <option value="huggingface">huggingface</option>
        </select>
      </div>
    </div>
    <button class="btn btn-primary" id="b-btn" onclick="runBench()" style="margin-top:16px">▶ Run Benchmark</button>

    <div class="result-area" id="b-result">
      <div id="b-provider-results"></div>
      <div class="response-label" style="margin-top:12px">Full JSON</div>
      <div class="json-box" id="b-json"></div>
    </div>
  </div>
</div>

<!-- ═══ HISTORY ═══ -->
<div class="page" id="page-history">
  <div class="section">
    <div class="section-title">📋 Evaluation History <button class="btn btn-secondary" onclick="loadHistory()" style="font-size:0.78rem;padding:6px 14px;margin-left:8px">↻ Refresh</button></div>
    <div id="history-content"><div class="empty-state"><div class="icon">📋</div>Loading...</div></div>
  </div>
</div>

<script>
// ── Navigation ──────────────────────────────────────────────────────────────
function showPage(name, tab) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  tab.classList.add('active');
  if (name === 'dashboard') loadDashboard();
  if (name === 'history') loadHistory();
}

// ── Dashboard ───────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [stats, evals] = await Promise.all([
      fetch('/api/v1/dashboard/stats').then(r => r.json()),
      fetch('/api/v1/evaluations').then(r => r.json()),
    ]);

    document.getElementById('s-runs').textContent = stats.total_runs ?? 0;
    document.getElementById('s-safety').textContent = stats.avg_safety_score != null ? stats.avg_safety_score.toFixed(2) : '—';
    document.getElementById('s-overall').textContent = stats.avg_overall_score != null ? stats.avg_overall_score.toFixed(2) : '—';
    document.getElementById('s-provider').textContent = stats.best_performing_provider || 'mock';

    renderScoreBars(stats);
    renderRecentList(evals.slice(0, 5));
    renderChart(evals.slice(0, 10).reverse());
  } catch(e) { console.error(e); }
}

function renderScoreBars(stats) {
  const bars = [
    { name: 'Accuracy',    val: stats.avg_accuracy_score,    color: '#4299e1' },
    { name: 'Safety',      val: stats.avg_safety_score,      color: '#48bb78' },
    { name: 'Coherence',   val: stats.avg_coherence_score,   color: '#ecc94b' },
    { name: 'Consistency', val: stats.avg_consistency_score, color: '#9f7aea' },
    { name: 'Overall',     val: stats.avg_overall_score,     color: '#fc8181' },
  ];
  const wrap = document.getElementById('score-bars');
  if (!bars.some(b => b.val != null && b.val > 0)) {
    wrap.innerHTML = '<div class="empty-state"><div class="icon">📊</div>Run an evaluation first</div>';
    return;
  }
  wrap.innerHTML = bars.map(b => {
    const v = b.val != null ? b.val : 0;
    const pct = (v * 100).toFixed(0);
    return `<div class="score-row">
      <span class="name">${b.name}</span>
      <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${pct}%;background:${b.color}"></div></div>
      <span class="val">${v.toFixed(2)}</span>
    </div>`;
  }).join('');
}

function renderRecentList(evals) {
  const wrap = document.getElementById('recent-list');
  if (!evals.length) { wrap.innerHTML = '<div class="empty-state"><div class="icon">⚡</div>No runs yet</div>'; return; }
  wrap.innerHTML = evals.map(e => `
    <div style="padding:8px 0;border-bottom:1px solid #2d3748;font-size:0.82rem">
      <div style="color:#e2e8f0;margin-bottom:2px">${e.prompt.slice(0,55)}${e.prompt.length>55?'…':''}</div>
      <div style="color:#4a5568">${e.category} · ${new Date(e.created_at).toLocaleTimeString()}</div>
    </div>`).join('');
}

function renderChart(evals) {
  const wrap = document.getElementById('chart-wrap');
  if (!evals.length) { wrap.innerHTML = '<div class="empty-state" style="padding:20px"><div class="icon">📉</div>Run some evaluations to see a chart</div>'; return; }

  const W = 700, H = 180, pad = { l:30, r:10, t:10, b:30 };
  const n = evals.length;
  const barW = Math.min(40, (W - pad.l - pad.r) / n - 4);
  const step = (W - pad.l - pad.r) / n;

  const bars = evals.map((e, i) => {
    const x = pad.l + i * step + step / 2 - barW / 2;
    const h = H - pad.t - pad.b;
    return `<rect x="${x.toFixed(1)}" y="${(pad.t + h * (1 - 0)).toFixed(1)}" width="${barW}" height="0" fill="#4299e1" rx="3" opacity="0.85">
      <animate attributeName="height" from="0" to="${(0).toFixed(1)}" dur="0s" fill="freeze"/>
      <animate attributeName="y" from="${(pad.t + h).toFixed(1)}" to="${(pad.t + h).toFixed(1)}" dur="0s" fill="freeze"/>
    </rect>`;
  });

  // Simple overall score line chart
  const points = evals.map((e, i) => {
    const x = pad.l + i * step + step / 2;
    const y = pad.t + (H - pad.t - pad.b) * (1 - 0);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  // Re-fetch full eval data to get scores
  fetchScoresAndDraw(evals, W, H, pad, step, barW, wrap);
}

async function fetchScoresAndDraw(evals, W, H, pad, step, barW, wrap) {
  const scores = [];
  for (const e of evals) {
    try {
      const r = await fetch(`/api/v1/evaluate/${e.run_id}`);
      const d = await r.json();
      const s = d.results && d.results[0] ? d.results[0].scores.overall_score : 0;
      scores.push(s || 0);
    } catch { scores.push(0); }
  }

  const h = H - pad.t - pad.b;
  const rects = scores.map((s, i) => {
    const x = (pad.l + i * step + step / 2 - barW / 2).toFixed(1);
    const barH = (h * s).toFixed(1);
    const y = (pad.t + h * (1 - s)).toFixed(1);
    const color = s >= 0.8 ? '#48bb78' : s >= 0.6 ? '#ecc94b' : '#fc8181';
    return `<rect x="${x}" y="${y}" width="${barW}" height="${barH}" fill="${color}" rx="3" opacity="0.85"/>
      <text x="${(parseFloat(x)+barW/2).toFixed(1)}" y="${(parseFloat(y)-4).toFixed(1)}" text-anchor="middle" font-size="9" fill="#718096">${(s*100).toFixed(0)}%</text>`;
  });

  const labels = evals.map((e, i) => {
    const x = (pad.l + i * step + step / 2).toFixed(1);
    return `<text x="${x}" y="${(H - 6).toFixed(1)}" text-anchor="middle" font-size="9" fill="#4a5568">#${i+1}</text>`;
  });

  // Y axis
  const yLines = [0, 0.25, 0.5, 0.75, 1].map(v => {
    const y = (pad.t + h * (1 - v)).toFixed(1);
    return `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#2d3748" stroke-width="1"/>
      <text x="${pad.l - 4}" y="${(parseFloat(y)+3).toFixed(1)}" text-anchor="end" font-size="9" fill="#4a5568">${v.toFixed(2)}</text>`;
  });

  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%">
    ${yLines.join('')}
    ${rects.join('')}
    ${labels.join('')}
  </svg>`;
}

// ── Evaluate ────────────────────────────────────────────────────────────────
async function runEval() {
  const btn = document.getElementById('e-btn');
  const resultArea = document.getElementById('e-result');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';

  const body = {
    prompt: document.getElementById('e-prompt').value,
    category: document.getElementById('e-category').value,
    reference_answer: document.getElementById('e-ref').value || null,
    providers: [document.getElementById('e-provider').value.split(' ')[0]]
  };

  try {
    const res = await fetch('/api/v1/evaluate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    resultArea.classList.add('show');

    if (data.results && data.results[0]) {
      const s = data.results[0].scores;
      const metrics = [
        { label:'Accuracy',    val: s.accuracy_score,    cls:'sc-blue' },
        { label:'Safety',      val: s.safety_score,      cls:'sc-green' },
        { label:'Coherence',   val: s.coherence_score,   cls:'sc-yellow' },
        { label:'Consistency', val: s.consistency_score, cls:'sc-purple' },
        { label:'Overall',     val: s.overall_score,     cls:'sc-green' },
        { label:'Latency',     val: s.latency_ms,        cls:'sc-blue', unit:'ms', raw:true },
      ];
      document.getElementById('e-score-cards').innerHTML = metrics.map(m => `
        <div class="score-card">
          <div class="sc-label">${m.label}</div>
          <div class="sc-val ${m.cls}">${m.val != null ? (m.raw ? m.val.toFixed(0) : (m.val*100).toFixed(0)+'%') : '—'}</div>
        </div>`).join('');
      document.getElementById('e-response').textContent = data.results[0].response_text || '(no response)';
    }

    document.getElementById('e-json').textContent = JSON.stringify(data, null, 2);
    document.getElementById('e-json').className = 'json-box' + (res.ok ? '' : ' err');
  } catch(e) {
    resultArea.classList.add('show');
    document.getElementById('e-json').textContent = 'Error: ' + e.message;
    document.getElementById('e-json').className = 'json-box err';
  }

  btn.disabled = false;
  btn.innerHTML = '▶ Run Evaluation';
}

// ── Benchmark ───────────────────────────────────────────────────────────────
async function runBench() {
  const btn = document.getElementById('b-btn');
  const resultArea = document.getElementById('b-result');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running benchmark...';

  const body = {
    providers: [document.getElementById('b-provider').value.split(' ')[0]],
    prompt_set: document.getElementById('b-set').value.split(' ')[0]
  };

  try {
    const res = await fetch('/api/v1/benchmark', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    resultArea.classList.add('show');

    if (data.provider_scores) {
      const best = data.best_provider;
      document.getElementById('b-provider-results').innerHTML = Object.entries(data.provider_scores).map(([prov, scores]) => `
        <div class="provider-result">
          <div class="provider-name">${prov} ${prov === best ? '<span class="best-badge">✓ Best</span>' : ''}</div>
          ${['overall_score','accuracy_score','safety_score','coherence_score','consistency_score'].map(k => {
            const v = scores[k];
            if (v == null) return '';
            const pct = (v * 100).toFixed(0);
            const color = v >= 0.8 ? '#48bb78' : v >= 0.6 ? '#ecc94b' : '#fc8181';
            return `<div class="score-row">
              <span class="name" style="width:110px">${k.replace('_score','').replace('_',' ')}</span>
              <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${pct}%;background:${color}"></div></div>
              <span class="val">${v.toFixed(2)}</span>
            </div>`;
          }).join('')}
        </div>`).join('');
    }

    document.getElementById('b-json').textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    resultArea.classList.add('show');
    document.getElementById('b-json').textContent = 'Error: ' + e.message;
    document.getElementById('b-json').className = 'json-box err';
  }

  btn.disabled = false;
  btn.innerHTML = '▶ Run Benchmark';
}

// ── History ──────────────────────────────────────────────────────────────────
async function loadHistory() {
  const wrap = document.getElementById('history-content');
  wrap.innerHTML = '<div class="empty-state"><div class="icon">⏳</div>Loading...</div>';
  try {
    const evals = await fetch('/api/v1/evaluations').then(r => r.json());
    if (!evals.length) { wrap.innerHTML = '<div class="empty-state"><div class="icon">📋</div>No evaluations yet — run one first</div>'; return; }
    wrap.innerHTML = `<table class="history-table">
      <thead><tr><th>Prompt</th><th>Category</th><th>Time</th></tr></thead>
      <tbody>${evals.map(e => `<tr>
        <td style="color:#e2e8f0;max-width:400px">${e.prompt.slice(0,70)}${e.prompt.length>70?'…':''}</td>
        <td><span class="cat-badge cat-${e.category}">${e.category}</span></td>
        <td>${new Date(e.created_at).toLocaleString()}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  } catch(e) {
    wrap.innerHTML = '<div class="empty-state"><div class="icon">❌</div>Failed to load history</div>';
  }
}

// Auto-load dashboard
loadDashboard();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/docs", response_class=HTMLResponse)
async def docs_redirect():
    return HTMLResponse(content='<meta http-equiv="refresh" content="0;url=/">')
