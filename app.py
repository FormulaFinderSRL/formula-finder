#!/usr/bin/env python3
"""
Correlation Discovery Engine — powered by Claude AI
"""
import numpy as np
import pandas as pd
import base64
import io
import os
import warnings
import json
warnings.filterwarnings("ignore")

# ─────────────────────── MATH CORE ───────────────────────

def compute_correlations(df):
    """Compute Pearson correlation matrix for all numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr()

def detect_relationship_type(x, y):
    """Try multiple fit types, return best one with r2 and label."""
    from scipy import stats
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"type": "unknown", "r2": 0, "formula": "n/a", "confidence": 0}

    results = []

    # Linear
    try:
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r**2
        results.append({"type": "Linear", "r2": r2,
            "formula": f"y = {slope:.3f}x + {intercept:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    # Logarithmic
    try:
        xp = np.where(x > 0, x, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), y)
        r2 = r**2
        results.append({"type": "Logarithmic", "r2": r2,
            "formula": f"y = {slope:.3f}·ln(x) + {intercept:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    # Power
    try:
        xp = np.where(x > 0, x, 1e-10)
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), np.log(yp))
        r2 = r**2
        results.append({"type": "Power", "r2": r2,
            "formula": f"y = e^{intercept:.3f} · x^{slope:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    # Exponential
    try:
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(x, np.log(yp))
        r2 = r**2
        results.append({"type": "Exponential", "r2": r2,
            "formula": f"y = e^({intercept:.3f} + {slope:.3f}·x)",
            "confidence": round(r2 * 100, 1)})
    except: pass

    # Polynomial (degree 2)
    try:
        coeffs = np.polyfit(x, y, 2)
        yp = np.polyval(coeffs, x)
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results.append({"type": "Polynomial (deg 2)", "r2": r2,
            "formula": f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    if not results:
        return {"type": "unknown", "r2": 0, "formula": "n/a", "confidence": 0}

    best = max(results, key=lambda d: d["r2"])
    best["all_types"] = results
    return best

def make_chart_b64(df, x_col, y_col, rel_type=None):
    """Generate the best chart for the data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#0D1B2A"
    MID = "#112233"
    TEAL = "#1B9AAA"
    NEON = "#06D6A0"
    AMBER = "#FCD34D"
    RED = "#EF4444"

    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sidx = np.argsort(x)
    xs, ys = x[sidx], y[sidx]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(MID)

    ax.scatter(xs, ys, color=TEAL, s=40, alpha=0.8, zorder=5, label="Data points")

    # Overlay best fit curve
    if rel_type and rel_type.get("type") not in ("unknown", None):
        try:
            from scipy import stats
            xline = np.linspace(xs.min(), xs.max(), 200)
            t = rel_type["type"]
            if t == "Linear":
                slope, intercept, *_ = stats.linregress(xs, ys)
                yline = slope * xline + intercept
            elif t == "Logarithmic":
                xp = np.where(xs > 0, xs, 1e-10)
                slope, intercept, *_ = stats.linregress(np.log(xp), ys)
                xline2 = np.where(xline > 0, xline, 1e-10)
                yline = slope * np.log(xline2) + intercept
            elif t == "Power":
                xp = np.where(xs > 0, xs, 1e-10)
                yp = np.where(ys > 0, ys, 1e-10)
                slope, intercept, *_ = stats.linregress(np.log(xp), np.log(yp))
                xline2 = np.where(xline > 0, xline, 1e-10)
                yline = np.exp(intercept) * xline2**slope
            elif t == "Exponential":
                yp = np.where(ys > 0, ys, 1e-10)
                slope, intercept, *_ = stats.linregress(xs, np.log(yp))
                yline = np.exp(intercept + slope * xline)
            elif t.startswith("Polynomial"):
                coeffs = np.polyfit(xs, ys, 2)
                yline = np.polyval(coeffs, xline)
            else:
                yline = None

            if yline is not None:
                ax.plot(xline, yline, color=NEON, lw=2.5, label=f"{t} fit", zorder=4)
        except:
            pass

    ax.set_xlabel(x_col, color="#aaa", fontsize=11)
    ax.set_ylabel(y_col, color="#aaa", fontsize=11)
    ax.set_title(f"{x_col}  →  {y_col}", color="#0a0a0a", fontsize=13, pad=14)
    ax.tick_params(colors="#888")
    for s in ax.spines.values():
        s.set_edgecolor("#1e3a5f")
    ax.legend(facecolor=MID, labelcolor="#0a0a0a", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_heatmap_b64(df):
    """Correlation heatmap for multi-variable datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    BG = "#0D1B2A"
    MID = "#112233"

    corr = df.select_dtypes(include=[np.number]).corr()
    n = len(corr)
    if n < 2:
        return ""

    fig, ax = plt.subplots(figsize=(max(6, n*1.2), max(5, n)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(MID)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#0a0a0a", fontsize=9)
    ax.set_yticklabels(corr.columns, color="#0a0a0a", fontsize=9)
    ax.set_title("Correlation Matrix", color="#0a0a0a", fontsize=13, pad=14)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "black" if abs(val) > 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def call_claude_insight(x_col, y_col, rel_type, r2, sector, user_question, n_rows):
    """Call Anthropic Claude to generate expert correlation insight."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return fallback_insight(x_col, y_col, rel_type, r2)

        client = anthropic.Anthropic(api_key=api_key)

        confidence_label = "strong" if r2 > 0.8 else "moderate" if r2 > 0.5 else "weak"
        prompt = f"""You are a world-class data scientist and expert in ALL scientific domains: physics, biology, economics, psychology, sociology, medicine, engineering, finance, climate science, and more.

A user uploaded a dataset with {n_rows} rows.
- X variable: "{x_col}"
- Y variable: "{y_col}"
- Best relationship type detected: {rel_type.get('type', 'unknown')}
- Hypothetical formula: {rel_type.get('formula', 'n/a')}
- Confidence (R²): {round(r2*100, 1)}% ({confidence_label} correlation)
- User's sector: {sector if sector else 'not specified'}
- User's question: {user_question if user_question else 'not specified'}

Write a compelling, expert insight explaining:
1. WHY these two variables might be correlated in the real world (draw from any relevant field of knowledge)
2. What the relationship TYPE tells us (is it linear? exponential? what does that shape mean?)
3. What could be CAUSING this correlation (direct cause? indirect? confounding variable?)
4. One surprising or non-obvious interpretation that most people would miss
5. A practical action the user could take based on this correlation

Be specific, bold, and fascinating. Use both English and Italian (alternate by paragraph).
Keep total response under 280 words. No bullet points — flowing prose only."""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return fallback_insight(x_col, y_col, rel_type, r2)

def fallback_insight(x_col, y_col, rel_type, r2):
    """Rule-based insight when Claude is unavailable."""
    t = rel_type.get("type", "unknown")
    conf = round(r2 * 100, 1)
    strength = "strong" if r2 > 0.8 else "moderate" if r2 > 0.5 else "weak"

    lines = [
        f"A {strength} {t.lower()} correlation ({conf}%) was detected between **{x_col}** and **{y_col}**.",
        "",
        f"Una correlazione {strength} di tipo {t.lower()} ({conf}%) è stata rilevata tra **{x_col}** e **{y_col}**.",
        "",
        f"This means that as {x_col} changes, {y_col} tends to follow a {t.lower()} pattern. "
        f"The hypothetical formula {rel_type.get('formula','n/a')} could serve as a starting model, "
        f"but remember: correlation does not imply causation — a third hidden variable may be at play.",
        "",
        f"Questo suggerisce che al variare di {x_col}, il valore di {y_col} segue un andamento {t.lower()}. "
        f"Aggiungi la tua API key Anthropic nelle variabili d'ambiente su Render per sbloccare insight AI completi."
    ]
    return "\n".join(lines)

# ─────────────────────── HTML PAGE ───────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Correlation Discovery Engine</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<style>
:root{--bg:#ffffff;--mid:#f5f5f7;--card:#ffffff;--ink:#0a0a0a;--ink2:#3a3a3a;--muted:#6b6b6b;--line:#e5e5ea;--purple:#6b3fa0;--purple-dark:#4b2575;--purple-soft:#efe7fa;--accent:#6b3fa0;--ok:#1f7a3a;--warn:#a15c00;--red:#b00020;--white:#ffffff;--black:#000000;--silver:#6b6b6b;--teal:#6b3fa0;--neon:#6b3fa0;--amber:#a15c00}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:'Inter','Segoe UI',-apple-system,Helvetica,Arial,sans-serif;min-height:100vh;-webkit-font-smoothing:antialiased}
header{background:var(--white);border-bottom:1px solid var(--line);padding:18px 32px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:1.5rem;letter-spacing:3px;color:var(--ink)}
header .badge{font-size:.72rem;color:var(--neon);border:1px solid var(--neon);border-radius:20px;padding:3px 12px}
.container{max-width:1100px;margin:0 auto;padding:28px 20px}

/* Upload */
.upload-zone{border:2px dashed var(--teal);border-radius:14px;padding:44px 32px;text-align:center;background:var(--card);margin-bottom:24px;transition:.2s;cursor:pointer}
.upload-zone.dragover{border-color:var(--purple-dark);background:var(--purple-soft)}
.upload-zone h2{color:var(--ink);margin-bottom:6px;font-size:1.3rem;font-weight:800}
.upload-zone p{color:var(--muted);font-size:.88rem;margin-top:6px}
#fi{display:none}
.upload-label{display:inline-block;margin-top:16px;background:var(--purple);color:var(--white);padding:11px 30px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;transition:.2s;letter-spacing:.3px}
.upload-label:hover{background:var(--purple-dark)}
#fn{margin-top:12px;color:var(--purple);font-weight:700;min-height:22px;font-size:.9rem}

/* Sample data */
.sample-box{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 24px;margin-bottom:24px}
.sample-box h3{color:var(--teal);font-size:.88rem;letter-spacing:1px;margin-bottom:4px;font-weight:700}
.sample-box p{color:var(--muted);font-size:.78rem;margin-bottom:14px}
.sample-row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.sample-select{background:var(--mid);color:var(--ink);border:1px solid #1e3a5f;border-radius:8px;padding:9px 12px;font-size:.85rem;flex:1;min-width:200px;max-width:420px}
.sample-btn{background:var(--teal);color:var(--bg);border:none;padding:9px 22px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.82rem;letter-spacing:.5px;transition:.2s;white-space:nowrap}
.sample-btn:hover{background:var(--neon)}

/* Preview */
.preview-box{display:none;background:var(--mid);border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:20px;overflow-x:auto}
.preview-box h4{color:var(--teal);font-size:.78rem;letter-spacing:1px;margin-bottom:10px}
.preview-table{border-collapse:collapse;font-size:.75rem;width:100%}
.preview-table th{background:var(--mid);color:var(--neon);padding:5px 10px;text-align:left;border-bottom:1px solid var(--line)}
.preview-table td{color:var(--muted);padding:4px 10px;border-bottom:1px solid #162840}
.preview-meta{font-size:.72rem;color:var(--muted);margin-top:8px;opacity:.7}

/* Context Wizard */
.wizard-box{display:none;background:var(--card);border:1px solid var(--teal);border-radius:14px;padding:24px;margin-bottom:20px}
.wizard-box h3{color:var(--teal);margin-bottom:18px;letter-spacing:1px;font-size:1rem}
.wizard-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.sector-btn{background:var(--mid);border:1px solid #1e3a5f;border-radius:10px;padding:12px;cursor:pointer;color:var(--muted);font-size:.85rem;transition:.2s;text-align:center}
.sector-btn:hover,.sector-btn.active{border-color:var(--neon);color:var(--neon);background:#0d1f30}
.wizard-input{width:100%;background:var(--mid);color:var(--ink);border:1px solid #1e3a5f;border-radius:8px;padding:10px 14px;font-size:.88rem;margin-bottom:12px;transition:.2s}
.wizard-input:focus{outline:none;border-color:var(--teal)}
.wizard-label{color:var(--muted);font-size:.8rem;margin-bottom:6px;display:block;font-weight:600;letter-spacing:.5px}

/* Configure columns */
.col-select{display:none;background:var(--white);border-radius:14px;padding:24px;margin-bottom:20px;border:1px solid var(--purple)}
.col-select h3{color:var(--ink);margin-bottom:18px;letter-spacing:1px;font-size:1rem}
.col-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:6px;min-width:150px}
.col-group > label{color:var(--muted);font-size:.8rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase}
select{background:var(--mid);color:var(--ink);border:1px solid #1e3a5f;border-radius:8px;padding:9px 12px;font-size:.88rem;width:100%;transition:.2s}
select:focus{outline:none;border-color:var(--purple);box-shadow:0 0 0 3px var(--purple-soft)}
.hint{font-size:.72rem;color:var(--muted);opacity:.7;margin-top:2px}
.run-btn{background:var(--black);color:var(--white);border:none;padding:13px 38px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;margin-top:4px;letter-spacing:.5px;transition:.2s}
.run-btn:hover{background:var(--purple)}
.run-btn:disabled{opacity:.5;cursor:not-allowed}

/* Spinner */
.spinner{display:none;text-align:center;padding:36px;color:var(--purple);font-size:1rem}
.spinner::after{content:'';display:inline-block;width:20px;height:20px;border:3px solid var(--purple);border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;margin-left:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{background:#fdecef;border:1px solid var(--red);border-radius:10px;padding:14px 16px;color:var(--red);margin-bottom:16px;font-size:.88rem}

/* Results */
.results{display:none}
.result-hero{background:linear-gradient(135deg,#0d4f5c,#0a3d47);border-radius:14px;padding:28px;margin-bottom:20px;border:1px solid var(--teal)}
.result-hero h2{font-size:.75rem;letter-spacing:3px;color:rgba(255,255,255,.5);margin-bottom:14px;text-transform:uppercase}
.correlation-type{font-size:1.6rem;font-weight:700;color:var(--neon);margin-bottom:8px}
.correlation-formula{font-family:monospace;font-size:1rem;color:var(--amber);margin-bottom:14px;word-break:break-all}
.confidence-bar-wrap{height:10px;background:var(--mid);border-radius:5px;overflow:hidden;margin-bottom:6px;max-width:400px}
.confidence-bar{height:100%;background:linear-gradient(90deg,var(--purple),var(--purple-dark));border-radius:5px;transition:width 1s ease}
.confidence-label{font-size:.82rem;color:var(--muted)}
.result-actions{display:flex;gap:10px;margin-top:18px;flex-wrap:wrap}
.action-btn{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);color:var(--ink);font-size:.78rem;font-weight:700;padding:6px 16px;border-radius:20px;cursor:pointer;transition:.2s;letter-spacing:.4px}
.action-btn:hover{background:rgba(255,255,255,.18)}

/* Chart + Insight panels */
.panels-row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}
@media(max-width:700px){.panels-row{grid-template-columns:1fr}}
.panel-box{background:var(--card);border-radius:14px;padding:22px;border:1px solid #1e3a5f}
.panel-box h3{color:var(--teal);font-size:.88rem;letter-spacing:1px;margin-bottom:14px;font-weight:700}
.panel-box img{width:100%;border-radius:8px}
.insight-text{font-size:.83rem;color:var(--muted);line-height:1.8;white-space:pre-wrap}
.insight-loading{color:var(--teal);font-size:.85rem;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Heatmap */
.heatmap-box{background:var(--card);border-radius:14px;padding:22px;border:1px solid #1e3a5f;margin-bottom:20px}
.heatmap-box h3{color:var(--teal);font-size:.88rem;letter-spacing:1px;margin-bottom:14px;font-weight:700}
.heatmap-box img{width:100%;border-radius:8px}

/* Top correlations table */
.top-corr-box{background:var(--card);border-radius:14px;padding:22px;border:1px solid #1e3a5f;margin-bottom:20px}
.top-corr-box h3{color:var(--teal);font-size:.88rem;letter-spacing:1px;margin-bottom:14px;font-weight:700}
.corr-row{display:flex;align-items:center;padding:8px 0;border-bottom:1px solid var(--line);gap:12px}
.corr-pair{font-family:monospace;color:var(--ink);font-size:.82rem;min-width:200px}
.corr-bar-wrap{flex:1;height:8px;background:var(--purple-soft);border-radius:4px;overflow:hidden}
.corr-bar{height:100%;border-radius:4px}
.corr-bar.pos{background:linear-gradient(90deg,var(--purple),var(--purple-dark))}
.corr-bar.neg{background:linear-gradient(90deg,var(--red),#ff8c00)}
.corr-val{color:var(--amber);font-size:.82rem;min-width:60px;text-align:right;font-family:monospace;font-weight:700}

footer{text-align:center;padding:24px;color:var(--muted);font-size:.78rem;border-top:1px solid var(--line);margin-top:40px}
</style>
</head>
<body>
<header>
  <h1>⚡ CORRELATION DISCOVERY ENGINE</h1>
  <span class="badge">Powered by Claude AI</span>
</header>
<div class="container">

  <!-- Sample data -->
  <div class="sample-box">
    <h3>📊 NO FILE? EXPLORE WITH SAMPLE DATA / PROVA CON UN ESEMPIO</h3>
    <p>Pick a ready-made dataset and discover hidden correlations instantly. / Scegli un dataset e scopri correlazioni nascoste.</p>
    <div class="sample-row">
      <select class="sample-select" id="sampleSelect">
        <option value="">— Choose a sample / Scegli un esempio —</option>
        <option value="energy">⚡ Energy consumption vs cost</option>
        <option value="realestate">🏠 Real estate: size, location, price</option>
        <option value="health">🩺 Health: sleep, stress, performance</option>
        <option value="marketing">📈 Marketing: spend vs conversions</option>
      </select>
      <button class="sample-btn" onclick="loadSample()">▶ Load sample</button>
    </div>
  </div>

  <!-- Upload -->
  <div class="upload-zone" id="dropZone">
    <h2>📄 Drop your CSV file here / Trascina il tuo CSV qui</h2>
    <p>or click below / oppure clicca sotto</p>
    <input type="file" id="fi" accept=".csv">
    <label for="fi" class="upload-label">📁 Choose File / Scegli file</label>
    <p id="fn"></p>
  </div>

  <!-- Preview -->
  <div class="preview-box" id="previewBox">
    <h4>📊 FILE PREVIEW / ANTEPRIMA</h4>
    <div id="previewTable"></div>
    <div class="preview-meta" id="previewMeta"></div>
  </div>

  <!-- Context Wizard -->
  <div class="wizard-box" id="wizardBox">
    <h3>🌍 GIVE CONTEXT — GET BETTER INSIGHTS / DAI CONTESTO — OTTIENI INSIGHT MIGLIORI</h3>
    <label class="wizard-label">1. What is your sector? / In quale settore lavori?</label>
    <div class="wizard-grid" id="sectorGrid">
      <div class="sector-btn" onclick="selectSector(this,'Finance')">💰 Finance / Finanza</div>
      <div class="sector-btn" onclick="selectSector(this,'Health')">🩺 Health / Salute</div>
      <div class="sector-btn" onclick="selectSector(this,'Energy')">⚡ Energy / Energia</div>
      <div class="sector-btn" onclick="selectSector(this,'Marketing')">📈 Marketing</div>
      <div class="sector-btn" onclick="selectSector(this,'Science')">🔬 Science / Scienza</div>
      <div class="sector-btn" onclick="selectSector(this,'Real Estate')">🏠 Real Estate / Immobiliare</div>
      <div class="sector-btn" onclick="selectSector(this,'Other')">✨ Other / Altro</div>
    </div>
    <label class="wizard-label">2. What are you trying to understand? / Cosa vuoi capire? <span style="opacity:.5">(optional)</span></label>
    <input class="wizard-input" id="userQuestion" placeholder='e.g. "why do sales drop on Mondays?" / "perché le vendite calano il lunedì?"'>
    <label class="wizard-label">3. Do your data have a time dimension? / I dati hanno una dimensione temporale?</label>
    <div style="display:flex;gap:10px;margin-bottom:16px">
      <div class="sector-btn" onclick="selectTime(this,'yes')" id="timeYes">📅 Yes / Sì</div>
      <div class="sector-btn" onclick="selectTime(this,'no')" id="timeNo">🔢 No</div>
    </div>
  </div>

  <!-- Configure -->
  <div class="col-select" id="colSel">
    <h3>⚙ CONFIGURE ANALYSIS / CONFIGURA L'ANALISI</h3>
    <div class="col-row">
      <div class="col-group">
        <label>🎯 Target Y</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The variable you want to explain / La variabile da spiegare</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>📈 Variables X</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Cmd/Ctrl to select multiple</span>
      </div>
    </div>
    <button class="run-btn" id="runBtn" onclick="run()">🔍 DISCOVER CORRELATIONS / SCOPRI CORRELAZIONI</button>
  </div>

  <div class="spinner" id="spin">Analyzing correlations… / Analisi in corso…</div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- Results -->
  <div class="results" id="res">

    <div class="result-hero">
      <h2>BEST HYPOTHETICAL CORRELATION / MIGLIOR CORRELAZIONE IPOTETICA</h2>
      <div class="correlation-type" id="corrType"></div>
      <div class="correlation-formula" id="corrFormula"></div>
      <div class="confidence-bar-wrap"><div class="confidence-bar" id="confBar" style="width:0%"></div></div>
      <div class="confidence-label" id="confLabel"></div>
      <div class="result-actions">
        <button class="action-btn" onclick="exportResults()">⬇ Download results CSV</button>
      </div>
    </div>

    <div class="panels-row">
      <div class="panel-box">
        <h3>📊 CORRELATION CHART</h3>
        <div id="chartWrap"><p style="color:var(--muted);font-size:.85rem">Chart loading…</p></div>
      </div>
      <div class="panel-box">
        <h3>🧠 AI INSIGHT — Expert Analysis</h3>
        <div class="insight-text" id="insightText"><span class="insight-loading">Claude is thinking… / Claude sta analizzando…</span></div>
      </div>
    </div>

    <div class="heatmap-box" id="heatmapBox" style="display:none">
      <h3>🔥 CORRELATION MATRIX / MATRICE DI CORRELAZIONE</h3>
      <img id="heatmapImg" src="" alt="Correlation heatmap">
    </div>

    <div class="top-corr-box" id="topCorrBox" style="display:none">
      <h3>🏆 TOP CORRELATIONS FOUND / TOP CORRELAZIONI TROVATE</h3>
      <div id="topCorrList"></div>
    </div>

  </div>

</div>
<footer>Correlation Discovery Engine — undoubtedly created by surely not A.G.</footer>

<script>
var csv = null;
var allCols = [];
var selectedSector = '';
var selectedTime = '';
var lastResults = null;

var SAMPLES = {
  energy: "kwh,cost_eur,temp_c,hour\n100,18,22,8\n200,36,25,12\n350,63,30,15\n500,90,28,18\n750,135,20,20\n1000,180,18,22\n150,27,24,10\n420,75,31,16\n600,108,26,19\n800,144,19,21",
  realestate: "sqm,distance_center_km,floor,price_eur\n50,1,2,200000\n80,2,4,280000\n60,0.5,1,250000\n100,5,3,220000\n120,3,5,350000\n70,1.5,3,260000\n90,0.8,6,320000\n110,4,2,230000\n75,1.2,3,270000\n130,2.5,7,390000",
  health: "sleep_hours,stress_score,performance_score,caffeine_cups\n8,2,90,1\n6,7,65,3\n7,5,75,2\n5,9,50,4\n9,1,95,1\n6.5,6,70,3\n7.5,3,85,2\n4,10,40,5\n8.5,2,92,1\n7,4,78,2",
  marketing: "ad_spend_eur,impressions,clicks,conversions,revenue_eur\n1000,50000,2500,125,3750\n2000,95000,4750,237,7125\n500,22000,1100,55,1650\n3000,140000,7000,350,10500\n1500,70000,3500,175,5250\n2500,115000,5750,287,8625\n800,37000,1850,92,2775\n4000,185000,9250,462,13875"
};

function loadSample() {
  var s = document.getElementById('sampleSelect').value;
  if (!s) { alert('Select a sample first!'); return; }
  var raw = SAMPLES[s];
  Papa.parse(raw, {
    header: true, skipEmptyLines: true, dynamicTyping: false,
    complete: function(result) {
      csv = Papa.unparse(result.data);
      allCols = result.meta.fields || [];
      document.getElementById('fn').textContent = '✅ Sample loaded: ' + s;
      showPreview(result);
      parseCols();
      document.getElementById('wizardBox').style.display = 'block';
    }
  });
}

var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');

fi.addEventListener('change', function() {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});
dz.addEventListener('dragover', function(e){ e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function(){ dz.classList.remove('dragover'); });
dz.addEventListener('drop', function(e){
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(f) {
  if (!f) return;
  document.getElementById('fn').textContent = '✅ ' + f.name;
  Papa.parse(f, {
    header: true, skipEmptyLines: true, dynamicTyping: false,
    complete: function(result) {
      if (!result.data || result.data.length === 0) {
        document.getElementById('fn').textContent = '❌ Could not parse file.';
        return;
      }
      csv = Papa.unparse(result.data);
      allCols = result.meta.fields || [];
      showPreview(result);
      parseCols();
      document.getElementById('wizardBox').style.display = 'block';
    },
    error: function(err) {
      document.getElementById('fn').textContent = '❌ Parse error: ' + err.message;
    }
  });
}

function showPreview(result) {
  var box = document.getElementById('previewBox');
  var wrap = document.getElementById('previewTable');
  var meta = document.getElementById('previewMeta');
  var cols = result.meta.fields || [];
  var rows = result.data.slice(0, 4);
  var html = '<table class="preview-table"><thead><tr>';
  cols.forEach(function(c){ html += '<th>' + escHtml(c) + '</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(function(r){
    html += '<tr>';
    cols.forEach(function(c){ html += '<td>' + escHtml(String(r[c] !== undefined ? r[c] : '')) + '</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
  meta.textContent = result.data.length + ' rows · ' + cols.length + ' columns detected';
  box.style.display = 'block';
}

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function parseCols() {
  var yHints = ['y','Y','target','output','result','price','revenue','cost','score','value'];
  var guessedY = allCols.length - 1;
  for (var i = 0; i < allCols.length; i++) {
    if (yHints.indexOf(allCols[i]) !== -1) { guessedY = i; break; }
  }
  var yE = document.getElementById('yCol');
  yE.innerHTML = '';
  allCols.forEach(function(col, i) {
    var opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    if (i === guessedY) opt.selected = true;
    yE.appendChild(opt);
  });
  syncXcols();
  document.getElementById('colSel').style.display = 'block';
}

function syncXcols() {
  var yc = document.getElementById('yCol').value;
  var xE = document.getElementById('xCols');
  var prev = Array.from(xE.selectedOptions).map(function(o){ return o.value; });
  xE.innerHTML = '';
  allCols.forEach(function(col) {
    if (col === yc) return;
    var opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    opt.selected = (prev.length === 0 || prev.indexOf(col) !== -1);
    xE.appendChild(opt);
  });
}

function selectSector(el, val) {
  document.querySelectorAll('#sectorGrid .sector-btn').forEach(function(b){ b.classList.remove('active'); });
  el.classList.add('active');
  selectedSector = val;
}

function selectTime(el, val) {
  document.getElementById('timeYes').classList.remove('active');
  document.getElementById('timeNo').classList.remove('active');
  el.classList.add('active');
  selectedTime = val;
}

async function run() {
  if (!csv) { alert('Upload a CSV file first!'); return; }
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  if (xc.length === 0) { alert('Select at least one X variable.'); return; }

  document.getElementById('spin').style.display = 'block';
  document.getElementById('res').style.display = 'none';
  document.getElementById('err').style.display = 'none';
  document.getElementById('runBtn').disabled = true;

  try {
    var resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        csv: csv,
        y_col: yc,
        x_cols: xc,
        sector: selectedSector,
        user_question: document.getElementById('userQuestion').value,
        has_time: selectedTime === 'yes'
      })
    });
    var d = await resp.json();
    if (!d.success) throw new Error(d.error);
    lastResults = d;
    showResults(d, yc, xc);
  } catch(e) {
    document.getElementById('err').style.display = 'block';
    document.getElementById('err').textContent = '❌ ' + e.message;
  } finally {
    document.getElementById('spin').style.display = 'none';
    document.getElementById('runBtn').disabled = false;
  }
}

function showResults(d, yc, xc) {
  document.getElementById('res').style.display = 'block';

  var best = d.best_correlation;
  document.getElementById('corrType').textContent = best.type + ' correlation';
  document.getElementById('corrFormula').textContent = '⚡ ' + best.formula;
  var conf = best.confidence || 0;
  document.getElementById('confBar').style.width = conf + '%';
  document.getElementById('confLabel').textContent =
    'Confidence (R²): ' + conf + '% — ' + (conf > 80 ? '🟢 Strong' : conf > 50 ? '🟡 Moderate' : '🔴 Weak') +
    ' | Between: ' + xc[0] + ' → ' + yc;

  // Chart
  var cw = document.getElementById('chartWrap');
  if (d.chart_b64) {
    cw.innerHTML = '<img src="data:image/png;base64,' + d.chart_b64 + '" alt="chart">';
  } else {
    cw.innerHTML = '<p style="color:var(--muted)">Chart not available.</p>';
  }

  // Insight
  document.getElementById('insightText').textContent = d.insight || 'No insight available.';

  // Heatmap
  if (d.heatmap_b64 && xc.length > 1) {
    document.getElementById('heatmapBox').style.display = 'block';
    document.getElementById('heatmapImg').src = 'data:image/png;base64,' + d.heatmap_b64;
  }

  // Top correlations
  if (d.top_correlations && d.top_correlations.length > 0) {
    document.getElementById('topCorrBox').style.display = 'block';
    var html = '';
    var max = Math.abs(d.top_correlations[0].value);
    d.top_correlations.forEach(function(c) {
      var pct = Math.min(100, Math.abs(c.value) / (max || 1) * 100).toFixed(1);
      var cls = c.value >= 0 ? 'pos' : 'neg';
      var sign = c.value >= 0 ? '+' : '';
      html += '<div class="corr-row">' +
        '<span class="corr-pair">' + c.pair + '</span>' +
        '<div class="corr-bar-wrap"><div class="corr-bar ' + cls + '" style="width:' + pct + '%"></div></div>' +
        '<span class="corr-val">' + sign + c.value.toFixed(3) + '</span>' +
        '</div>';
    });
    document.getElementById('topCorrList').innerHTML = html;
  }
}

function exportResults() {
  if (!lastResults) return;
  var rows = [['pair','correlation','type','formula','confidence_pct']];
  (lastResults.top_correlations || []).forEach(function(c) {
    rows.push(['"'+c.pair+'"', c.value, lastResults.best_correlation.type,
               '"'+lastResults.best_correlation.formula+'"',
               lastResults.best_correlation.confidence]);
  });
  var out = rows.map(function(r){ return r.join(','); }).join('\n');
  var blob = new Blob([out], {type:'text/csv'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = 'correlation_results.csv';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""


# ─────────────────────── FLASK APP ───────────────────────

def create_app():
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    @app.route('/')
    def home():
        return Response(HTML_PAGE, mimetype='text/html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'version': '5.0-correlation'})

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        try:
            body = request.get_json()
            df = pd.read_csv(io.StringIO(body['csv']))
            y_col = body['y_col']
            x_cols = body['x_cols']
            sector = body.get('sector', '')
            user_question = body.get('user_question', '')

            # Use first X col as primary for chart + relationship
            primary_x = x_cols[0]
            x_data = df[primary_x].astype(float).values
            y_data = df[y_col].astype(float).values

            # Detect relationship
            rel = detect_relationship_type(x_data, y_data)

            # Chart
            chart_b64 = make_chart_b64(df, primary_x, y_col, rel)

            # Heatmap (if multiple cols)
            heatmap_b64 = ""
            if len(x_cols) > 1:
                cols_for_heatmap = x_cols + [y_col]
                heatmap_b64 = make_heatmap_b64(df[cols_for_heatmap])

            # Top correlations between all X and Y
            top_correlations = []
            for xc in x_cols:
                try:
                    val = float(df[[xc, y_col]].corr().iloc[0, 1])
                    if not np.isnan(val):
                        top_correlations.append({"pair": f"{xc} → {y_col}", "value": round(val, 4)})
                except:
                    pass
            # Also between X cols themselves
            for i in range(len(x_cols)):
                for j in range(i+1, len(x_cols)):
                    try:
                        val = float(df[[x_cols[i], x_cols[j]]].corr().iloc[0, 1])
                        if not np.isnan(val):
                            top_correlations.append({"pair": f"{x_cols[i]} ↔ {x_cols[j]}", "value": round(val, 4)})
                    except:
                        pass
            top_correlations.sort(key=lambda d: abs(d["value"]), reverse=True)
            top_correlations = top_correlations[:10]

            # AI Insight
            insight = call_claude_insight(
                primary_x, y_col, rel, rel.get("r2", 0),
                sector, user_question, len(df)
            )

            return jsonify({
                'success': True,
                'best_correlation': {
                    'type': rel.get('type', 'Unknown'),
                    'formula': rel.get('formula', 'n/a'),
                    'confidence': rel.get('confidence', 0),
                    'r2': rel.get('r2', 0)
                },
                'chart_b64': chart_b64,
                'heatmap_b64': heatmap_b64,
                'top_correlations': top_correlations,
                'insight': insight
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    return app


app = create_app()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--host', default='0.0.0.0')
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
