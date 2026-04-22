#!/usr/bin/env python3
"""
SumUp Insights — Correlation Discovery Engine powered by Claude AI
"""
import numpy as np
import pandas as pd
import base64
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────── MATH CORE ───────────────────────

def detect_relationship_type(x, y):
    from scipy import stats
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"type": "Insufficient data", "r2": 0, "formula": "n/a", "confidence": 0}

    results = []

    try:
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r**2
        sign = "+" if intercept >= 0 else "-"
        results.append({"type": "Linear", "r2": r2,
            "formula": f"y = {slope:.3f}x {sign} {abs(intercept):.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    try:
        xp = np.where(x > 0, x, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), y)
        r2 = r**2
        results.append({"type": "Logarithmic", "r2": r2,
            "formula": f"y = {slope:.3f} · ln(x) + {intercept:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    try:
        xp = np.where(x > 0, x, 1e-10)
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), np.log(yp))
        r2 = r**2
        results.append({"type": "Power Law", "r2": r2,
            "formula": f"y = {np.exp(intercept):.3f} · x^{slope:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    try:
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(x, np.log(yp))
        r2 = r**2
        results.append({"type": "Exponential", "r2": r2,
            "formula": f"y = e^({intercept:.3f} + {slope:.3f}·x)",
            "confidence": round(r2 * 100, 1)})
    except: pass

    try:
        coeffs = np.polyfit(x, y, 2)
        yp = np.polyval(coeffs, x)
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0
        results.append({"type": "Polynomial", "r2": r2,
            "formula": f"y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}",
            "confidence": round(r2 * 100, 1)})
    except: pass

    if not results:
        return {"type": "No pattern found", "r2": 0, "formula": "n/a", "confidence": 0}

    best = max(results, key=lambda d: d["r2"])
    best["all_types"] = results
    return best


def make_chart_b64(df, x_col, y_col, rel_type=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    BG = "#FFFFFF"
    GRID = "#F3F4F6"
    PRIMARY = "#7B2FBE"
    SECONDARY = "#A855F7"
    DOT = "#1F2937"
    TEXT = "#374151"
    LIGHT = "#9CA3AF"

    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sidx = np.argsort(x)
    xs, ys = x[sidx], y[sidx]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.scatter(xs, ys, color=PRIMARY, s=50, alpha=0.75, zorder=5, label="Data points")

    if rel_type and rel_type.get("type") not in ("No pattern found", "Insufficient data", None):
        try:
            xline = np.linspace(xs.min(), xs.max(), 300)
            t = rel_type["type"]
            if t == "Linear":
                slope, intercept, *_ = stats.linregress(xs, ys)
                yline = slope * xline + intercept
            elif t == "Logarithmic":
                xp = np.where(xs > 0, xs, 1e-10)
                slope, intercept, *_ = stats.linregress(np.log(xp), ys)
                xline2 = np.where(xline > 0, xline, 1e-10)
                yline = slope * np.log(xline2) + intercept
            elif t == "Power Law":
                xp = np.where(xs > 0, xs, 1e-10)
                yp = np.where(ys > 0, ys, 1e-10)
                slope, intercept, *_ = stats.linregress(np.log(xp), np.log(yp))
                xline2 = np.where(xline > 0, xline, 1e-10)
                yline = np.exp(intercept) * xline2**slope
            elif t == "Exponential":
                yp = np.where(ys > 0, ys, 1e-10)
                slope, intercept, *_ = stats.linregress(xs, np.log(yp))
                yline = np.exp(intercept + slope * xline)
            elif t == "Polynomial":
                coeffs = np.polyfit(xs, ys, 2)
                yline = np.polyval(coeffs, xline)
            else:
                yline = None

            if yline is not None:
                ax.plot(xline, yline, color=SECONDARY, lw=2.5,
                        label=f"{t} trend", zorder=4, linestyle="--")
        except:
            pass

    ax.set_xlabel(x_col, color=TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel(y_col, color=TEXT, fontsize=11, labelpad=8)
    ax.set_title(f"{x_col}  →  {y_col}", color=DOT, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor(GRID)
    ax.spines["bottom"].set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=9, framealpha=1,
              edgecolor=GRID)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_heatmap_b64(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#FFFFFF"
    TEXT = "#374151"
    GRID = "#F3F4F6"

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    n = len(corr)
    if n < 2:
        return ""

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 0.9)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=8, colors=TEXT)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", color=TEXT, fontsize=9)
    ax.set_yticklabels(corr.columns, color=TEXT, fontsize=9)
    ax.set_title("Correlation Matrix", color=TEXT, fontsize=12, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor(GRID)
    ax.spines["bottom"].set_edgecolor(GRID)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.65 else TEXT
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def call_claude_insight(x_col, y_col, rel_type, r2, sector, user_question, n_rows):
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return fallback_insight(x_col, y_col, rel_type, r2)

        client = anthropic.Anthropic(api_key=api_key)
        confidence_label = "strong" if r2 > 0.8 else "moderate" if r2 > 0.5 else "weak"

        prompt = f"""You are a world-class management consultant and data scientist with deep expertise across ALL domains: business, finance, economics, psychology, sociology, operations, marketing, medicine, engineering, and more.

A manager uploaded a business dataset with {n_rows} rows for analysis.
- Variable being explained (Y): "{y_col}"
- Variable used to explain it (X): "{x_col}"
- Best relationship pattern detected: {rel_type.get('type', 'unknown')}
- Hypothetical formula: {rel_type.get('formula', 'n/a')}
- Confidence (R²): {round(r2*100,1)}% — {confidence_label} correlation
- Industry/sector: {sector if sector else 'not specified'}
- Manager's question: {user_question if user_question else 'not specified'}

Write a sharp, executive-level insight in flowing prose (no bullet points). Cover:
1. What this correlation likely means in a real business context
2. Why this specific relationship TYPE (linear/exponential/etc.) matters — what does the shape tell us?
3. What could be driving this — direct cause, indirect effect, or hidden variable?
4. One non-obvious insight a junior analyst would miss
5. One concrete action management could take based on this finding

Write in a mix of English and Italian (alternate paragraphs). Be bold, specific, and compelling.
Max 260 words. No bullet points — only flowing paragraphs."""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return fallback_insight(x_col, y_col, rel_type, r2)


def fallback_insight(x_col, y_col, rel_type, r2):
    t = rel_type.get("type", "unknown")
    conf = round(r2 * 100, 1)
    strength = "strong" if r2 > 0.8 else "moderate" if r2 > 0.5 else "weak"
    return (
        f"A {strength} {t.lower()} correlation ({conf}%) was detected between "
        f"{x_col} and {y_col}.\n\n"
        f"Una correlazione {strength} di tipo {t.lower()} ({conf}%) è stata rilevata "
        f"tra {x_col} e {y_col}.\n\n"
        f"This pattern suggests that as {x_col} changes, {y_col} follows a "
        f"{t.lower()} trend. Remember: correlation does not imply causation — "
        f"always validate with your domain knowledge before acting.\n\n"
        f"To unlock full AI-powered insights, ensure the ANTHROPIC_API_KEY "
        f"environment variable is set on your server."
    )


def read_uploaded_file(file_bytes, filename):
    """Read CSV or Excel file and return a DataFrame."""
    fname = filename.lower()
    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raw = file_bytes.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(raw))


# ─────────────────────── HTML ───────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SumUp Insights</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<style>
:root{
  --white:#ffffff;
  --bg:#F9FAFB;
  --border:#E5E7EB;
  --card:#ffffff;
  --text:#111827;
  --muted:#6B7280;
  --light:#9CA3AF;
  --purple:#7B2FBE;
  --purple-light:#F3E8FF;
  --purple-mid:#A855F7;
  --green:#059669;
  --red:#DC2626;
  --amber:#D97706;
  --radius:12px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-height:100vh;font-size:15px}

/* Header */
header{background:var(--white);border-bottom:1px solid var(--border);padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{width:28px;height:28px;background:var(--purple);border-radius:6px;display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:13px}
.logo-text{font-size:1rem;font-weight:700;color:var(--text);letter-spacing:-.3px}
.logo-sub{font-size:.72rem;color:var(--muted);margin-left:6px;font-weight:400}
.header-badge{font-size:.7rem;background:var(--purple-light);color:var(--purple);border-radius:20px;padding:3px 10px;font-weight:600}

.container{max-width:1080px;margin:0 auto;padding:32px 20px}

/* How it works */
.how-section{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:24px}
.how-section h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.steps{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
.step{display:flex;gap:12px;align-items:flex-start}
.step-num{width:28px;height:28px;min-width:28px;background:var(--purple-light);color:var(--purple);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:800}
.step-body h4{font-size:.85rem;font-weight:600;color:var(--text);margin-bottom:3px}
.step-body p{font-size:.78rem;color:var(--muted);line-height:1.55}

/* Sample data */
.sample-section{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:20px 28px;margin-bottom:20px}
.sample-section h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:12px}
.sample-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.sample-select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem;flex:1;min-width:220px;max-width:380px}
.sample-select:focus{outline:none;border-color:var(--purple)}
.btn-ghost{background:var(--white);color:var(--purple);border:1px solid var(--purple);border-radius:8px;padding:8px 20px;cursor:pointer;font-weight:600;font-size:.82rem;transition:.15s}
.btn-ghost:hover{background:var(--purple-light)}

/* Upload */
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:36px;text-align:center;background:var(--white);margin-bottom:20px;transition:.2s;cursor:pointer}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--purple);background:var(--purple-light)}
.upload-zone h2{color:var(--text);font-size:1rem;font-weight:600;margin-bottom:6px}
.upload-zone p{color:var(--muted);font-size:.83rem;margin-top:4px}
#fi{display:none}
.btn-primary{display:inline-block;margin-top:14px;background:var(--purple);color:white;padding:10px 28px;border-radius:8px;cursor:pointer;font-weight:600;font-size:.9rem;transition:.15s;border:none;letter-spacing:.01em}
.btn-primary:hover{background:var(--purple-mid)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
#fn{margin-top:10px;color:var(--purple);font-weight:600;min-height:20px;font-size:.85rem}

/* Preview */
.preview-box{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;margin-bottom:20px;overflow-x:auto}
.preview-box h4{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:12px}
.preview-table{border-collapse:collapse;font-size:.78rem;width:100%}
.preview-table th{background:var(--bg);color:var(--muted);padding:6px 12px;text-align:left;border-bottom:1px solid var(--border);font-weight:600;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase}
.preview-table td{color:var(--text);padding:5px 12px;border-bottom:1px solid var(--bg)}
.preview-meta{font-size:.72rem;color:var(--light);margin-top:8px}

/* Wizard */
.wizard-box{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px}
.wizard-box h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.wizard-q{margin-bottom:16px}
.wizard-q label{display:block;font-size:.83rem;font-weight:600;color:var(--text);margin-bottom:8px}
.sector-grid{display:flex;flex-wrap:wrap;gap:8px}
.sector-btn{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:7px 14px;cursor:pointer;color:var(--muted);font-size:.8rem;transition:.15s;font-weight:500}
.sector-btn:hover{border-color:var(--purple);color:var(--purple)}
.sector-btn.active{border-color:var(--purple);color:var(--purple);background:var(--purple-light);font-weight:600}
.wizard-input{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 14px;font-size:.85rem;transition:.15s}
.wizard-input:focus{outline:none;border-color:var(--purple)}

/* Configure */
.col-select{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px}
.col-select h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.col-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:18px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:5px;min-width:150px}
.col-group > label{color:var(--text);font-size:.8rem;font-weight:600}
select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem;width:100%;transition:.15s}
select:focus{outline:none;border-color:var(--purple)}
.hint{font-size:.72rem;color:var(--light);margin-top:2px}
.val-msg{display:none;color:var(--red);font-size:.8rem;margin-top:8px;padding:8px 12px;background:#FEF2F2;border-radius:6px;border-left:3px solid var(--red)}

/* Spinner */
.spinner{display:none;text-align:center;padding:40px;color:var(--muted);font-size:.9rem}
.spinner-ring{display:inline-block;width:22px;height:22px;border:3px solid var(--border);border-top-color:var(--purple);border-radius:50%;animation:spin .7s linear infinite;margin-right:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{background:#FEF2F2;border:1px solid #FECACA;border-radius:var(--radius);padding:14px 18px;color:var(--red);margin-bottom:16px;font-size:.85rem}

/* Results */
.results{display:none}

.result-hero{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:28px 32px;margin-bottom:20px}
.result-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:10px}
.result-type{font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:6px}
.result-formula{font-family:'SF Mono',Monaco,monospace;font-size:.9rem;color:var(--purple);background:var(--purple-light);display:inline-block;padding:5px 14px;border-radius:6px;margin-bottom:16px}
.conf-wrap{max-width:380px;margin-bottom:6px}
.conf-track{height:8px;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--border)}
.conf-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--purple),var(--purple-mid));transition:width 1s ease}
.conf-label{font-size:.78rem;color:var(--muted);margin-top:5px}
.result-actions{display:flex;gap:10px;margin-top:18px;flex-wrap:wrap}
.btn-sm{background:var(--bg);border:1px solid var(--border);color:var(--text);font-size:.78rem;font-weight:600;padding:6px 16px;border-radius:8px;cursor:pointer;transition:.15s}
.btn-sm:hover{border-color:var(--purple);color:var(--purple)}

/* Panels */
.panels-row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
@media(max-width:680px){.panels-row{grid-template-columns:1fr}}
.panel-card{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:22px 24px}
.panel-card h3{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.panel-card img{width:100%;border-radius:8px;border:1px solid var(--border)}
.insight-text{font-size:.83rem;color:var(--muted);line-height:1.85;white-space:pre-wrap}
.insight-loading{color:var(--purple);font-size:.83rem;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* Heatmap */
.full-card{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:22px 24px;margin-bottom:18px}
.full-card h3{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.full-card img{width:100%;border-radius:8px;border:1px solid var(--border)}

/* Top correlations */
.corr-row{display:flex;align-items:center;padding:9px 0;border-bottom:1px solid var(--bg);gap:12px}
.corr-pair{font-size:.8rem;color:var(--text);min-width:220px;font-weight:500}
.corr-bar-wrap{flex:1;height:6px;background:var(--bg);border-radius:3px;overflow:hidden}
.corr-bar-pos{height:100%;background:var(--purple);border-radius:3px}
.corr-bar-neg{height:100%;background:var(--red);border-radius:3px}
.corr-val{font-size:.78rem;min-width:52px;text-align:right;font-weight:700}
.corr-val.pos{color:var(--purple)}
.corr-val.neg{color:var(--red)}

/* Strength pill */
.pill{display:inline-block;font-size:.68rem;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:10px;vertical-align:middle}
.pill-strong{background:#D1FAE5;color:var(--green)}
.pill-moderate{background:#FEF3C7;color:var(--amber)}
.pill-weak{background:#FEE2E2;color:var(--red)}

footer{text-align:center;padding:28px;color:var(--light);font-size:.75rem;border-top:1px solid var(--border);margin-top:16px}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">S+</div>
    <span class="logo-text">SumUp Insights<span class="logo-sub">by FormulaFinder</span></span>
  </div>
  <span class="header-badge">Powered by Claude AI</span>
</header>

<div class="container">

  <!-- HOW IT WORKS -->
  <div class="how-section">
    <h2>How it works / Come funziona</h2>
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <h4>Upload your data</h4>
          <p>Drop a CSV or Excel file. The first row must contain column names. All data must be numeric.<br><em>Carica un file CSV o Excel con la prima riga come intestazione.</em></p>
        </div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          <h4>Give context</h4>
          <p>Tell us your sector and what you want to understand. The AI uses this to tailor its analysis.<br><em>Indica il settore e la domanda di business che vuoi rispondere.</em></p>
        </div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          <h4>Choose your variables</h4>
          <p>Select which metric to explain (Y) and which factors might drive it (X).<br><em>Scegli la metrica da spiegare e i fattori potenzialmente correlati.</em></p>
        </div>
      </div>
      <div class="step">
        <div class="step-num">4</div>
        <div class="step-body">
          <h4>Discover correlations</h4>
          <p>SumUp finds the best pattern, visualises it, and gives you an expert explanation.<br><em>SumUp trova il pattern, lo visualizza e ti spiega cosa significa.</em></p>
        </div>
      </div>
    </div>
  </div>

  <!-- SAMPLE DATA -->
  <div class="sample-section">
    <h2>No file yet? Explore a sample / Nessun file? Prova un esempio</h2>
    <div class="sample-row">
      <select class="sample-select" id="sampleSelect">
        <option value="">— Select a sample dataset —</option>
        <option value="energy">Energy consumption vs cost</option>
        <option value="realestate">Real estate: size, location, price</option>
        <option value="health">Health: sleep, stress, performance</option>
        <option value="marketing">Marketing: spend vs conversions</option>
      </select>
      <button class="btn-ghost" onclick="loadSample()">Load sample</button>
    </div>
  </div>

  <!-- UPLOAD -->
  <div class="upload-zone" id="dropZone">
    <h2>Drop your file here</h2>
    <p>Supports CSV and Excel (.xlsx) &nbsp;|&nbsp; Trascina qui il tuo file CSV o Excel</p>
    <input type="file" id="fi" accept=".csv,.xlsx,.xls">
    <label for="fi" class="btn-primary">Choose file / Scegli file</label>
    <p id="fn"></p>
  </div>

  <!-- PREVIEW -->
  <div class="preview-box" id="previewBox">
    <h4>File preview / Anteprima</h4>
    <div id="previewTable"></div>
    <div class="preview-meta" id="previewMeta"></div>
  </div>

  <!-- CONTEXT WIZARD -->
  <div class="wizard-box" id="wizardBox">
    <h2>Add context for better insights / Aggiungi contesto</h2>
    <div class="wizard-q">
      <label>1. What is your sector? / In quale settore lavori?</label>
      <div class="sector-grid" id="sectorGrid">
        <div class="sector-btn" onclick="selectSector(this,'Finance')">Finance</div>
        <div class="sector-btn" onclick="selectSector(this,'Health')">Health</div>
        <div class="sector-btn" onclick="selectSector(this,'Energy')">Energy</div>
        <div class="sector-btn" onclick="selectSector(this,'Marketing')">Marketing</div>
        <div class="sector-btn" onclick="selectSector(this,'Operations')">Operations</div>
        <div class="sector-btn" onclick="selectSector(this,'Real Estate')">Real Estate</div>
        <div class="sector-btn" onclick="selectSector(this,'Retail')">Retail</div>
        <div class="sector-btn" onclick="selectSector(this,'Other')">Other</div>
      </div>
    </div>
    <div class="wizard-q">
      <label>2. What are you trying to understand? <span style="color:var(--light);font-weight:400">(optional)</span></label>
      <input class="wizard-input" id="userQuestion"
        placeholder='e.g. "Why do sales drop on Mondays?" / "Perché le vendite calano il lunedì?"'>
    </div>
  </div>

  <!-- CONFIGURE -->
  <div class="col-select" id="colSel">
    <h2>Configure analysis / Configura l'analisi</h2>
    <div class="col-row">
      <div class="col-group">
        <label>Target to explain (Y)</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The metric you want to understand</span>
      </div>
      <div class="col-group" style="flex:1;min-width:200px">
        <label>Potential drivers (X)</label>
        <select id="xCols" multiple style="height:100px"></select>
        <span class="hint">Hold Cmd/Ctrl to select multiple columns</span>
      </div>
    </div>
    <div class="val-msg" id="valMsg"></div>
    <button class="btn-primary" id="runBtn" onclick="run()">Discover correlations</button>
  </div>

  <div class="spinner" id="spin"><span class="spinner-ring"></span>Analysing your data… / Analisi in corso…</div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- RESULTS -->
  <div class="results" id="res">

    <div class="result-hero">
      <div class="result-eyebrow">Best hypothetical correlation / Miglior correlazione ipotetica</div>
      <div class="result-type" id="corrType"></div>
      <div class="result-formula" id="corrFormula"></div>
      <div class="conf-wrap">
        <div class="conf-track"><div class="conf-fill" id="confBar" style="width:0%"></div></div>
        <div class="conf-label" id="confLabel"></div>
      </div>
      <div class="result-actions">
        <button class="btn-sm" onclick="exportResults()">Download results CSV</button>
      </div>
    </div>

    <div class="panels-row">
      <div class="panel-card">
        <h3>Correlation chart</h3>
        <div id="chartWrap"></div>
      </div>
      <div class="panel-card">
        <h3>AI Expert Insight</h3>
        <div class="insight-text" id="insightText">
          <span class="insight-loading">Claude is analysing your data…</span>
        </div>
      </div>
    </div>

    <div class="full-card" id="heatmapBox" style="display:none">
      <h3>Correlation matrix / Matrice di correlazione</h3>
      <img id="heatmapImg" src="" alt="Correlation matrix">
    </div>

    <div class="full-card" id="topCorrBox" style="display:none">
      <h3>All correlations found / Tutte le correlazioni</h3>
      <div id="topCorrList"></div>
    </div>

  </div>

</div>

<footer>SumUp Insights — Data correlation engine for management &nbsp;|&nbsp; FormulaFinder SRL</footer>

<script>
var csv = null;
var allCols = [];
var selectedSector = '';
var lastResults = null;
var uploadedFilename = 'data.csv';

var SAMPLES = {
  energy: "kwh,cost_eur,temp_c,hour\n100,18,22,8\n200,36,25,12\n350,63,30,15\n500,90,28,18\n750,135,20,20\n1000,180,18,22\n150,27,24,10\n420,75,31,16\n600,108,26,19\n800,144,19,21",
  realestate: "sqm,distance_center_km,floor,price_eur\n50,1,2,200000\n80,2,4,280000\n60,0.5,1,250000\n100,5,3,220000\n120,3,5,350000\n70,1.5,3,260000\n90,0.8,6,320000\n110,4,2,230000\n75,1.2,3,270000\n130,2.5,7,390000",
  health: "sleep_hours,stress_score,performance_score,caffeine_cups\n8,2,90,1\n6,7,65,3\n7,5,75,2\n5,9,50,4\n9,1,95,1\n6.5,6,70,3\n7.5,3,85,2\n4,10,40,5\n8.5,2,92,1\n7,4,78,2",
  marketing: "ad_spend_eur,impressions,clicks,conversions,revenue_eur\n1000,50000,2500,125,3750\n2000,95000,4750,237,7125\n500,22000,1100,55,1650\n3000,140000,7000,350,10500\n1500,70000,3500,175,5250\n2500,115000,5750,287,8625\n800,37000,1850,92,2775\n4000,185000,9250,462,13875"
};

function loadSample() {
  var s = document.getElementById('sampleSelect').value;
  if (!s) { alert('Select a sample first.'); return; }
  Papa.parse(SAMPLES[s], {
    header:true, skipEmptyLines:true, dynamicTyping:false,
    complete: function(result) {
      csv = Papa.unparse(result.data);
      uploadedFilename = s + '.csv';
      allCols = result.meta.fields || [];
      document.getElementById('fn').textContent = 'Sample loaded: ' + s;
      showPreview(result);
      parseCols();
      document.getElementById('wizardBox').style.display = 'block';
    }
  });
}

var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');
fi.addEventListener('change', function(){ if(fi.files && fi.files.length>0) handleFile(fi.files[0]); });
dz.addEventListener('dragover',  function(e){ e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function(){ dz.classList.remove('dragover'); });
dz.addEventListener('drop', function(e){
  e.preventDefault(); dz.classList.remove('dragover');
  if(e.dataTransfer.files && e.dataTransfer.files.length>0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(f) {
  if (!f) return;
  uploadedFilename = f.name;
  document.getElementById('fn').textContent = f.name;
  var ext = f.name.split('.').pop().toLowerCase();
  if (ext === 'xlsx' || ext === 'xls') {
    // Send to server for parsing
    var reader = new FileReader();
    reader.onload = async function(e) {
      var b64 = btoa(String.fromCharCode.apply(null, new Uint8Array(e.target.result)));
      try {
        var resp = await fetch('/api/parse_excel', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({b64: b64, filename: f.name})
        });
        var d = await resp.json();
        if (!d.success) throw new Error(d.error);
        csv = d.csv;
        allCols = d.columns;
        showPreviewFromServer(d);
        parseCols();
        document.getElementById('wizardBox').style.display = 'block';
        document.getElementById('fn').textContent = 'Loaded: ' + f.name + ' (' + d.rows + ' rows)';
      } catch(err) {
        document.getElementById('fn').textContent = 'Error: ' + err.message;
      }
    };
    reader.readAsArrayBuffer(f);
  } else {
    Papa.parse(f, {
      header:true, skipEmptyLines:true, dynamicTyping:false,
      complete: function(result) {
        if (!result.data || result.data.length===0) {
          document.getElementById('fn').textContent = 'Could not parse file. Check format.';
          return;
        }
        csv = Papa.unparse(result.data);
        allCols = result.meta.fields || [];
        showPreview(result);
        parseCols();
        document.getElementById('wizardBox').style.display = 'block';
        document.getElementById('fn').textContent = 'Loaded: ' + f.name + ' (' + result.data.length + ' rows)';
      },
      error: function(err){ document.getElementById('fn').textContent = 'Parse error: ' + err.message; }
    });
  }
}

function showPreview(result) {
  var cols = result.meta.fields || [];
  var rows = result.data.slice(0,4);
  renderPreviewTable(cols, rows, result.data.length, cols.length);
}
function showPreviewFromServer(d) {
  var cols = d.columns;
  var rows = d.preview;
  renderPreviewTable(cols, rows, d.rows, cols.length);
}
function renderPreviewTable(cols, rows, totalRows, totalCols) {
  var box  = document.getElementById('previewBox');
  var wrap = document.getElementById('previewTable');
  var meta = document.getElementById('previewMeta');
  var html = '<table class="preview-table"><thead><tr>';
  cols.forEach(function(c){ html += '<th>'+escHtml(c)+'</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(function(r){
    html += '<tr>';
    cols.forEach(function(c){ html += '<td>'+escHtml(String(r[c]!==undefined?r[c]:''))+'</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
  meta.textContent = totalRows + ' rows · ' + totalCols + ' columns detected';
  box.style.display = 'block';
}
function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function parseCols() {
  var yHints = ['y','target','output','result','price','revenue','cost','score','value','profit','sales'];
  var guessedY = allCols.length - 1;
  for(var i=0;i<allCols.length;i++){
    if(yHints.indexOf(allCols[i].toLowerCase())!==-1){guessedY=i;break;}
  }
  var yE = document.getElementById('yCol');
  yE.innerHTML = '';
  allCols.forEach(function(col,i){
    var opt = document.createElement('option');
    opt.value=col; opt.textContent=col;
    if(i===guessedY) opt.selected=true;
    yE.appendChild(opt);
  });
  syncXcols();
  document.getElementById('colSel').style.display='block';
}
function syncXcols() {
  var yc = document.getElementById('yCol').value;
  var xE = document.getElementById('xCols');
  var prev = Array.from(xE.selectedOptions).map(function(o){return o.value;});
  xE.innerHTML='';
  allCols.forEach(function(col){
    if(col===yc) return;
    var opt=document.createElement('option');
    opt.value=col; opt.textContent=col;
    opt.selected=(prev.length===0||prev.indexOf(col)!==-1);
    xE.appendChild(opt);
  });
}
function selectSector(el,val){
  document.querySelectorAll('#sectorGrid .sector-btn').forEach(function(b){b.classList.remove('active');});
  el.classList.add('active'); selectedSector=val;
}

async function run() {
  if (!csv) { showVal('Upload a file first.'); return; }
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){return o.value;});
  if(xc.length===0){showVal('Select at least one driver (X).'); return;}
  hideVal();
  document.getElementById('spin').style.display='block';
  document.getElementById('res').style.display='none';
  document.getElementById('err').style.display='none';
  document.getElementById('runBtn').disabled=true;
  try {
    var resp = await fetch('/api/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({csv:csv,y_col:yc,x_cols:xc,
        sector:selectedSector,
        user_question:document.getElementById('userQuestion').value})
    });
    var d = await resp.json();
    if(!d.success) throw new Error(d.error);
    lastResults=d; lastResults._yc=yc; lastResults._xc=xc;
    showResults(d,yc,xc);
  } catch(e){
    document.getElementById('err').style.display='block';
    document.getElementById('err').textContent='Error: '+e.message;
  } finally {
    document.getElementById('spin').style.display='none';
    document.getElementById('runBtn').disabled=false;
  }
}

function showResults(d,yc,xc){
  document.getElementById('res').style.display='block';
  var best = d.best_correlation;
  var conf = best.confidence||0;
  var strength = conf>80?'Strong':conf>50?'Moderate':'Weak';
  var pillCls = conf>80?'pill-strong':conf>50?'pill-moderate':'pill-weak';

  document.getElementById('corrType').innerHTML =
    best.type + ' correlation' +
    '<span class="pill '+pillCls+'">'+strength+'</span>';
  document.getElementById('corrFormula').textContent = best.formula;
  document.getElementById('confBar').style.width = conf+'%';
  document.getElementById('confLabel').textContent =
    'Confidence (R²): '+conf+'% — between '+xc[0]+' and '+yc;

  var cw = document.getElementById('chartWrap');
  cw.innerHTML = d.chart_b64
    ? '<img src="data:image/png;base64,'+d.chart_b64+'" alt="chart">'
    : '<p style="color:var(--light);font-size:.83rem;padding:20px 0">Chart not available.</p>';

  document.getElementById('insightText').textContent = d.insight||'No insight available.';

  if(d.heatmap_b64 && xc.length>1){
    document.getElementById('heatmapBox').style.display='block';
    document.getElementById('heatmapImg').src='data:image/png;base64,'+d.heatmap_b64;
  } else {
    document.getElementById('heatmapBox').style.display='none';
  }

  if(d.top_correlations && d.top_correlations.length>0){
    document.getElementById('topCorrBox').style.display='block';
    var html=''; var mx=Math.abs(d.top_correlations[0].value)||1;
    d.top_correlations.forEach(function(c){
      var pct=Math.min(100,Math.abs(c.value)/mx*100).toFixed(1);
      var pos=c.value>=0;
      var sign=pos?'+':'';
      html+='<div class="corr-row">'+
        '<span class="corr-pair">'+escHtml(c.pair)+'</span>'+
        '<div class="corr-bar-wrap"><div class="'+(pos?'corr-bar-pos':'corr-bar-neg')+'" style="width:'+pct+'%"></div></div>'+
        '<span class="corr-val '+(pos?'pos':'neg')+'">'+sign+c.value.toFixed(3)+'</span>'+
        '</div>';
    });
    document.getElementById('topCorrList').innerHTML=html;
  } else {
    document.getElementById('topCorrBox').style.display='none';
  }
}

function exportResults(){
  if(!lastResults) return;
  var rows=[['pair','pearson_r','type','formula','confidence_pct']];
  (lastResults.top_correlations||[]).forEach(function(c){
    rows.push(['"'+c.pair+'"',c.value,lastResults.best_correlation.type,
               '"'+lastResults.best_correlation.formula+'"',
               lastResults.best_correlation.confidence]);
  });
  var out=rows.map(function(r){return r.join(',');}).join('\n');
  var blob=new Blob([out],{type:'text/csv'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url; a.download='sumup_insights_results.csv';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function showVal(m){var e=document.getElementById('valMsg');e.textContent=m;e.style.display='block';}
function hideVal(){document.getElementById('valMsg').style.display='none';}
</script>
</body>
</html>"""


# ─────────────────────── FLASK ───────────────────────

def create_app():
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    @app.route('/')
    def home():
        return Response(HTML_PAGE, mimetype='text/html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'version': '6.0-sumup'})

    @app.route('/api/parse_excel', methods=['POST'])
    def api_parse_excel():
        try:
            body = request.get_json()
            import base64 as b64m
            file_bytes = b64m.b64decode(body['b64'])
            filename   = body.get('filename','file.xlsx')
            df = pd.read_excel(io.BytesIO(file_bytes))
            df = df.select_dtypes(include=[np.number, 'object'])
            csv_out = df.to_csv(index=False)
            preview = df.head(4).fillna('').to_dict('records')
            return jsonify({
                'success': True,
                'csv': csv_out,
                'columns': list(df.columns),
                'rows': len(df),
                'preview': preview
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        try:
            body     = request.get_json()
            df       = pd.read_csv(io.StringIO(body['csv']))
            y_col    = body['y_col']
            x_cols   = body['x_cols']
            sector   = body.get('sector','')
            user_q   = body.get('user_question','')

            primary_x = x_cols[0]
            x_data = df[primary_x].astype(float).values
            y_data = df[y_col].astype(float).values

            rel      = detect_relationship_type(x_data, y_data)
            chart    = make_chart_b64(df, primary_x, y_col, rel)

            heatmap = ""
            if len(x_cols) > 1:
                cols_h = list(set(x_cols + [y_col]))
                heatmap = make_heatmap_b64(df[cols_h])

            top_corr = []
            for xc in x_cols:
                try:
                    v = float(df[[xc, y_col]].corr().iloc[0,1])
                    if not np.isnan(v):
                        top_corr.append({"pair": f"{xc} → {y_col}", "value": round(v,4)})
                except: pass
            for i in range(len(x_cols)):
                for j in range(i+1, len(x_cols)):
                    try:
                        v = float(df[[x_cols[i],x_cols[j]]].corr().iloc[0,1])
                        if not np.isnan(v):
                            top_corr.append({"pair": f"{x_cols[i]} ↔ {x_cols[j]}", "value": round(v,4)})
                    except: pass
            top_corr.sort(key=lambda d: abs(d["value"]), reverse=True)
            top_corr = top_corr[:12]

            insight = call_claude_insight(
                primary_x, y_col, rel, rel.get("r2",0),
                sector, user_q, len(df)
            )

            return jsonify({
                'success': True,
                'best_correlation': {
                    'type':       rel.get('type','Unknown'),
                    'formula':    rel.get('formula','n/a'),
                    'confidence': rel.get('confidence',0),
                    'r2':         rel.get('r2',0)
                },
                'chart_b64':       chart,
                'heatmap_b64':     heatmap,
                'top_correlations': top_corr,
                'insight':         insight
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
