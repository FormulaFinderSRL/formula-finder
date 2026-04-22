#!/usr/bin/env python3
"""
SumUp Insights — Correlation Discovery Engine for Call Center Management
Powered by Claude AI | FormulaFinder SRL
"""
import numpy as np
import pandas as pd
import base64
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────── MATH CORE ───────────────────────

def detect_relationship(x, y):
    """Find the best-fitting relationship between two variables.
    Returns business-friendly labels, not math jargon."""
    from scipy import stats
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"type": "insufficient", "r2": 0, "formula": "", "label": "Not enough data",
                "direction": "unknown", "strength": "none", "business_type": "Inconclusive"}

    results = []

    # Linear: y = ax + b
    try:
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r**2
        direction = "positive" if slope > 0 else "negative"
        results.append({
            "type": "linear", "r2": r2,
            "formula": f"y = {slope:.3f}x + {intercept:.3f}",
            "slope": slope, "intercept": intercept,
            "label": "Proportional" if abs(intercept) < abs(slope * np.mean(x) * 0.1) else "Linear with offset",
            "business_type": f"{'More' if slope > 0 else 'Less'} X → {'More' if slope > 0 else 'Less'} Y (proportional)",
            "direction": direction,
            "impact_per_unit": abs(slope),
            "impact_text": f"Every +1 in X {'increases' if slope > 0 else 'decreases'} Y by {abs(slope):.2f}"
        })
    except: pass

    # Logarithmic: y = a·ln(x) + b  (diminishing returns)
    try:
        xp = np.where(x > 0, x, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), y)
        r2 = r**2
        results.append({
            "type": "logarithmic", "r2": r2,
            "formula": f"y = {slope:.3f}·ln(x) + {intercept:.3f}",
            "slope": slope, "intercept": intercept,
            "label": "Diminishing returns",
            "business_type": "Big gains early, then plateau — law of diminishing returns",
            "direction": "positive" if slope > 0 else "negative",
            "impact_per_unit": abs(slope),
            "impact_text": f"Doubling X {'adds' if slope > 0 else 'removes'} ~{abs(slope * 0.693):.2f} to Y (but gains shrink over time)"
        })
    except: pass

    # Exponential: y = e^(a + bx)  (compounding/snowball)
    try:
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(x, np.log(yp))
        r2 = r**2
        results.append({
            "type": "exponential", "r2": r2,
            "formula": f"y = e^({intercept:.3f} + {slope:.3f}x)",
            "slope": slope, "intercept": intercept,
            "label": "Compounding / snowball effect",
            "business_type": f"Y {'accelerates' if slope > 0 else 'decays'} — small changes in X have {'growing' if slope > 0 else 'shrinking'} impact",
            "direction": "positive" if slope > 0 else "negative",
            "impact_per_unit": abs(slope),
            "impact_text": f"Each +1 in X {'multiplies' if slope > 0 else 'reduces'} Y by {abs(np.exp(slope)):.1%}"
        })
    except: pass

    # Power law: y = a · x^b
    try:
        xp = np.where(x > 0, x, 1e-10)
        yp = np.where(y > 0, y, 1e-10)
        slope, intercept, r, p, se = stats.linregress(np.log(xp), np.log(yp))
        r2 = r**2
        results.append({
            "type": "power", "r2": r2,
            "formula": f"y = {np.exp(intercept):.3f}·x^{slope:.3f}",
            "slope": slope, "intercept": intercept,
            "label": "Scaling law" if abs(slope) > 0.8 else "Diminishing returns" if slope < 1 else "Accelerating growth",
            "business_type": f"10% more X → ~{abs(slope)*10:.1f}% {'more' if slope > 0 else 'less'} Y",
            "direction": "positive" if slope > 0 else "negative",
            "impact_per_unit": abs(slope),
            "impact_text": f"A 10% increase in X leads to ~{abs(slope)*10:.1f}% change in Y"
        })
    except: pass

    # Polynomial (quadratic): y = ax² + bx + c  (sweet spot / turning point)
    try:
        coeffs = np.polyfit(x, y, 2)
        yp = np.polyval(coeffs, x)
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        turning_point = -coeffs[1] / (2 * coeffs[0]) if abs(coeffs[0]) > 1e-10 else None
        tp_text = f" (sweet spot around X={turning_point:.1f})" if turning_point and np.min(x) <= turning_point <= np.max(x) else ""
        results.append({
            "type": "polynomial", "r2": r2,
            "formula": f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}",
            "slope": coeffs[1], "intercept": coeffs[2],
            "label": f"Sweet spot / optimal range{tp_text}",
            "business_type": f"There's an optimal range for X — too much or too little hurts Y{tp_text}",
            "direction": "curved",
            "impact_per_unit": abs(coeffs[1]),
            "impact_text": f"Y peaks at an optimal X value{tp_text} — beyond that, returns reverse"
        })
    except: pass

    if not results:
        return {"type": "none", "r2": 0, "formula": "", "label": "No clear pattern",
                "direction": "unknown", "strength": "none", "business_type": "No clear relationship found"}

    best = max(results, key=lambda d: d["r2"])
    r2 = best["r2"]
    best["strength"] = "very strong" if r2 > 0.9 else "strong" if r2 > 0.7 else "moderate" if r2 > 0.4 else "weak"
    best["confidence_pct"] = round(r2 * 100, 1)
    best["all_types"] = results
    return best


def make_chart_b64(df, x_col, y_col, rel=None):
    """Create a clean, business-friendly chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PINK = "#E91E63"
    PINK_LIGHT = "#F8BBD0"
    BLACK = "#1A1A1A"
    GRAY = "#9E9E9E"
    BG = "#FFFFFF"
    GRID = "#F5F5F5"

    x = df[x_col].astype(float).values
    y = df[y_col].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sidx = np.argsort(x)
    xs, ys = x[sidx], y[sidx]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.scatter(xs, ys, color=PINK, s=55, alpha=0.8, zorder=5, edgecolors="white", linewidths=0.5)

    # Fit line
    if rel and rel.get("r2", 0) > 0.3:
        try:
            from scipy import stats
            xline = np.linspace(xs.min(), xs.max(), 300)
            t = rel.get("type", "")
            yline = None
            if t == "linear":
                s, i, *_ = stats.linregress(xs, ys)
                yline = s * xline + i
            elif t == "logarithmic":
                xp = np.where(xs > 0, xs, 1e-10)
                s, i, *_ = stats.linregress(np.log(xp), ys)
                yline = s * np.log(np.where(xline > 0, xline, 1e-10)) + i
            elif t == "exponential":
                yp = np.where(ys > 0, ys, 1e-10)
                s, i, *_ = stats.linregress(xs, np.log(yp))
                yline = np.exp(i + s * xline)
            elif t == "power":
                xp = np.where(xs > 0, xs, 1e-10)
                yp = np.where(ys > 0, ys, 1e-10)
                s, i, *_ = stats.linregress(np.log(xp), np.log(yp))
                yline = np.exp(i) * np.where(xline > 0, xline, 1e-10)**s
            elif t == "polynomial":
                coeffs = np.polyfit(xs, ys, 2)
                yline = np.polyval(coeffs, xline)

            if yline is not None:
                ax.plot(xline, yline, color=BLACK, lw=2, zorder=4, linestyle="--", alpha=0.7)
        except: pass

    ax.set_xlabel(x_col.replace("_", " ").title(), color=BLACK, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_col.replace("_", " ").title(), color=BLACK, fontsize=11, fontweight="bold")
    ax.set_title(f"{x_col.replace('_',' ').title()}  vs  {y_col.replace('_',' ').title()}",
                 color=BLACK, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor(GRID)
    ax.spines["bottom"].set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_drivers_chart_b64(correlations, y_col):
    """Bar chart showing which factors drive Y most — replaces the heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not correlations:
        return ""

    PINK = "#E91E63"
    RED = "#D32F2F"
    BLACK = "#1A1A1A"
    GRAY = "#9E9E9E"
    BG = "#FFFFFF"

    # Filter only X → Y correlations
    pairs = [c for c in correlations if "→" in c["pair"]]
    if not pairs:
        pairs = correlations[:6]

    names = [c["pair"].split("→")[0].strip().replace("_", " ").title() if "→" in c["pair"]
             else c["pair"].replace("_", " ") for c in pairs]
    vals = [c["value"] for c in pairs]
    colors = [PINK if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(names) * 0.6)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(range(len(names)), [abs(v) for v in vals], color=colors, height=0.55, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, color=BLACK, fontweight="500")
    ax.set_xlabel("Impact strength", fontsize=10, color=GRAY)
    ax.set_title(f"What drives {y_col.replace('_',' ').title()}?",
                 fontsize=13, fontweight="bold", color=BLACK, pad=12)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, vals)):
        sign = "↑" if val > 0 else "↓"
        label = f"{sign} {abs(val):.0%}"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=9, color=BLACK, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#EEE")
    ax.spines["bottom"].set_edgecolor("#EEE")
    ax.tick_params(axis="x", colors=GRAY, labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def call_claude_insight(x_col, y_col, rel, r2, sector, user_question, n_rows, all_correlations=None):
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return fallback_insight(x_col, y_col, rel, r2)

        client = anthropic.Anthropic(api_key=api_key)
        confidence_label = "very strong" if r2 > 0.9 else "strong" if r2 > 0.7 else "moderate" if r2 > 0.4 else "weak"

        corr_context = ""
        if all_correlations:
            corr_context = "\n".join([f"  - {c['pair']}: {c['value']:.3f}" for c in all_correlations[:8]])

        prompt = f"""You are a senior data strategist embedded inside a company.

Your job is NOT to explain formulas or how the math works.
Your job is to extract clear, defensible business insights from data.

The user is a manager (not technical). They need to justify decisions, explain performance changes, and act quickly.

Context:
- Target metric (Y): "{y_col}"
- Key driver (X): "{x_col}"
- Pattern detected: {rel.get('label', 'unknown')} ({rel.get('business_type', '')})
- Impact: {rel.get('impact_text', 'n/a')}
- Confidence: {confidence_label} ({round(r2*100,1)}% match)
- Sector: {sector if sector else 'call center / customer support'}
- Dataset: {n_rows} data points
- Manager's question: {user_question if user_question else 'general analysis'}
- All correlations found:
{corr_context}

STRICT RULES:
- Do NOT mention regression, R², models, algorithms, logarithmic, exponential, polynomial, or any statistical term
- Do NOT show or reference any formula
- Do NOT say "correlation does not imply causation" (the user knows)
- Write as if you're presenting to a VP in a meeting

Structure your answer EXACTLY like this (use these exact emoji headers):

🎯 What's driving {y_col.replace('_',' ')}
One paragraph explaining what is happening in plain business language. Use the actual column names.

📊 What matters most
List the top 2-3 drivers ranked by importance. Use ↑↓ arrows to show direction. Be specific with numbers.

⚡ What to do next
2-3 concrete, specific actions management can take this week.

⚠️ Watch out
One sentence on what could mislead or what to validate before acting.

Tone: sharp, confident, executive-level, no fluff. Mix English and Italian naturally.
Max 250 words."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return fallback_insight(x_col, y_col, rel, r2)


def fallback_insight(x_col, y_col, rel, r2):
    label = rel.get("label", "pattern")
    impact = rel.get("impact_text", "")
    btype = rel.get("business_type", "")
    strength = "very strong" if r2 > 0.9 else "strong" if r2 > 0.7 else "moderate" if r2 > 0.4 else "weak"
    xn = x_col.replace("_", " ")
    yn = y_col.replace("_", " ")

    return (
        f"🎯 What's driving {yn}\n"
        f"There is a {strength} link between {xn} and {yn}. "
        f"{btype}. {impact}.\n\n"
        f"📊 What matters most\n"
        f"↑ {xn.title()} is the #1 factor affecting {yn} (confidence: {round(r2*100)}%)\n"
        f"The pattern is: {label.lower()}\n\n"
        f"⚡ What to do next\n"
        f"Focus on optimizing {xn}. Track it weekly alongside {yn} to confirm the trend.\n\n"
        f"⚠️ Watch out\n"
        f"Validate this with your team before making major changes — "
        f"other factors not in the data may also play a role."
    )


# ─────────────────────── HTML PAGE ───────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SumUp Insights</title>
<!-- Inline CSV Parser (zero CDN dependency) -->
<script>
var Papa=(function(){
  function detectSep(t){var f=t.split(/\r?\n/,2)[0];var sc=(f.match(/;/g)||[]).length;var cc=(f.match(/,/g)||[]).length;var tc=(f.match(/\t/g)||[]).length;if(tc>=sc&&tc>=cc&&tc>0)return'\t';if(sc>cc)return';';return',';}
  function splitRow(line,sep){var cols=[],cur='',inQ=false;for(var i=0;i<line.length;i++){var ch=line[i];if(inQ){if(ch==='"'){if(i+1<line.length&&line[i+1]==='"'){cur+='"';i++;}else{inQ=false;}}else{cur+=ch;}}else{if(ch==='"'){inQ=true;}else if(ch===sep){cols.push(cur);cur='';}else{cur+=ch;}}}cols.push(cur);return cols;}
  function parseFile(file,opts){var reader=new FileReader();reader.onload=function(ev){var text=ev.target.result.replace(/^\xEF\xBB\xBF/,'');var sep=detectSep(text);var lines=text.split(/\r?\n/);while(lines.length>0&&lines[lines.length-1].trim()==='')lines.pop();if(lines.length<2){if(opts.error)opts.error({message:'File has fewer than 2 lines'});return;}var hdr=splitRow(lines[0],sep).map(function(h){return h.trim();});var data=[];for(var i=1;i<lines.length;i++){if(opts.skipEmptyLines&&lines[i].trim()==='')continue;var vals=splitRow(lines[i],sep);var obj={};for(var j=0;j<hdr.length;j++){obj[hdr[j]]=j<vals.length?vals[j].trim():'';}data.push(obj);}var result={data:data,meta:{fields:hdr,delimiter:sep}};if(opts.complete)opts.complete(result);};reader.onerror=function(){if(opts.error)opts.error({message:'FileReader error'});};reader.readAsText(file);}
  function unparse(data){if(!data||data.length===0)return'';var keys=Object.keys(data[0]);var lines=[keys.join(',')];data.forEach(function(row){var vals=keys.map(function(k){var v=String(row[k]!==undefined?row[k]:'');if(v.indexOf(',')!==-1||v.indexOf('"')!==-1||v.indexOf('\n')!==-1)return'"'+v.replace(/"/g,'""')+'"';return v;});lines.push(vals.join(','));});return lines.join('\n');}
  return{parse:parseFile,unparse:unparse};
})();
</script>
<style>
:root{
  --white:#ffffff;--bg:#F9FAFB;--border:#E5E7EB;--card:#ffffff;
  --text:#111827;--muted:#6B7280;--light:#9CA3AF;
  --pink:#E91E63;--pink-light:#FCE4EC;--pink-mid:#F06292;
  --green:#059669;--red:#D32F2F;--amber:#D97706;
  --radius:12px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-height:100vh;font-size:15px}

header{background:#1A1A1A;border-bottom:3px solid var(--pink);padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{width:28px;height:28px;background:var(--pink);border-radius:6px;display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:13px}
.logo-text{font-size:1rem;font-weight:700;color:#FFFFFF;letter-spacing:-.3px}
.logo-sub{font-size:.72rem;color:#999;margin-left:6px;font-weight:400}
.header-badge{font-size:.7rem;background:var(--pink-light);color:var(--pink);border-radius:20px;padding:3px 10px;font-weight:600}

.container{max-width:1080px;margin:0 auto;padding:32px 20px}

.how-section{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:24px}
.how-section h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.steps{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
.step{display:flex;gap:12px;align-items:flex-start}
.step-num{width:28px;height:28px;min-width:28px;background:var(--pink-light);color:var(--pink);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:800}
.step-body h4{font-size:.85rem;font-weight:600;color:var(--text);margin-bottom:3px}
.step-body p{font-size:.78rem;color:var(--muted);line-height:1.55}

.sample-section{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:20px 28px;margin-bottom:20px}
.sample-section h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:12px}
.sample-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.sample-select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem;flex:1;min-width:220px;max-width:380px}
.sample-select:focus{outline:none;border-color:var(--pink)}
.btn-ghost{background:var(--white);color:var(--pink);border:1px solid var(--pink);border-radius:8px;padding:8px 20px;cursor:pointer;font-weight:600;font-size:.82rem;transition:.15s}
.btn-ghost:hover{background:var(--pink-light)}

.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:36px;text-align:center;background:var(--white);margin-bottom:20px;transition:.2s;cursor:pointer}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--pink);background:var(--pink-light)}
.upload-zone h2{color:var(--text);font-size:1rem;font-weight:600;margin-bottom:6px}
.upload-zone p{color:var(--muted);font-size:.83rem;margin-top:4px}
#fi{display:none}
.btn-primary{display:inline-block;margin-top:14px;background:var(--pink);color:white;padding:10px 28px;border-radius:8px;cursor:pointer;font-weight:600;font-size:.9rem;transition:.15s;border:none}
.btn-primary:hover{background:var(--pink-mid)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
#fn{margin-top:10px;color:var(--pink);font-weight:600;min-height:20px;font-size:.85rem}

.preview-box{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;margin-bottom:20px;overflow-x:auto}
.preview-box h4{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:12px}
.preview-table{border-collapse:collapse;font-size:.78rem;width:100%}
.preview-table th{background:var(--bg);color:var(--muted);padding:6px 12px;text-align:left;border-bottom:1px solid var(--border);font-weight:600;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase}
.preview-table td{color:var(--text);padding:5px 12px;border-bottom:1px solid var(--bg)}
.preview-meta{font-size:.72rem;color:var(--light);margin-top:8px}

.wizard-box{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px}
.wizard-box h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.wizard-q{margin-bottom:16px}
.wizard-q label{display:block;font-size:.83rem;font-weight:600;color:var(--text);margin-bottom:8px}
.sector-grid{display:flex;flex-wrap:wrap;gap:8px}
.sector-btn{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:7px 14px;cursor:pointer;color:var(--muted);font-size:.8rem;transition:.15s;font-weight:500}
.sector-btn:hover{border-color:var(--pink);color:var(--pink)}
.sector-btn.active{border-color:var(--pink);color:var(--pink);background:var(--pink-light);font-weight:600}
.wizard-input{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 14px;font-size:.85rem}
.wizard-input:focus{outline:none;border-color:var(--pink)}

.col-select{display:none;background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px}
.col-select h2{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.col-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:18px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:5px;min-width:150px}
.col-group > label{color:var(--text);font-size:.8rem;font-weight:600}
select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem;width:100%}
select:focus{outline:none;border-color:var(--pink)}
.hint{font-size:.72rem;color:var(--light);margin-top:2px}
.val-msg{display:none;color:var(--red);font-size:.8rem;margin-top:8px;padding:8px 12px;background:#FEF2F2;border-radius:6px;border-left:3px solid var(--red)}

.spinner{display:none;text-align:center;padding:40px;color:var(--muted);font-size:.9rem}
.spinner-ring{display:inline-block;width:22px;height:22px;border:3px solid var(--border);border-top-color:var(--pink);border-radius:50%;animation:spin .7s linear infinite;margin-right:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{background:#FEF2F2;border:1px solid #FECACA;border-radius:var(--radius);padding:14px 18px;color:var(--red);margin-bottom:16px;font-size:.85rem}

.results{display:none}

/* ── HERO RESULT: business-first, not formula-first ── */
.result-hero{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:28px 32px;margin-bottom:20px}
.result-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:10px}
.result-headline{font-size:1.4rem;font-weight:700;color:var(--text);margin-bottom:8px;line-height:1.3}
.result-subline{font-size:.9rem;color:var(--muted);margin-bottom:14px;line-height:1.5}
.result-impact{background:var(--pink-light);border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:.88rem;color:var(--text);font-weight:500;line-height:1.5}
.result-impact strong{color:var(--pink)}
.conf-wrap{max-width:380px;margin-bottom:6px}
.conf-track{height:8px;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--border)}
.conf-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--pink),var(--pink-mid));transition:width 1s ease}
.conf-label{font-size:.75rem;color:var(--muted);margin-top:5px}
.formula-toggle{margin-top:12px}
.formula-toggle summary{font-size:.75rem;color:var(--light);cursor:pointer;user-select:none}
.formula-toggle pre{font-family:'SF Mono',Monaco,monospace;font-size:.78rem;color:var(--muted);background:var(--bg);padding:8px 12px;border-radius:6px;margin-top:6px}

.pill{display:inline-block;font-size:.68rem;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:10px;vertical-align:middle}
.pill-strong{background:#D1FAE5;color:var(--green)}
.pill-moderate{background:#FEF3C7;color:var(--amber)}
.pill-weak{background:#FEE2E2;color:var(--red)}

/* ── PANELS ── */
.panels-row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
@media(max-width:680px){.panels-row{grid-template-columns:1fr}}
.panel-card{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:22px 24px}
.panel-card h3{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.panel-card img{width:100%;border-radius:8px;border:1px solid var(--border)}
.insight-text{font-size:.85rem;color:#333;line-height:1.85;white-space:pre-wrap}
.insight-loading{color:var(--pink);font-size:.83rem;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

.full-card{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:22px 24px;margin-bottom:18px}
.full-card h3{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.full-card img{width:100%;border-radius:8px;border:1px solid var(--border)}

/* ── IMPACT SIMULATOR ── */
.simulator-box{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:18px}
.simulator-box h3{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.sim-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px}
.sim-label{font-size:.85rem;color:var(--text);min-width:180px}
.sim-slider{flex:1;min-width:200px;accent-color:var(--pink)}
.sim-value{font-size:.9rem;font-weight:700;color:var(--pink);min-width:60px}
.sim-result{font-size:1rem;font-weight:700;color:var(--text);background:var(--pink-light);padding:12px 16px;border-radius:8px;margin-top:8px}

footer{text-align:center;padding:28px;color:var(--light);font-size:.75rem;border-top:1px solid var(--border);margin-top:16px}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">S+</div>
    <span class="logo-text">SumUp Insights<span class="logo-sub">by FormulaFinder</span></span>
  </div>
  <span class="header-badge">Powered by AI</span>
</header>

<div class="container">

  <div class="how-section">
    <h2>How it works / Come funziona</h2>
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <h4>Upload your data</h4>
          <p>Drop a CSV or Excel file with column headers and numeric data.<br><em>Carica un file CSV o Excel con intestazioni.</em></p>
        </div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          <h4>Tell us what to explain</h4>
          <p>Pick the metric you care about (e.g. resolution time, CSAT) and the factors that might drive it.<br><em>Scegli la metrica da capire e i fattori potenziali.</em></p>
        </div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          <h4>Get actionable insights</h4>
          <p>SumUp finds what drives your metric and tells you <strong>what to do about it</strong> — in plain language.<br><em>SumUp trova cosa influenza la tua metrica e ti dice <strong>cosa fare</strong>.</em></p>
        </div>
      </div>
    </div>
  </div>

  <!-- SAMPLE DATA: Call Center only -->
  <div class="sample-section">
    <h2>No file yet? Try a call center example / Nessun file? Prova un esempio</h2>
    <div class="sample-row">
      <select class="sample-select" id="sampleSelect">
        <option value="">Select an example...</option>
        <optgroup label="Call Center / Customer Support">
          <option value="agent_performance">Agent performance vs resolution time</option>
          <option value="wait_time">Wait time vs customer satisfaction (CSAT)</option>
          <option value="call_volume">Call volume vs abandon rate</option>
          <option value="training_hours">Training hours vs first call resolution</option>
          <option value="shift_staffing">Shift staffing vs avg handle time</option>
        </optgroup>
      </select>
      <button class="btn-ghost" onclick="loadSample()">Load example</button>
    </div>
  </div>

  <!-- UPLOAD -->
  <div class="upload-zone" id="dropZone">
    <h2>Drop your file here</h2>
    <p>Supports CSV and Excel (.xlsx) | Trascina qui il tuo file CSV o Excel</p>
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

  <!-- CONTEXT -->
  <div class="wizard-box" id="wizardBox">
    <h2>Add context for better insights / Aggiungi contesto</h2>
    <div class="wizard-q">
      <label>1. What is your sector? / In quale settore lavori?</label>
      <div class="sector-grid" id="sectorGrid">
        <button class="sector-btn active" onclick="selectSector(this,'Call Center / Customer Support')">Call Center</button>
        <button class="sector-btn" onclick="selectSector(this,'Finance')">Finance</button>
        <button class="sector-btn" onclick="selectSector(this,'Health')">Health</button>
        <button class="sector-btn" onclick="selectSector(this,'Operations')">Operations</button>
        <button class="sector-btn" onclick="selectSector(this,'Retail')">Retail</button>
        <button class="sector-btn" onclick="selectSector(this,'Other')">Other</button>
      </div>
    </div>
    <div class="wizard-q">
      <label>2. What are you trying to understand? (optional)</label>
      <input class="wizard-input" id="userQuestion" placeholder='e.g. "Why is resolution time increasing?" / "Perché il tempo di risoluzione sta aumentando?"'>
    </div>
  </div>

  <!-- CONFIGURE -->
  <div class="col-select" id="colSel">
    <h2>Configure analysis / Configura l'analisi</h2>
    <div class="col-row">
      <div class="col-group" style="min-width:180px">
        <label>🎯 Metric to explain (Y)</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The metric you want to understand</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>📊 Potential drivers (X)</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Hold Cmd/Ctrl to select multiple columns</span>
      </div>
    </div>
    <button class="btn-primary" id="runBtn" onclick="run()">Discover what drives your metric</button>
    <div class="val-msg" id="valMsg"></div>
  </div>

  <!-- SPINNER -->
  <div class="spinner" id="spin">
    <span class="spinner-ring"></span>
    <span>Analyzing your data... / Analizzando i dati...</span>
  </div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- RESULTS -->
  <div class="results" id="res">

    <!-- HERO: Business-first result -->
    <div class="result-hero">
      <div class="result-eyebrow">Discovery / Scoperta</div>
      <div class="result-headline" id="resultHeadline"></div>
      <div class="result-subline" id="resultSubline"></div>
      <div class="result-impact" id="resultImpact"></div>
      <div class="conf-wrap">
        <div class="conf-track"><div class="conf-fill" id="confBar" style="width:0%"></div></div>
        <div class="conf-label" id="confLabel"></div>
      </div>
      <details class="formula-toggle">
        <summary>Show technical formula (for data team)</summary>
        <pre id="formulaPre"></pre>
      </details>
    </div>

    <!-- CHARTS + INSIGHT -->
    <div class="panels-row">
      <div class="panel-card">
        <h3>Trend visualization</h3>
        <div id="chartWrap"></div>
      </div>
      <div class="panel-card">
        <h3>🧠 AI Strategic Insight</h3>
        <div class="insight-text" id="insightText">
          <span class="insight-loading">AI is analyzing your data...</span>
        </div>
      </div>
    </div>

    <!-- DRIVERS CHART (replaces heatmap) -->
    <div class="full-card" id="driversBox" style="display:none">
      <h3>What drives your metric / Cosa influenza la tua metrica</h3>
      <img id="driversImg" src="" alt="Drivers chart">
    </div>

    <!-- IMPACT SIMULATOR (replaces CSV download) -->
    <div class="simulator-box" id="simBox" style="display:none">
      <h3>📐 Impact simulator / Simulatore di impatto</h3>
      <p style="font-size:.82rem;color:var(--muted);margin-bottom:14px">Drag the slider to see how changing the driver affects your metric.<br><em>Trascina il cursore per vedere l'impatto sul tuo risultato.</em></p>
      <div class="sim-row">
        <span class="sim-label" id="simLabel">Driver value:</span>
        <input type="range" class="sim-slider" id="simSlider" min="0" max="100" value="50" oninput="updateSim()">
        <span class="sim-value" id="simVal">50</span>
      </div>
      <div class="sim-result" id="simResult">Estimated outcome: --</div>
    </div>

  </div>

</div>

<footer>SumUp Insights — Data insight engine for management &nbsp;|&nbsp; FormulaFinder SRL</footer>

<script>
var csv = null;
var allCols = [];
var selectedSector = 'Call Center / Customer Support';
var lastResults = null;
var uploadedFilename = 'data.csv';
var simParams = null;

var SAMPLES = {
  agent_performance: "calls_handled,avg_handle_time_min,experience_months,training_score,resolution_rate_pct\n45,8.2,24,85,78\n32,11.5,6,62,55\n50,7.1,36,91,88\n28,13.0,3,58,45\n42,8.8,18,79,72\n55,6.5,48,95,92\n38,9.5,12,74,65\n48,7.5,30,88,82\n25,14.2,2,52,38\n52,6.8,42,93,90\n35,10.2,9,68,58\n44,8.0,20,82,76\n30,12.0,4,60,48\n47,7.3,28,87,84\n40,9.0,15,76,70",
  wait_time: "avg_wait_seconds,csat_score,abandon_rate_pct,calls_per_hour,agents_on_shift\n30,92,2,120,25\n60,85,5,150,22\n90,75,8,180,20\n120,65,14,200,18\n45,88,3,135,24\n150,55,22,220,16\n75,78,6,165,21\n20,95,1,100,28\n105,70,11,190,19\n180,45,30,240,14\n55,82,4,145,23\n135,58,18,210,17\n40,90,2.5,125,26\n100,72,10,185,20\n160,50,25,230,15",
  call_volume: "daily_calls,abandon_rate_pct,avg_wait_seconds,agents_available,service_level_pct\n500,3,25,30,92\n750,5,45,28,85\n1000,8,70,28,75\n1200,12,95,26,65\n600,4,30,30,90\n1500,18,130,24,52\n800,6,50,28,82\n400,2,18,32,95\n1100,10,85,26,70\n1800,25,180,22,40\n650,4.5,35,29,88\n1300,14,110,25,58\n900,7,60,28,78\n550,3.5,28,30,91\n1400,16,120,24,55",
  training_hours: "training_hours,first_call_resolution_pct,avg_handle_time_min,errors_per_100_calls,csat_score\n4,45,14,12,58\n8,55,12,9,65\n16,68,10,6,75\n24,75,8.5,4,82\n32,80,7.8,3,86\n40,83,7.2,2.5,88\n48,85,7.0,2,90\n56,86,6.8,1.8,91\n64,87,6.7,1.5,92\n80,87.5,6.6,1.4,92\n12,60,11,8,70\n20,72,9,5,78\n36,82,7.5,2.8,87\n44,84,7.1,2.2,89\n72,87,6.7,1.5,92",
  shift_staffing: "agents_on_shift,avg_handle_time_min,calls_in_queue,wait_time_seconds,csat_score\n15,12,45,180,55\n18,11,35,120,62\n20,10,25,90,70\n22,9.5,18,65,76\n25,8.5,10,40,84\n28,8,6,25,88\n30,7.8,4,18,91\n32,7.5,3,15,93\n35,7.2,2,10,95\n38,7.1,1,8,95.5\n16,11.5,40,160,58\n19,10.5,30,100,66\n24,9,12,50,80\n27,8.2,7,30,86\n33,7.4,3,12,94"
};

function loadSample() {
  var s = document.getElementById('sampleSelect').value;
  if (!s) return;
  Papa.parse(SAMPLES[s], {
    header:true, skipEmptyLines:true, dynamicTyping:false,
    complete: function(result) {
      csv = Papa.unparse(result.data);
      uploadedFilename = s + '.csv';
      allCols = result.meta.fields || [];
      document.getElementById('fn').textContent = 'Example loaded: ' + s.replace(/_/g,' ');
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
    var reader = new FileReader();
    reader.onload = async function(e) {
      var b64 = btoa(String.fromCharCode.apply(null, new Uint8Array(e.target.result)));
      try {
        var resp = await fetch('/api/parse_excel', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({b64: b64, filename: f.name})
        });
        var d = await resp.json();
        if (!d.success) throw new Error(d.error);
        csv = d.csv; allCols = d.columns;
        renderPreviewTable(d.columns, d.preview, d.rows, d.columns.length);
        parseCols();
        document.getElementById('wizardBox').style.display = 'block';
        document.getElementById('fn').textContent = 'Loaded: ' + f.name + ' (' + d.rows + ' rows)';
      } catch(err) { document.getElementById('fn').textContent = 'Error: ' + err.message; }
    };
    reader.readAsArrayBuffer(f);
  } else {
    Papa.parse(f, {
      header:true, skipEmptyLines:true, dynamicTyping:false,
      complete: function(result) {
        if (!result.data || result.data.length===0) {
          document.getElementById('fn').textContent = 'Could not parse file.';
          return;
        }
        csv = Papa.unparse(result.data);
        allCols = result.meta.fields || [];
        showPreview(result);
        parseCols();
        document.getElementById('wizardBox').style.display = 'block';
        document.getElementById('fn').textContent = 'Loaded: ' + f.name + ' (' + result.data.length + ' rows)';
      },
      error: function(err){ document.getElementById('fn').textContent = 'Error: ' + err.message; }
    });
  }
}

function showPreview(result) {
  renderPreviewTable(result.meta.fields||[], result.data.slice(0,4), result.data.length, (result.meta.fields||[]).length);
}
function renderPreviewTable(cols, rows, total, ncols) {
  var box = document.getElementById('previewBox');
  var html = '<table class="preview-table"><thead><tr>';
  cols.forEach(function(c){ html += '<th>'+esc(c)+'</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(function(r){
    html += '<tr>';
    cols.forEach(function(c){ html += '<td>'+esc(String(r[c]!==undefined?r[c]:''))+'</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('previewTable').innerHTML = html;
  document.getElementById('previewMeta').textContent = total + ' rows · ' + ncols + ' columns';
  box.style.display = 'block';
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function parseCols() {
  var yHints = ['csat','satisfaction','resolution','score','rate','time','revenue','cost','performance'];
  var guessedY = allCols.length - 1;
  for(var i=0;i<allCols.length;i++){
    var lo = allCols[i].toLowerCase();
    for(var j=0;j<yHints.length;j++){
      if(lo.indexOf(yHints[j])!==-1){guessedY=i;break;}
    }
  }
  var yE = document.getElementById('yCol');
  yE.innerHTML = '';
  allCols.forEach(function(col,i){
    var opt = document.createElement('option');
    opt.value=col; opt.textContent=col.replace(/_/g,' ');
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
    opt.value=col; opt.textContent=col.replace(/_/g,' ');
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
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({csv:csv, y_col:yc, x_cols:xc,
        sector:selectedSector,
        user_question:document.getElementById('userQuestion').value})
    });
    var d = await resp.json();
    if(!d.success) throw new Error(d.error);
    lastResults=d; lastResults._yc=yc; lastResults._xc=xc;
    showResults(d,yc,xc);
