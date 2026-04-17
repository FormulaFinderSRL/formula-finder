#!/usr/bin/env python3
"""
Formula Finder — Discovery Engine
Find the law behind your numbers.
Powered by EML + Adam optimizer.
"""
import numpy as np, pandas as pd, base64, io, warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  MATH CORE  —  DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════

class Adam:
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None; self.t = 0
    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w, dtype=float)
            self.v = np.zeros_like(w, dtype=float)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)

def eml(x, y, eps=1e-10):
    y_s = np.where(np.array(y) > 0, np.array(y), eps)
    return np.exp(np.clip(np.array(x), -500, 500)) - np.log(y_s)

def build_features(X_dict):
    feats = [np.ones(len(list(X_dict.values())[0]))]
    names = ["1"]
    for v, x in X_dict.items():
        x  = np.asarray(x, float)
        xc = np.clip(x, -15, 15)
        xp = np.where(x > 1e-6, x, 1e-6)
        feats += [x, x**2, x**3,
                  np.exp(xc), np.exp(-xc),
                  np.log(xp), np.sin(x), np.cos(x),
                  np.tanh(x), np.sqrt(np.abs(x)),
                  1/(1+x**2), x*np.exp(xc), x*np.sin(x)]
        names += [v, v+"^2", v+"^3",
                  "exp("+v+")", "exp(-"+v+")",
                  "ln("+v+")", "sin("+v+")", "cos("+v+")",
                  "tanh("+v+")", "sqrt|"+v+"|",
                  "1/(1+"+v+"^2)", v+"*exp("+v+")", v+"*sin("+v+")"]
    vlist = list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1, len(vlist)):
            v1, v2 = vlist[i], vlist[j]
            x1 = np.asarray(X_dict[v1], float)
            x2 = np.asarray(X_dict[v2], float)
            feats += [x1*x2, eml(x1, x2)]
            names += [v1+"*"+v2, "eml("+v1+","+v2+")"]
    return np.column_stack(feats), names

class EMLAdamRegressor:
    def __init__(self, lr=0.05, epochs=1500, l1=1e-3):
        self.lr, self.epochs, self.l1 = lr, epochs, l1
        self.w = self.names = None
    def fit(self, X_dict, y):
        F, self.names = build_features(X_dict)
        y = np.asarray(y, float)
        self.w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        opt = Adam(lr=self.lr)
        for _ in range(self.epochs):
            yp = F @ self.w
            g  = (2/len(y)) * (F.T @ (yp - y)) + self.l1 * np.sign(self.w)
            self.w = opt.step(self.w, g)
        return self
    def predict(self, X_dict):
        F, _ = build_features(X_dict)
        return F @ self.w
    def r2(self, X_dict, y):
        yp = self.predict(X_dict)
        y  = np.asarray(y, float)
        ss_r = np.sum((y - yp)**2)
        ss_t = np.sum((y - np.mean(y))**2)
        return float(1 - ss_r/ss_t) if ss_t > 0 else 1.0
    def formula(self, thr=0.05):
        terms = []
        for n, w in zip(self.names, self.w):
            if abs(w) > thr:
                terms.append("%.4f" % w if n == "1" else "%.4f*%s" % (w, n))
        if not terms: return "y = 0"
        result = "y = " + terms[0]
        for t in terms[1:]:
            result += (" - " + t[1:]) if t.startswith("-") else (" + " + t)
        return result
    def top_terms(self, n=8):
        idx = np.argsort(np.abs(self.w))[::-1][:n]
        return [{"term": self.names[i], "weight": round(float(self.w[i]), 4)}
                for i in idx if abs(self.w[i]) > 0.01]

def quick_search(X_dict, y, top_n=8, min_r2=0.5):
    ops = {}
    for v, x in X_dict.items():
        ops["exp("+v+")"]   = lambda d, v=v: np.exp(np.clip(d[v], -500, 500))
        ops["ln("+v+")"]    = lambda d, v=v: np.log(np.where(d[v]>0, d[v], 1e-10))
        ops[v+"^2"]         = lambda d, v=v: d[v]**2
        ops[v+"^3"]         = lambda d, v=v: d[v]**3
        ops["sqrt("+v+")"]  = lambda d, v=v: np.sqrt(np.abs(d[v]))
        ops["sin("+v+")"]   = lambda d, v=v: np.sin(d[v])
        ops["cos("+v+")"]   = lambda d, v=v: np.cos(d[v])
        ops[v]              = lambda d, v=v: d[v]
        ops["eml("+v+",1)"] = lambda d, v=v: eml(d[v], np.ones(len(d[v])))
    vlist = list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1, len(vlist)):
            v1, v2 = vlist[i], vlist[j]
            ops[v1+"*"+v2] = lambda d, v1=v1, v2=v2: d[v1]*d[v2]
    y = np.asarray(y, float); n = len(y); res = []
    for nm, fn in ops.items():
        try:
            f = np.asarray(fn(X_dict), float)
            if np.any(np.isnan(f)) | np.any(np.isinf(f)): continue
            A = np.column_stack([f, np.ones(n)])
            c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(c[0]), float(c[1])
            yp   = a*f + b
            ss_r = np.sum((y-yp)**2); ss_t = np.sum((y-np.mean(y))**2)
            r2   = float(1 - ss_r/ss_t) if ss_t > 0 else 1.0
            if r2 >= min_r2:
                if   abs(a-1) < 0.005 and abs(b) < 0.005: fs = "y = " + nm
                elif abs(b)   < 0.005:                     fs = "y = %.4f*%s" % (a, nm)
                elif b < 0:                                fs = "y = %.4f*%s − %.4f" % (a, nm, abs(b))
                else:                                      fs = "y = %.4f*%s + %.4f" % (a, nm, b)
                res.append({"formula": fs, "r2": round(r2,6),
                            "accuracy": "%.4f%%" % (r2*100),
                            "quality": "PERFECT" if r2>0.9999 else "GREAT" if r2>0.99 else "GOOD"})
        except: pass
    res.sort(key=lambda d: d["r2"], reverse=True)
    return res[:top_n]

def make_chart_b64(X_dict, y_true, model=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        BG, MID = "#0A0F1E", "#0D1627"
        vlist   = list(X_dict.keys())
        x_vals  = np.asarray(X_dict[vlist[0]], float)
        y_arr   = np.asarray(y_true, float)
        sidx    = np.argsort(x_vals)
        xs, ys  = x_vals[sidx], y_arr[sidx]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(BG)
        # — left: scatter + fit
        ax = axes[0]; ax.set_facecolor(MID)
        ax.scatter(xs, ys, color="#00C8FF", s=28, alpha=0.75, label="Observed", zorder=5)
        if model is not None:
            try:
                yp = model.predict({vlist[0]: xs})
                r2 = model.r2(X_dict, y_arr)
                ax.plot(xs, yp, color="#FF4B6E", lw=2.5,
                        label="Model  R²=%.4f" % r2, zorder=4)
            except: pass
        ax.set_title("Observed vs Model  (%s)" % vlist[0], color="white", fontsize=10, pad=8)
        ax.set_xlabel(vlist[0], color="#6B8CAE", fontsize=9)
        ax.set_ylabel("target", color="#6B8CAE", fontsize=9)
        ax.tick_params(colors="#4A6080")
        ax.legend(facecolor="#0A1828", labelcolor="white", fontsize=8, framealpha=0.8)
        for s in ax.spines.values(): s.set_edgecolor("#1E3050")
        # — right: residuals
        ax2 = axes[1]; ax2.set_facecolor(MID)
        if model is not None:
            try:
                res = ys - model.predict({vlist[0]: xs})
                colors = ["#00C8FF" if r >= 0 else "#FF4B6E" for r in res]
                ax2.bar(range(len(res)), res, color=colors, alpha=0.7)
                ax2.axhline(0, color="#FFD166", lw=1.5, ls="--", alpha=0.8)
                ax2.set_title("Residuals", color="white", fontsize=10, pad=8)
            except:
                ax2.set_title("Residuals (unavailable)", color="#4A6080", fontsize=10)
        else:
            ax2.set_title("Residuals", color="#4A6080", fontsize=10)
        ax2.tick_params(colors="#4A6080")
        for s in ax2.spines.values(): s.set_edgecolor("#1E3050")
        plt.tight_layout(pad=2.0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
        plt.close(fig); buf.seek(0)
        data = base64.b64encode(buf.read()).decode(); buf.close()
        return data
    except Exception as ex:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig2, ax3 = plt.subplots(figsize=(6,2))
            fig2.patch.set_facecolor("#0A0F1E"); ax3.set_facecolor("#0A0F1E")
            ax3.text(0.5, 0.5, "Chart error: %s" % str(ex),
                     color="#FF4B6E", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=9)
            ax3.axis("off"); buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=80, bbox_inches="tight", facecolor="#0A0F1E")
            plt.close(fig2); buf2.seek(0)
            return base64.b64encode(buf2.read()).decode()
        except: return ""

def _plain_english(top3, y_col):
    if not top3: return "No clear pattern found."
    parts = []
    for t in top3:
        dirn = "increases" if t["weight"] > 0 else "decreases"
        parts.append("%s %s %s" % (t["term"], dirn, y_col))
    return "; ".join(parts) + "."

# ═══════════════════════════════════════════════════════════════
#  HTML  —  NEW CONCEPT: DISCOVERY ENGINE
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formula Finder — Find the law behind your numbers</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<style>
/* ── Reset & tokens ── */
:root {
  --bg:     #0A0F1E;
  --layer1: #0D1627;
  --layer2: #111D35;
  --layer3: #162240;
  --accent: #00C8FF;
  --green:  #00E5A0;
  --red:    #FF4B6E;
  --amber:  #FFD166;
  --white:  #F0F6FF;
  --silver: #7A9BBF;
  --dim:    #3A5070;
  --radius: 12px;
  --font:   'Inter', 'Segoe UI', system-ui, sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--white);font-family:var(--font);min-height:100vh;line-height:1.6}

/* ── Header ── */
header {
  position:sticky;top:0;z-index:100;
  background:rgba(10,15,30,0.92);
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--dim);
  padding:14px 40px;
  display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;
}
.logo { display:flex;align-items:center;gap:12px; }
.logo-mark {
  width:34px;height:34px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),var(--green));
  display:flex;align-items:center;justify-content:center;
  font-size:.9rem;font-weight:900;color:#0A0F1E;letter-spacing:-1px;
}
.logo-text { font-size:1.1rem;font-weight:700;letter-spacing:1.5px;color:var(--white); }
.logo-sub  { font-size:.65rem;color:var(--silver);letter-spacing:.5px;margin-top:1px; }
.header-pill {
  font-size:.68rem;font-weight:700;letter-spacing:1.5px;
  border:1px solid var(--accent);color:var(--accent);
  border-radius:20px;padding:4px 14px;opacity:.85;
}

/* ── Hero ── */
.hero {
  max-width:820px;margin:64px auto 52px;padding:0 24px;text-align:center;
}
.hero-eyebrow {
  font-size:.72rem;font-weight:700;letter-spacing:3px;
  color:var(--accent);text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;justify-content:center;gap:8px;
}
.hero-eyebrow::before,.hero-eyebrow::after {
  content:'';flex:1;max-width:48px;height:1px;background:var(--accent);opacity:.4;
}
.hero h1 {
  font-size:clamp(2rem,5vw,3.2rem);font-weight:800;letter-spacing:-1px;
  line-height:1.15;margin-bottom:20px;
  background:linear-gradient(135deg,var(--white) 40%,var(--accent));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub {
  font-size:1.05rem;color:var(--silver);max-width:580px;margin:0 auto 32px;line-height:1.7;
}
.use-cases {
  display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-bottom:0;
}
.use-case-pill {
  font-size:.72rem;font-weight:600;letter-spacing:.6px;
  background:var(--layer2);border:1px solid var(--dim);
  color:var(--silver);border-radius:20px;padding:5px 14px;
}

/* ── Container ── */
.container { max-width:1100px;margin:0 auto;padding:0 24px 60px; }

/* ── Section label ── */
.section-label {
  font-size:.65rem;font-weight:700;letter-spacing:2.5px;
  color:var(--accent);text-transform:uppercase;margin-bottom:12px;opacity:.8;
}

/* ── Upload zone ── */
.upload-zone {
  border:2px dashed var(--dim);border-radius:var(--radius);
  padding:48px 32px;text-align:center;background:var(--layer1);
  margin-bottom:20px;transition:.25s;cursor:pointer;
}
.upload-zone:hover,.upload-zone.dragover {
  border-color:var(--accent);background:var(--layer2);
}
.upload-icon { font-size:2.4rem;margin-bottom:12px;opacity:.7; }
.upload-zone h2 { font-size:1.1rem;font-weight:600;color:var(--white);margin-bottom:6px; }
.upload-zone p  { font-size:.85rem;color:var(--silver); }
#fi { display:none }
.upload-btn {
  display:inline-block;margin-top:18px;
  background:linear-gradient(135deg,var(--accent),#0095BF);
  color:#0A0F1E;font-weight:700;font-size:.9rem;
  padding:11px 32px;border-radius:8px;cursor:pointer;
  letter-spacing:.5px;transition:.2s;border:none;
}
.upload-btn:hover { opacity:.88;transform:translateY(-1px); }
#fn { margin-top:14px;color:var(--green);font-weight:600;font-size:.88rem;min-height:20px; }

/* ── Template panel ── */
.tpl-panel {
  background:var(--layer1);border:1px solid var(--dim);border-radius:var(--radius);
  padding:20px 24px;margin-bottom:20px;
}
.tpl-panel p { font-size:.8rem;color:var(--silver);margin-bottom:14px;margin-top:4px; }
.tpl-row { display:flex;gap:10px;flex-wrap:wrap;align-items:center; }
.tpl-select {
  flex:1;min-width:200px;max-width:400px;
  background:var(--layer2);color:var(--white);
  border:1px solid var(--dim);border-radius:8px;
  padding:9px 12px;font-size:.84rem;
}
.tpl-btn {
  background:var(--layer2);color:var(--accent);
  border:1px solid var(--accent);border-radius:8px;
  padding:9px 20px;cursor:pointer;font-weight:700;
  font-size:.8rem;letter-spacing:.5px;transition:.2s;white-space:nowrap;
}
.tpl-btn:hover { background:var(--accent);color:#0A0F1E; }
.tpl-desc {
  display:none;font-size:.75rem;color:var(--silver);
  margin-top:10px;padding:7px 12px;
  background:var(--layer2);border-radius:6px;
  border-left:3px solid var(--accent);
}

/* ── Preview ── */
.preview-box {
  display:none;background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);padding:16px;margin-bottom:20px;overflow-x:auto;
}
.preview-box h4 { color:var(--accent);font-size:.72rem;letter-spacing:1.5px;margin-bottom:12px; }
.preview-table { border-collapse:collapse;font-size:.76rem;width:100%; }
.preview-table th {
  background:#050A14;color:var(--green);
  padding:6px 12px;text-align:left;
  border-bottom:1px solid var(--dim);font-weight:600;
}
.preview-table td { color:var(--silver);padding:5px 12px;border-bottom:1px solid var(--layer2); }
.preview-table tr:last-child td { border-bottom:none; }
.preview-meta { font-size:.7rem;color:var(--silver);margin-top:8px;opacity:.6; }

/* ── Configure ── */
.col-select {
  display:none;background:var(--layer1);border:1px solid var(--accent);
  border-radius:var(--radius);padding:24px;margin-bottom:20px;
}
.col-select h3 { color:var(--white);font-size:.95rem;font-weight:700;margin-bottom:18px; }
.col-row { display:flex;gap:20px;flex-wrap:wrap;margin-bottom:18px;align-items:flex-start; }
.col-group { display:flex;flex-direction:column;gap:6px;min-width:150px; }
.col-group > label {
  color:var(--silver);font-size:.72rem;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
}
select {
  background:var(--layer2);color:var(--white);
  border:1px solid var(--dim);border-radius:8px;
  padding:9px 12px;font-size:.88rem;width:100%;
  transition:.2s;
}
select:focus { outline:none;border-color:var(--accent); }
.hint { font-size:.7rem;color:var(--silver);opacity:.6;margin-top:2px; }
.formula-preview {
  font-size:.78rem;color:var(--silver);background:var(--layer2);
  border-radius:6px;padding:8px 14px;margin-top:8px;margin-bottom:4px;
  min-height:24px;font-family:monospace;border-left:3px solid var(--accent);
}
.val-msg {
  display:none;color:var(--red);font-size:.82rem;
  margin-top:8px;padding:8px 14px;
  background:#1A080E;border-radius:6px;border-left:3px solid var(--red);
}
.run-btn {
  background:linear-gradient(135deg,var(--green),#00A870);
  color:#0A0F1E;border:none;
  padding:13px 42px;border-radius:8px;cursor:pointer;
  font-weight:800;font-size:.95rem;margin-top:8px;
  letter-spacing:.8px;transition:.2s;
}
.run-btn:hover { opacity:.9;transform:translateY(-1px); }
.run-btn:disabled { opacity:.4;cursor:not-allowed;transform:none; }

/* ── Data quality ── */
.dq-panel {
  display:none;background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);padding:20px 24px;margin-bottom:20px;
}
.dq-title { font-size:.65rem;font-weight:700;letter-spacing:2px;color:var(--white);margin-bottom:14px; }
.dq-row {
  display:flex;align-items:center;gap:12px;
  padding:7px 0;border-bottom:1px solid var(--layer2);font-size:.78rem;
}
.dq-row:last-child { border-bottom:none; }
.dq-icon { font-size:.95rem;min-width:20px;text-align:center; }
.dq-label { color:var(--silver);min-width:140px;font-size:.74rem;letter-spacing:.3px; }
.dq-value { color:var(--white);flex:1;font-size:.74rem; }
.dq-ok  { color:var(--green)  }
.dq-warn{ color:var(--amber) }
.dq-err { color:var(--red)   }
.dq-score-bar { height:4px;border-radius:2px;background:var(--layer2);margin:14px 0 4px;overflow:hidden; }
.dq-score-fill { height:100%;border-radius:2px;transition:width .6s ease; }
.dq-score-label { font-size:.68rem;color:var(--silver);text-align:right; }
.dq-blocker {
  display:none;background:#1A080E;border:1px solid var(--red);
  border-radius:8px;padding:13px 16px;margin-bottom:16px;
  font-size:.8rem;color:var(--red);line-height:1.7;
}
.dq-warning {
  display:none;background:#1A1200;border:1px solid var(--amber);
  border-radius:8px;padding:13px 16px;margin-bottom:16px;
  font-size:.8rem;color:var(--amber);line-height:1.7;
}

/* ── Spinner ── */
.spinner {
  display:none;text-align:center;padding:40px;
  color:var(--accent);font-size:.9rem;letter-spacing:.5px;
}
.spinner::after {
  content:'';display:inline-block;
  width:18px;height:18px;
  border:2px solid var(--accent);border-top-color:transparent;
  border-radius:50%;animation:spin .75s linear infinite;
  margin-left:10px;vertical-align:middle;
}
@keyframes spin { to { transform:rotate(360deg) } }

/* ── Error ── */
.error-box {
  background:#1A080E;border:1px solid var(--red);border-radius:8px;
  padding:14px 16px;color:var(--red);margin-bottom:16px;font-size:.86rem;
}

/* ── Results ── */
.results { display:none; }

/* Discovery banner */
.discovery-banner {
  background:linear-gradient(135deg,#0D2A3A,#0A1E30);
  border:1px solid var(--accent);border-radius:var(--radius);
  padding:28px 32px;margin-bottom:20px;position:relative;overflow:hidden;
}
.discovery-banner::before {
  content:'';position:absolute;top:-40px;right:-40px;
  width:160px;height:160px;
  background:radial-gradient(circle,rgba(0,200,255,.08),transparent 70%);
  border-radius:50%;
}
.discovery-label {
  font-size:.62rem;font-weight:700;letter-spacing:2.5px;
  color:var(--accent);text-transform:uppercase;margin-bottom:10px;
}
.discovery-formula {
  font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:1.05rem;font-weight:700;color:var(--white);
  word-break:break-all;line-height:1.7;
  max-height:110px;overflow:hidden;transition:max-height .4s ease;
}
.discovery-formula.expanded { max-height:3000px; }
.expand-btn {
  background:rgba(0,200,255,.1);border:1px solid rgba(0,200,255,.2);
  color:var(--accent);font-size:.72rem;cursor:pointer;
  margin-top:8px;padding:4px 12px;border-radius:20px;font-weight:600;
  transition:.2s;
}
.expand-btn:hover { background:rgba(0,200,255,.2); }
.discovery-meta { font-size:.84rem;color:var(--silver);margin-top:10px; }
.discovery-meta strong { color:var(--green); }
.result-actions { display:flex;gap:10px;margin-top:16px;flex-wrap:wrap; }
.action-btn {
  background:rgba(0,0,0,.3);border:1px solid var(--dim);
  color:var(--silver);font-size:.74rem;font-weight:700;
  padding:6px 16px;border-radius:20px;cursor:pointer;transition:.2s;
  letter-spacing:.4px;
}
.action-btn:hover { border-color:var(--accent);color:var(--accent); }

/* ── 3-panel row ── */
.action-panels { display:none;gap:16px;margin-bottom:20px;flex-wrap:wrap; }
.action-panels.visible { display:flex; }
.panel {
  background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);padding:22px;flex:1;min-width:280px;
}
.panel h3 {
  color:var(--accent);font-size:.72rem;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px;
}
.input-grid { display:flex;flex-direction:column;gap:10px;margin-bottom:14px; }
.input-row  { display:flex;align-items:center;gap:10px; }
.input-row label { color:var(--silver);font-size:.76rem;min-width:100px;font-family:monospace; }
.input-row input {
  background:var(--layer2);color:var(--white);
  border:1px solid var(--dim);border-radius:6px;
  padding:7px 10px;font-size:.85rem;width:100%;transition:.2s;
}
.input-row input:focus { outline:none;border-color:var(--accent); }
.predict-result {
  font-size:1.6rem;font-weight:800;color:var(--green);
  font-family:monospace;margin:10px 0 4px;
}
.predict-label { font-size:.72rem;color:var(--silver); }
.sens-row { display:flex;align-items:center;padding:6px 0;border-bottom:1px solid var(--layer2);gap:10px; }
.sens-name { font-family:monospace;color:var(--white);font-size:.76rem;min-width:100px; }
.sens-bar-wrap { flex:1;height:5px;background:var(--layer2);border-radius:3px;overflow:hidden; }
.sens-bar { height:100%;border-radius:3px;transition:width .5s ease; }
.sens-bar.pos { background:var(--green); }
.sens-bar.neg { background:var(--red); }
.sens-val { font-size:.72rem;color:var(--silver);font-family:monospace;min-width:70px;text-align:right; }
.explain-box {
  font-size:.8rem;color:var(--silver);line-height:1.75;
  white-space:pre-wrap;background:var(--layer2);
  border-radius:8px;padding:14px;border-left:3px solid var(--accent);
}
.panel-btn {
  background:linear-gradient(135deg,var(--accent),#0095BF);
  color:#0A0F1E;border:none;
  padding:9px 22px;border-radius:7px;cursor:pointer;
  font-weight:700;font-size:.8rem;letter-spacing:.5px;transition:.2s;margin-top:4px;
}
.panel-btn:hover { opacity:.88; }
.panel-btn:disabled { opacity:.4;cursor:not-allowed; }

/* ── Chart ── */
.chart-box {
  background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);padding:20px;margin-bottom:20px;
}
.chart-box h3 { color:var(--white);font-size:.78rem;font-weight:700;letter-spacing:1px;margin-bottom:14px; }
.chart-box img { width:100%;border-radius:8px;display:block; }
.chart-err { color:var(--silver);font-size:.83rem;padding:20px;text-align:center;background:var(--layer2);border-radius:8px; }

/* ── Top terms ── */
.terms-box {
  background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);padding:20px;margin-bottom:20px;
}
.terms-box h3 { color:var(--white);font-size:.78rem;font-weight:700;letter-spacing:1px;margin-bottom:14px; }
.term-row { display:flex;align-items:center;padding:7px 0;border-bottom:1px solid var(--layer2); }
.term-name { font-family:monospace;color:var(--white);min-width:160px;font-size:.8rem; }
.term-bar-wrap { flex:1;margin:0 14px;height:6px;background:var(--layer2);border-radius:3px;overflow:hidden; }
.term-bar { height:100%;background:linear-gradient(90deg,var(--accent),var(--green));border-radius:3px;transition:width .6s ease; }
.term-w { color:var(--amber);font-size:.78rem;min-width:90px;text-align:right;font-family:monospace; }

/* ── Quick search cards ── */
.cards-section h3 { color:var(--white);font-size:.78rem;font-weight:700;letter-spacing:1px;margin-bottom:14px; }
.cards-grid {
  display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:12px;margin-bottom:24px;
}
.card {
  background:var(--layer1);border:1px solid var(--dim);
  border-radius:10px;padding:16px;transition:.2s;
}
.card:hover { border-color:var(--accent);transform:translateY(-2px); }
.card-rank { font-size:.68rem;color:var(--silver);letter-spacing:1px;margin-bottom:4px; }
.card-formula {
  font-family:monospace;font-size:.86rem;color:var(--white);
  margin:6px 0;word-break:break-all;line-height:1.55;
}
.card-r2 { font-size:.76rem;margin-top:4px; }
.qp { color:var(--green) } .qg { color:#2ECC71 } .qb { color:var(--amber) }
.badge {
  display:inline-block;font-size:.64rem;padding:2px 8px;
  border-radius:20px;font-weight:700;margin-right:6px;
}
.badge-p { background:rgba(0,229,160,.12);color:var(--green) }
.badge-g { background:rgba(46,204,113,.12);color:#2ECC71 }
.badge-b { background:rgba(255,209,102,.12);color:var(--amber) }

/* ── How it works ── */
.how-box {
  background:var(--layer1);border:1px solid var(--dim);
  border-radius:var(--radius);margin-bottom:20px;overflow:hidden;
}
.how-toggle {
  width:100%;background:none;border:none;color:var(--silver);
  font-size:.8rem;font-weight:700;letter-spacing:1px;text-align:left;
  padding:15px 22px;cursor:pointer;
  display:flex;justify-content:space-between;align-items:center;
  transition:.2s;
}
.how-toggle:hover { color:var(--white);background:var(--layer2); }
.how-arrow { transition:transform .3s;font-size:.7rem;display:inline-block; }
.how-body { display:none;padding:18px 22px 22px;border-top:1px solid var(--dim); }
.how-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-bottom:14px; }
.how-card { background:var(--layer2);border-radius:10px;padding:14px 16px; }
.how-step { font-size:.64rem;color:var(--green);font-weight:700;letter-spacing:1.5px;margin-bottom:6px; }
.how-card p { font-size:.78rem;color:var(--silver);line-height:1.6; }
.how-card strong { color:var(--white); }
.how-card code { background:#050A14;color:var(--green);padding:1px 5px;border-radius:3px;font-size:.74rem; }
.how-warn {
  font-size:.76rem;color:var(--amber);background:#1A1200;
  border-left:3px solid var(--amber);padding:10px 14px;
  border-radius:6px;margin-top:6px;line-height:1.65;
}

/* ── Footer ── */
footer {
  text-align:center;padding:28px;color:var(--dim);
  font-size:.72rem;border-top:1px solid var(--layer2);
  margin-top:20px;letter-spacing:.3px;
}

/* ── Responsive ── */
@media(max-width:640px){
  header { padding:12px 20px; }
  .hero  { margin:40px auto 36px; }
  .col-row,.action-panels { flex-direction:column; }
  .col-group,.panel { min-width:100%; }
  .discovery-formula { font-size:.9rem; }
}
</style>
</head>
<body>

<!-- ══ Header ══ -->
<header>
  <div class="logo">
    <div class="logo-mark">FF</div>
    <div>
      <div class="logo-text">FORMULA FINDER</div>
      <div class="logo-sub">Discovery Engine &nbsp;·&nbsp; EML + Adam</div>
    </div>
  </div>
  <div class="header-pill">BETA</div>
</header>

<!-- ══ Hero ══ -->
<div class="hero">
  <div class="hero-eyebrow">Discovery Engine</div>
  <h1>Find the law behind<br>your numbers.</h1>
  <p class="hero-sub">
    The world is full of data. Nobody knows what it means.<br>
    Upload a CSV — Formula Finder extracts the mathematical law that governs your data. In 30 seconds.
  </p>
  <div class="use-cases">
    <span class="use-case-pill">⚡ Fintech &amp; Finance</span>
    <span class="use-case-pill">🔬 Scientific Research</span>
    <span class="use-case-pill">📊 Business Intelligence</span>
    <span class="use-case-pill">🏗️ Engineering</span>
  </div>
</div>

<!-- ══ Main ══ -->
<div class="container">

  <!-- Template panel -->
  <div class="section-label">Don't have a file yet?</div>
  <div class="tpl-panel">
    <p>Download a sample dataset, fill it with your real numbers, then upload it below.</p>
    <div class="tpl-row">
      <select class="tpl-select" id="tplSelect" onchange="updateTplDesc()">
        <option value="">— Choose a sample dataset —</option>
        <option value="bolletta_elettrica.csv"   data-desc="Discover the cost per kWh from your electricity bill">⚡ Electricity bill</option>
        <option value="consumo_benzina.csv"       data-desc="How much do you spend per km driven?">🚗 Fuel consumption</option>
        <option value="spesa_supermercato.csv"    data-desc="Estimate weekly grocery spend by household size">🛒 Grocery spend</option>
        <option value="rata_mutuo.csv"            data-desc="Find the formula for your monthly mortgage payment">🏦 Mortgage payment</option>
        <option value="risparmio_mensile.csv"     data-desc="How much can you save each month?">💧 Monthly savings</option>
        <option value="stipendio_esperienza.csv"  data-desc="How salary grows with years of experience">📈 Salary vs experience</option>
        <option value="bmi.csv"                   data-desc="Body mass index: the weight/height² law">⚕ BMI</option>
        <option value="calorie_camminata.csv"     data-desc="Calories burned walking — weight and distance">🔥 Calories walking</option>
        <option value="media_voti.csv"            data-desc="Calculate your grade average">📚 Grade average</option>
        <option value="studio_vs_voto.csv"        data-desc="How much study time is needed for a good grade?">⏱ Study vs grade</option>
        <option value="consumo_acqua.csv"         data-desc="Litres of water used by household size">💧 Water consumption</option>
        <option value="gas_riscaldamento.csv"     data-desc="Heating gas usage vs outside temperature">🌡 Gas heating</option>
      </select>
      <button class="tpl-btn" onclick="downloadTemplate()">↓ Download sample</button>
    </div>
    <div class="tpl-desc" id="tplDesc"></div>
  </div>

  <!-- Upload -->
  <div class="section-label">Upload your data</div>
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fi').click()">
    <div class="upload-icon">📂</div>
    <h2>Drop your CSV file here</h2>
    <p>or click to browse your files</p>
    <input type="file" id="fi" accept=".csv">
    <label class="upload-btn" onclick="event.stopPropagation();document.getElementById('fi').click()">Choose File</label>
    <p id="fn"></p>
  </div>

  <!-- How it works accordion -->
  <div class="how-box">
    <button class="how-toggle" id="howToggle" onclick="toggleHow()">
      💡 HOW TO USE FORMULA FINDER
      <span class="how-arrow" id="howArrow">▼</span>
    </button>
    <div class="how-body" id="howBody">
      <div class="how-grid">
        <div class="how-card">
          <div class="how-step">STEP 1 — PREPARE YOUR FILE</div>
          <p>Save your spreadsheet as a <strong>.csv file</strong>.<br>
          In Excel: <em>File → Save As → CSV UTF-8</em>.<br>
          First row must be <strong>column names</strong>. All values must be <strong>numeric</strong>.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 2 — CHOOSE TARGET Y</div>
          <p><strong>Y</strong> is the value you want to <strong>discover the law for</strong><br>
          (e.g. price, energy output, rate, score).<br>
          Auto-detected if your column is named <code>Y</code>, <code>target</code> or <code>output</code>.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 3 — SELECT DRIVERS X</div>
          <p><strong>X columns</strong> are the variables that <strong>drive Y</strong><br>
          (e.g. time, temperature, quantity, pressure).<br>
          Hold <strong>Cmd ⌘</strong> (Mac) or <strong>Ctrl</strong> (Win) to select multiple.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 4 — DISCOVER &amp; EXPLORE</div>
          <p><strong>Both</strong> = maximum accuracy (recommended).<br>
          <strong>Quick</strong> = fast scan of common laws.<br>
          Then <strong>predict</strong>, see <strong>which driver matters most</strong>, get a <strong>plain English</strong> explanation.</p>
        </div>
      </div>
      <div class="how-warn">
        ⚠ <strong>Excel &amp; semicolons:</strong> Italian/German/French Excel may use <strong>;</strong> as separator.
        Fix: choose <em>CSV UTF-8 (comma delimited)</em> when saving, or replace <code>;</code> with <code>,</code> in a text editor.<br><br>
        ⚠ <strong>Column named "Y":</strong> rename it (e.g. <code>coord_y</code>) before uploading to avoid auto-selection conflicts.
      </div>
    </div>
  </div>

  <!-- Preview -->
  <div class="preview-box" id="previewBox">
    <h4>📊 FILE PREVIEW</h4>
    <div id="previewTable"></div>
    <div class="preview-meta" id="previewMeta"></div>
  </div>

  <!-- Configure columns -->
  <div class="col-select" id="colSel">
    <h3>Configure your analysis</h3>
    <div class="col-row">
      <div class="col-group">
        <label>🎯 Target Y</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The variable you want to find the law for</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>📐 Drivers X</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Cmd/Ctrl to select multiple · Y is excluded automatically</span>
      </div>
      <div class="col-group">
        <label>⚙ Method</label>
        <select id="method" onchange="updatePreviewLabel()">
          <option value="both">Both (Quick Scan + Adam)</option>
          <option value="quick">Quick Scan only</option>
          <option value="adam">Adam only</option>
        </select>
        <span class="hint">Both = maximum discovery power</span>
      </div>
    </div>
    <div class="formula-preview" id="selPreview">Select Y and X columns above to preview your setup.</div>
    <div class="val-msg" id="valMsg"></div>

    <!-- Data Quality Gate -->
    <div class="dq-blocker" id="dqBlocker"></div>
    <div class="dq-warning" id="dqWarning"></div>
    <div class="dq-panel" id="dqPanel">
      <div class="dq-title">DATA QUALITY CHECK</div>
      <div id="dqRows"></div>
      <div class="dq-score-bar"><div class="dq-score-fill" id="dqFill"></div></div>
      <div class="dq-score-label" id="dqScoreLabel"></div>
    </div>

    <button type="button" class="run-btn" id="runBtn" onclick="run()">🔍 DISCOVER THE LAW</button>
  </div>

  <div class="spinner" id="spin">Analysing your data&hellip;</div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- ══ Results ══ -->
  <div class="results" id="res">

    <!-- Discovery banner -->
    <div class="discovery-banner">
      <div class="discovery-label">⚛ LAW DISCOVERED</div>
      <div class="discovery-formula" id="bf"></div>
      <button class="expand-btn" id="expandBtn" onclick="toggleFormula()" style="display:none">▼ Show full formula</button>
      <div class="discovery-meta" id="br"></div>
      <div class="result-actions">
        <button class="action-btn" id="copyBtn" onclick="copyFormula()">📋 Copy formula</button>
        <button class="action-btn" onclick="exportCSV()">↓ Export results CSV</button>
      </div>
    </div>

    <!-- Predict · Sensitivity · Explain -->
    <div class="action-panels" id="actionPanels">

      <div class="panel">
        <h3>Simulate a new scenario</h3>
        <div class="input-grid" id="predictInputs"></div>
        <button class="panel-btn" id="predictBtn" onclick="runPredict()">Calculate</button>
        <div style="margin-top:14px">
          <div class="predict-result" id="predictResult" style="display:none"></div>
          <div class="predict-label" id="predictLabel"></div>
        </div>
      </div>

      <div class="panel">
        <h3>Variable impact</h3>
        <div style="font-size:.72rem;color:var(--silver);margin-bottom:12px">
          Which driver moves Y the most? (at current input values)
        </div>
        <div id="sensResult"><span style="color:var(--silver);font-size:.78rem">Run a simulation to see impact.</span></div>
      </div>

      <div class="panel" style="flex:1.2">
        <h3>Plain English explanation</h3>
        <button class="panel-btn" id="explainBtn" onclick="runExplain()">Explain this law</button>
        <div style="margin-top:14px">
          <div class="explain-box" id="explainResult" style="display:none"></div>
        </div>
      </div>

    </div>

    <!-- Chart -->
    <div class="chart-box">
      <h3>OBSERVED vs MODEL — RESIDUALS</h3>
      <div id="chartWrap"></div>
    </div>

    <!-- Top terms -->
    <div class="terms-box">
      <h3>KEY DRIVERS — Adam weights</h3>
      <div id="tl"></div>
    </div>

    <!-- Quick search cards -->
    <div class="cards-section">
      <h3>CANDIDATE LAWS — Quick Scan</h3>
      <div class="cards-grid" id="cg"></div>
    </div>

  </div><!-- /results -->

</div><!-- /container -->

<footer>Formula Finder &nbsp;·&nbsp; Discovery Engine &nbsp;·&nbsp; EML + Adam &nbsp;·&nbsp; FormulaFinder SRL</footer>

<!-- ══ JS ══ -->
<script>
var csv           = null;
var allCols       = [];
var parsedData    = null;
var lastResults   = null;
var formulaExpanded = false;
var MIN_ROWS_PER_X  = 10;

var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');

fi.addEventListener('change', function() {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});
dz.addEventListener('dragover',  function(e){ e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function(){  dz.classList.remove('dragover'); });
dz.addEventListener('drop', function(e){
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

// ── Parse file ──
function handleFile(f) {
  if (!f) return;
  document.getElementById('fn').textContent = '✅ ' + f.name;
  Papa.parse(f, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    complete: function(result) {
      if (!result.data || result.data.length === 0) {
        document.getElementById('fn').textContent = '❌ Could not parse file. Check format.';
        return;
      }
      parsedData = result;
      csv        = Papa.unparse(result.data);
      allCols    = result.meta.fields || [];
      showPreview(result);
      parseCols();
      setTimeout(runDataQualityCheck, 100);
    },
    error: function(err) {
      document.getElementById('fn').textContent = '❌ Parse error: ' + err.message;
    }
  });
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showPreview(result) {
  var box  = document.getElementById('previewBox');
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
  meta.textContent = result.data.length + ' rows · ' + cols.length + ' columns detected' +
    (result.meta.delimiter !== ',' ? '  ⚠️ Separator: "' + result.meta.delimiter + '" (auto-fixed)' : '');
  box.style.display = 'block';
}

function parseCols() {
  var yHints = ['y','Y','target','output','result','out','label','response','dep'];
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
  setTimeout(runDataQualityCheck, 50);
  var yc = document.getElementById('yCol').value;
  var xE = document.getElementById('xCols');
  var prevSelected = Array.from(xE.selectedOptions).map(function(o){ return o.value; });
  xE.innerHTML = '';
  allCols.forEach(function(col) {
    if (col === yc) return;
    var opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    opt.selected = (prevSelected.length === 0 || prevSelected.indexOf(col) !== -1);
    xE.appendChild(opt);
  });
  hideValMsg();
  updatePreviewLabel();
}

function updatePreviewLabel() {
  setTimeout(runDataQualityCheck, 50);
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  var el = document.getElementById('selPreview');
  if (!yc || xc.length === 0) {
    el.textContent = 'Select Y and X columns above to preview your setup.';
    return;
  }
  el.textContent = 'Ready to discover: ' + yc + ' = f( ' + xc.join(', ') + ' )';
}

// ── Data Quality Gate ──
function runDataQualityCheck() {
  var blocker = document.getElementById('dqBlocker');
  var warning = document.getElementById('dqWarning');
  var panel   = document.getElementById('dqPanel');
  var runBtn  = document.getElementById('runBtn');
  blocker.style.display = 'none';
  warning.style.display = 'none';
  panel.style.display   = 'none';
  runBtn.disabled       = false;
  if (!parsedData || !parsedData.data) return;
  var yc   = document.getElementById('yCol').value;
  var xc   = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  var rows = parsedData.data;
  var n    = rows.length;
  var checks = [], score = 0, total = 0, blockers = [], warnings = [];

  // CHECK 1 — Volume
  total++;
  var minRows = Math.max(10, MIN_ROWS_PER_X * xc.length);
  if (n >= minRows) {
    checks.push({icon:'✓', cls:'dq-ok', label:'Data volume', value: n + ' rows — sufficient'});
    score++;
  } else if (n >= Math.floor(minRows * 0.6)) {
    checks.push({icon:'⚠', cls:'dq-warn', label:'Data volume', value: n + ' rows — minimum recommended: ' + minRows});
    warnings.push('Dataset has ' + n + ' rows. We recommend at least ' + minRows + ' for ' + xc.length + ' variable(s). Results may be approximate.');
    score += 0.5;
  } else {
    checks.push({icon:'✗', cls:'dq-err', label:'Data volume', value: n + ' rows — too few (min: ' + minRows + ')'});
    blockers.push('Not enough data: ' + n + ' rows found, minimum is ' + minRows + ' for ' + xc.length + ' variable(s).');
  }

  // CHECK 2 — Y variance
  total++;
  var yVals = rows.map(function(r){ return parseFloat(r[yc]); }).filter(function(v){ return !isNaN(v); });
  var yMean = yVals.reduce(function(a,b){return a+b;},0) / yVals.length;
  var yStd  = Math.sqrt(yVals.reduce(function(a,v){return a+(v-yMean)*(v-yMean);},0) / yVals.length);
  if (yStd > 1e-6) {
    checks.push({icon:'✓', cls:'dq-ok', label:'Target Y varies', value: 'Range: ' + Math.min.apply(null,yVals).toFixed(3) + ' → ' + Math.max.apply(null,yVals).toFixed(3)});
    score++;
  } else {
    checks.push({icon:'✗', cls:'dq-err', label:'Target Y varies', value: '"' + yc + '" is constant — nothing to discover'
    
