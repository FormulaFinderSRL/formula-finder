#!/usr/bin/env python3
import os, io, base64, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge


# ─────────────────────────── MATH CORE ───────────────────────────

class Adam:
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w, dtype=float)
            self.v = np.zeros_like(w, dtype=float)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g ** 2)
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
        x = np.asarray(x, float)
        xc = np.clip(x, -15, 15)
        xp = np.where(x > 1e-6, x, 1e-6)
        feats += [
            x, x**2, x**3,
            np.exp(xc), np.exp(-xc),
            np.log(xp),
            np.sin(x), np.cos(x), np.tanh(x),
            np.sqrt(np.abs(x)),
            1 / (1 + x**2),
            x * np.exp(xc),
            x * np.sin(x),
        ]
        names += [
            v, v+"^2", v+"^3",
            "exp("+v+")", "exp(-"+v+")",
            "ln("+v+")",
            "sin("+v+")", "cos("+v+")", "tanh("+v+")",
            "sqrt|"+v+"|",
            "1/(1+"+v+"^2)",
            v+"*exp("+v+")",
            v+"*sin("+v+")",
        ]
    vlist = list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            v1, v2 = vlist[i], vlist[j]
            x1 = np.asarray(X_dict[v1], float)
            x2 = np.asarray(X_dict[v2], float)
            feats += [x1 * x2, eml(x1, x2)]
            names += [v1 + "*" + v2, "eml(" + v1 + "," + v2 + ")"]
    return np.column_stack(feats), names


class EMLAdamRegressor:
    def __init__(self, lr=0.05, epochs=1500, l1=1e-3):
        self.lr, self.epochs, self.l1 = lr, epochs, l1
        self.w = None
        self.names = None

    def fit(self, X_dict, y):
        F, self.names = build_features(X_dict)
        y = np.asarray(y, float)
        self.w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        opt = Adam(lr=self.lr)
        for _ in range(self.epochs):
            yp = F @ self.w
            g = (2 / len(y)) * (F.T @ (yp - y)) + self.l1 * np.sign(self.w)
            self.w = opt.step(self.w, g)
        return self

    def predict(self, X_dict):
        F, _ = build_features(X_dict)
        return F @ self.w

    def r2(self, X_dict, y):
        yp = self.predict(X_dict)
        y = np.asarray(y, float)
        ss_r = np.sum((y - yp) ** 2)
        ss_t = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_r / ss_t) if ss_t > 0 else 1.0

    def formula(self, thr=0.05):
        terms = []
        for n, w in zip(self.names, self.w):
            if abs(w) > thr:
                if n == "1":
                    terms.append(f"{w:.3f}")
                else:
                    terms.append(f"{w:.3f}*{n}")
        if not terms:
            return "y = 0"
        result = "y = " + terms[0]
        for t in terms[1:]:
            result += (" - " + t[1:]) if t.startswith("-") else (" + " + t)
        return result

    def top_terms(self, n=5):
        idx = np.argsort(np.abs(self.w))[::-1][:n]
        return [{"term": self.names[i], "weight": round(float(self.w[i]), 4)}
                for i in idx if abs(self.w[i]) > 0.01]


def quick_search(X_dict, y, top_n=8, min_r2=0.5):
    ops = {}
    for v, x in X_dict.items():
        ops["exp("+v+")"]   = lambda d, v=v: np.exp(np.clip(d[v], -500, 500))
        ops["ln("+v+")"]    = lambda d, v=v: np.log(np.where(d[v] > 0, d[v], 1e-10))
        ops[v+"^2"]         = lambda d, v=v: d[v]**2
        ops[v+"^3"]         = lambda d, v=v: d[v]**3
        ops["sqrt("+v+")"]  = lambda d, v=v: np.sqrt(np.abs(d[v]))
        ops["sin("+v+")"]   = lambda d, v=v: np.sin(d[v])
        ops["cos("+v+")"]   = lambda d, v=v: np.cos(d[v])
        ops[v]              = lambda d, v=v: d[v]
        ops["eml("+v+",1)"] = lambda d, v=v: eml(d[v], np.ones(len(d[v])))

    vlist = list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            v1, v2 = vlist[i], vlist[j]
            ops[v1+"*"+v2] = lambda d, v1=v1, v2=v2: d[v1] * d[v2]

    y = np.asarray(y, float)
    n = len(y)
    res = []
    for nm, fn in ops.items():
        try:
            f = np.asarray(fn(X_dict), float)
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                continue
            A = np.column_stack([f, np.ones(n)])
            c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(c[0]), float(c[1])
            yp = a * f + b
            ss_r = np.sum((y - yp) ** 2)
            ss_t = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1 - ss_r / ss_t) if ss_t > 0 else 1.0
            if r2 >= min_r2:
                if abs(a - 1) < 0.005 and abs(b) < 0.005:
                    fs = "y = " + nm
                elif abs(b) < 0.005:
                    fs = "y = %.3f*%s" % (a, nm)
                elif b < 0:
                    fs = "y = %.3f*%s - %.3f" % (a, nm, abs(b))
                else:
                    fs = "y = %.3f*%s + %.3f" % (a, nm, b)

                res.append({
                    "formula": fs,
                    "r2": round(r2, 6),
                    "accuracy": "%.4f%%" % (r2 * 100),
                    "quality": "PERFECT" if r2 > 0.9999 else "GREAT" if r2 > 0.99 else "GOOD"
                })
        except Exception:
            pass
    res.sort(key=lambda d: d["r2"], reverse=True)
    return res[:top_n]


def make_chart_b64(X_dict, y_true, model=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vlist = list(X_dict.keys())
        x_vals = np.asarray(X_dict[vlist[0]], float)

        y_arr = np.asarray(y_true, float)
        sidx = np.argsort(x_vals)
        xs, ys = x_vals[sidx], y_arr[sidx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0D1B2A")

        ax = axes[0]
        ax.set_facecolor("#112233")
        ax.scatter(xs, ys, s=25, alpha=0.7, label="Data", zorder=5)

        if model is not None:
            try:
                yp = model.predict({vlist[0]: xs})
                r2 = model.r2(X_dict, y_arr)
                ax.plot(xs, yp, lw=2.5, label="Fit R²=%.4f" % r2, zorder=4)
            except Exception:
                pass

        ax.set_title("Data vs Fit (%s)" % vlist[0], color="white", fontsize=11)
        ax.set_xlabel(vlist[0], color="#888")
        ax.set_ylabel("y", color="#888")
        ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        for s in ax.spines.values():
            s.set_edgecolor("#333355")

        ax2 = axes[1]
        ax2.set_facecolor("#112233")
        if model is not None:
            try:
                res2 = ys - model.predict({vlist[0]: xs})
                ax2.bar(range(len(res2)), res2, alpha=0.7)
                ax2.axhline(0, lw=1.5, ls="--")
                ax2.set_title("Residuals", color="white", fontsize=11)
            except Exception:
                ax2.set_title("Residuals (unavailable)", color="#888", fontsize=11)
        else:
            ax2.set_title("Residuals (no model)", color="#888", fontsize=11)

        ax2.tick_params(colors="#888")
        for s in ax2.spines.values():
            s.set_edgecolor("#333355")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0D1B2A")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as ex:
        # Minimal fallback image with error text
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig2, ax3 = plt.subplots(figsize=(6, 2))
            fig2.patch.set_facecolor("#0D1B2A")
            ax3.set_facecolor("#0D1B2A")
            ax3.text(0.5, 0.5, "Chart error: %s" % str(ex),
                     color="#EF4444", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=9)
            ax3.axis("off")
            buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=80, bbox_inches="tight", facecolor="#0D1B2A")
            plt.close(fig2)
            buf2.seek(0)
            return base64.b64encode(buf2.read()).decode()
        except Exception:
            return ""


# ─────────────────────────── GEMINI EXPLAIN ───────────────────────────

def gemini_explain_formula(*, y_col, formula, r2, top_terms):
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("Gemini_API_Key") or "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        terms = top_terms or []
        terms_s = sorted(terms, key=lambda t: abs(float(t.get("weight", 0) or 0)), reverse=True)
        terms_txt = "\n".join([f"- {t.get('term')}: {t.get('weight')}" for t in terms_s[:8]])
        r2_pct = f"{float(r2) * 100:.2f}%" if isinstance(r2, (int, float)) else "n/a"

        prompt = f"""You are explaining a discovered formula to a Customer Support / Operations manager at a fintech.
They export CSVs from tools like Salesforce and Tableau.
Be clear, specific, and lightly witty (no sarcasm). Avoid sensitive/discriminatory examples.

Output format (exactly):
EN:
<max 140 words>

IT:
<max 140 parole>

Context:
- Target (Y): {y_col}
- R²: {r2_pct}
- Formula: {formula}
- Top terms (term: weight):
{terms_txt}

Include:
- What it means operationally
- One likely confounder / hidden driver
- One concrete next action for a support manager
"""
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        return text or None
    except Exception:
        return None


def _plain_english(top3, y_col):
    if not top3:
        return "No clear pattern found."
    parts = []
    for t in top3:
        dirn = "higher" if t["weight"] > 0 else "lower"
        parts.append("%s tends to make %s %s" % (t["term"], y_col, dirn))
    return "; ".join(parts) + "."


# ─────────────────────────── HTML PAGE ───────────────────────────
# IMPORTANT: we add a real upload to /upload and then use server-side parse.
# The UI stays the same.

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formula Finder</title>
<style>
/* (CSS unchanged – trimmed here for brevity in explanation; keep your original CSS) */
</style>
</head>
<body>
<header>
  <h1>FORMULA FINDER</h1>
  <span>Powered by EML + Adam</span>
</header>

<div class="container">
  <!-- Upload -->
  <div class="upload-zone" id="dropZone">
    <h2>&#128196; Drop your CSV file here</h2>
    <p>or click the button below to browse</p>
    <input type="file" id="fi" accept=".csv">
    <label for="fi" class="upload-label">&#128193; Choose File</label>
    <p id="fn"></p>
  </div>

  <!-- Configure Columns -->
  <div class="col-select" id="colSel" style="display:none">
    <h3>CONFIGURE COLUMNS</h3>
    <div class="col-row">
      <div class="col-group">
        <label>&#127919; Target Y</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The variable you want to predict</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>&#128200; Variables X</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Cmd/Ctrl to select multiple &bull; Y is automatically excluded</span>
      </div>
      <div class="col-group">
        <label>&#9881; Method</label>
        <select id="method">
          <option value="both">Both (Quick + Adam)</option>
          <option value="quick">Quick only</option>
          <option value="adam">Adam only</option>
        </select>
        <span class="hint">Both = maximum accuracy</span>
      </div>
    </div>

    <div class="formula-preview" id="selPreview">Select Y and X columns above to preview your setup.</div>
    <div class="val-msg" id="valMsg" style="display:none"></div>
    <button type="button" class="run-btn" id="runBtn" onclick="run()">&#128269; FIND FORMULA</button>
  </div>

  <div class="spinner" id="spin" style="display:none">Searching&hellip; please wait</div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- Results (keep your original results HTML if you want; minimal placeholder) -->
  <div class="results" id="res" style="display:none">
    <div class="best-box">
      <h2>Best Formula Found</h2>
      <div class="best-formula" id="bf"></div>
      <div class="best-r2" id="br"></div>
    </div>
    <div class="chart-box">
      <h3>DASHBOARD</h3>
      <div id="chartWrap"></div>
    </div>
  </div>
</div>

<script>
let uploadToken = null;   // server-side reference to uploaded file
let allCols = [];

const fi = document.getElementById('fi');
const dz = document.getElementById('dropZone');

fi.addEventListener('change', () => {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});

dz.addEventListener('dragover',  (e) => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', async (e) => {
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  if (!file) return;

  document.getElementById('fn').textContent = 'Uploading: ' + file.name + ' ...';
  hideValMsg();

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Upload failed');

    uploadToken = data.token;
    allCols = data.columns || [];

    document.getElementById('fn').textContent = '✅ ' + data.filename + ' (' + data.rows + ' rows, ' + data.columns.length + ' cols)';
    buildColumnSelectors();
  } catch (e) {
    document.getElementById('fn').textContent = '❌ Upload error: ' + e.message;
  }
}

function buildColumnSelectors() {
  if (!allCols.length) {
    showValMsg('No columns detected in CSV.');
    return;
  }

  const yHints = ['y','Y','target','output','result','out','label','response','dep'];
  let guessedY = allCols.length - 1;
  for (let i = 0; i < allCols.length; i++) {
    if (yHints.indexOf(allCols[i]) !== -1) { guessedY = i; break; }
  }

  const yE = document.getElementById('yCol');
  yE.innerHTML = '';
  allCols.forEach((col, i) => {
    const opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    if (i === guessedY) opt.selected = true;
    yE.appendChild(opt);
  });

  syncXcols();
  document.getElementById('colSel').style.display = 'block';
  updatePreviewLabel();
}

function syncXcols() {
  const yc = document.getElementById('yCol').value;
  const xE = document.getElementById('xCols');
  const prevSelected = Array.from(xE.selectedOptions).map(o => o.value);

  xE.innerHTML = '';
  allCols.forEach(col => {
    if (col === yc) return;
    const opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    opt.selected = (prevSelected.length === 0 || prevSelected.indexOf(col) !== -1);
    xE.appendChild(opt);
  });
  updatePreviewLabel();
}

function updatePreviewLabel() {
  const yc = document.getElementById('yCol').value;
  const xc = Array.from(document.getElementById('xCols').selectedOptions).map(o => o.value);
  const el = document.getElementById('selPreview');
  if (!yc || xc.length === 0) {
    el.textContent = 'Select Y and X columns above to preview your setup.';
    return;
  }
  el.textContent = 'Ready: predicting  Y = ' + yc + '  from  X = [' + xc.join(', ') + ']';
}

function showValMsg(msg) {
  const el = document.getElementById('valMsg');
  el.textContent = '⚠️ ' + msg;
  el.style.display = 'block';
}
function hideValMsg() {
  document.getElementById('valMsg').style.display = 'none';
}

async function run() {
  hideValMsg();
  if (!uploadToken) { showValMsg('Please upload a CSV file first!'); return; }

  const yc = document.getElementById('yCol').value;
  const xc = Array.from(document.getElementById('xCols').selectedOptions).map(o => o.value);
  if (!yc) { showValMsg('Select a target Y column.'); return; }
  if (!xc.length) { showValMsg('Select at least one X variable.'); return; }

  document.getElementById('spin').style.display = 'block';
  document.getElementById('res').style.display  = 'none';
  document.getElementById('err').style.display  = 'none';
  document.getElementById('runBtn').disabled    = true;

  try {
    const method = document.getElementById('method').value;

    const resp = await fetch('/api/find', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ token: uploadToken, y_col: yc, x_cols: xc, method })
    });
    const d = await resp.json();
    if (!d.success) throw new Error(d.error || 'analysis failed');

    document.getElementById('bf').textContent = d.adam_formula || (d.quick_results && d.quick_results[0] && d.quick_results[0].formula) || 'n/a';
    const acc = (typeof d.adam_r2 === 'number') ? (d.adam_r2 * 100).toFixed(4) + '%' :
                (d.quick_results && d.quick_results[0] && d.quick_results[0].accuracy) || 'n/a';
    document.getElementById('br').textContent = 'Accuracy (R²): ' + acc;

    const cw = document.getElementById('chartWrap');
    cw.innerHTML = '';
    if (d.chart_b64 && d.chart_b64.length > 200) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + d.chart_b64;
      img.style.width = '100%';
      img.style.borderRadius = '8px';
      cw.appendChild(img);
    } else {
      cw.textContent = 'Chart not available.';
    }

    document.getElementById('res').style.display = 'block';
  } catch (e) {
    const err = document.getElementById('err');
    err.style.display = 'block';
    err.textContent = '❌ ' + e.message;
  } finally {
    document.getElementById('spin').style.display = 'none';
    document.getElementById('runBtn').disabled = false;
  }
}
</script>
</body>
</html>
"""


# ─────────────────────────── SERVER HELPERS ───────────────────────────

def _read_csv_file_safely(path: str) -> pd.DataFrame:
    # robust encoding fallbacks
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        return pd.read_csv(path, engine="python")


def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Try to coerce all columns to numeric; keep those that become numeric enough
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # drop rows that are all NaN
    out = out.dropna(how="all")
    return out


# ─────────────────────────── APP FACTORY ───────────────────────────

def create_app():
    app = Flask(__name__)

    # Render/proxy safety
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return jsonify({"ok": False, "error": "File too large"}), 413

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "version": "definitive-upload"})

    @app.get("/")
    def home():
        return Response(HTML_PAGE, mimetype="text/html")

    # ✅ REAL upload endpoint (multipart/form-data)
    @app.post("/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file provided (field name must be 'file')."}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        filename = secure_filename(f.filename)
        tmp_dir = tempfile.gettempdir()
        token = f"ff_{next(tempfile._get_candidate_names())}"
        save_path = os.path.join(tmp_dir, token + "_" + filename)
        f.save(save_path)

        df = _read_csv_file_safely(save_path)
        df = _ensure_numeric_df(df)

        cols = [str(c) for c in df.columns.tolist()]
        rows = int(df.shape[0])

        # keep it server-side; client uses token
        return jsonify({
            "ok": True,
            "token": token,
            "filename": filename,
            "columns": cols,
            "rows": rows
        })

    # Analysis using token (server reads file saved in /tmp)
    @app.post("/api/find")
    def api_find():
        try:
            body = request.get_json(force=True) or {}
            token = body.get("token")
            y_col = body["y_col"]
            x_cols = body["x_cols"]
            method = body.get("method", "both")

            if not token:
                return jsonify({"success": False, "error": "Missing upload token. Upload a CSV first."}), 400

            # locate file by token in /tmp
            tmp_dir = tempfile.gettempdir()
            matches = [fn for fn in os.listdir(tmp_dir) if fn.startswith(token + "_")]
            if not matches:
                return jsonify({"success": False, "error": "Uploaded file expired. Upload again."}), 400
            path = os.path.join(tmp_dir, matches[0])

            df = _read_csv_file_safely(path)
            df = _ensure_numeric_df(df)

            # validate columns
            if y_col not in df.columns:
                return jsonify({"success": False, "error": f"y_col '{y_col}' not found in CSV."}), 400
            for c in x_cols:
                if c not in df.columns:
                    return jsonify({"success": False, "error": f"x_col '{c}' not found in CSV."}), 400

            # drop rows with NaN in needed cols
            needed = list(set([y_col] + list(x_cols)))
            df2 = df[needed].dropna()
            if df2.shape[0] < 5:
                return jsonify({"success": False, "error": "Not enough valid numeric rows after cleaning (need at least ~5)."}), 400

            X_dict = {c: df2[c].astype(float).values for c in x_cols}
            y_data = df2[y_col].astype(float).values

            result = {"success": True}

            if method in ("quick", "both"):
                result["quick_results"] = quick_search(X_dict, y_data, top_n=8)

            if method in ("adam", "both"):
                model = EMLAdamRegressor(lr=0.05, epochs=1200, l1=5e-4).fit(X_dict, y_data)
                result["adam_formula"] = model.formula(thr=0.05)
                result["adam_r2"] = round(model.r2(X_dict, y_data), 6)
                result["top_terms"] = model.top_terms(8)
                result["chart_b64"] = make_chart_b64(X_dict, y_data, model)
            else:
                result["chart_b64"] = make_chart_b64(X_dict, y_data)

            return jsonify(result)

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.post("/api/predict")
    def api_predict():
        try:
            body = request.get_json(force=True) or {}
            token = body.get("token")
            y_col = body["y_col"]
            x_cols = body["x_cols"]
            inputs = body["inputs"]

            if not token:
                return jsonify({"success": False, "error": "Missing upload token."}), 400

            tmp_dir = tempfile.gettempdir()
            matches = [fn for fn in os.listdir(tmp_dir) if fn.startswith(token + "_")]
            if not matches:
                return jsonify({"success": False, "error": "Uploaded file expired. Upload again."}), 400
            path = os.path.join(tmp_dir, matches[0])

            df = _ensure_numeric_df(_read_csv_file_safely(path))
            needed = list(set([y_col] + list(x_cols)))
            df2 = df[needed].dropna()
            X_dict = {c: df2[c].astype(float).values for c in x_cols}
            y_data = df2[y_col].astype(float).values

            model = EMLAdamRegressor(lr=0.05, epochs=1200, l1=5e-4).fit(X_dict, y_data)

            inp = {k: np.array([float(v)]) for k, v in inputs.items()}
            pred = float(model.predict(inp)[0])

            sens = {}
            for k in x_cols:
                base = {kk: np.array([float(inputs[kk])]) for kk in x_cols}
                delta = abs(float(inputs[k])) * 0.01 + 1e-6
                base_up = dict(base)
                base_up[k] = np.array([float(inputs[k]) + delta])
                sens[k] = round(float((model.predict(base_up)[0] - model.predict(base)[0]) / delta), 4)

            return jsonify({"success": True, "prediction": round(pred, 4), "sensitivity": sens})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.post("/api/explain")
    def api_explain():
        try:
            body = request.get_json(force=True) or {}
            terms = body.get("top_terms", [])
            formula = body.get("formula", "")
            r2 = body.get("r2", None)
            y_col = body.get("y_col", "Y")

            if not terms:
                return jsonify({"success": True, "explanation": "No significant terms found."})

            terms_s = sorted(terms, key=lambda t: abs(t["weight"]), reverse=True)
            total = sum(abs(t["weight"]) for t in terms_s) or 1

            lines = []
            for i, t in enumerate(terms_s[:5]):
                pct = round(abs(t["weight"]) / total * 100, 1)
                dirn = "increases" if t["weight"] > 0 else "decreases"
                lines.append("  %d. %s — %s %s by %.4f per unit (%.1f%% of total influence)" % (
                    i + 1, t["term"], t["term"], dirn, abs(t["weight"]), pct
                ))

            r2_str = ("The model explains %.2f%% of the variance in %s." % (float(r2) * 100, y_col)) if isinstance(r2, (int, float)) else ""
            explanation = (
                "Formula summary for %s:\n\n" % y_col +
                (r2_str + "\n\n" if r2_str else "") +
                "Key drivers (by importance):\n" +
                "\n".join(lines) +
                "\n\nIn plain English: " + _plain_english(terms_s[:3], y_col)
            )
            return jsonify({"success": True, "explanation": explanation})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.post("/api/explain_ai")
    def api_explain_ai():
        try:
            body = request.get_json(force=True) or {}
            terms = body.get("top_terms", [])
            formula = body.get("formula", "")
            r2 = body.get("r2", None)
            y_col = body.get("y_col", "Y")

            text = gemini_explain_formula(y_col=y_col, formula=formula, r2=r2, top_terms=terms)
            if text:
                return jsonify({"success": True, "explanation": text})

            return jsonify({
                "success": False,
                "error": "Gemini is not configured. Set GEMINI_API_KEY (or Gemini_API_Key) on the server."
            }), 400
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
