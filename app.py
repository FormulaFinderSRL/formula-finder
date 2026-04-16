#!/usr/bin/env python3
import numpy as np, pandas as pd, base64, io, warnings
warnings.filterwarnings("ignore")

class Adam:
    def __init__(self,lr=0.05,b1=0.9,b2=0.999,eps=1e-8):
        self.lr,self.b1,self.b2,self.eps=lr,b1,b2,eps
        self.m=self.v=None; self.t=0
    def step(self,w,g):
        if self.m is None: self.m=np.zeros_like(w,dtype=float); self.v=np.zeros_like(w,dtype=float)
        self.t+=1
        self.m=self.b1*self.m+(1-self.b1)*g; self.v=self.b2*self.v+(1-self.b2)*g**2
        mh=self.m/(1-self.b1**self.t); vh=self.v/(1-self.b2**self.t)
        return w-self.lr*mh/(np.sqrt(vh)+self.eps)

def eml(x,y,eps=1e-10):
    y_s=np.where(np.array(y)>0,np.array(y),eps)
    return np.exp(np.clip(np.array(x),-500,500))-np.log(y_s)

def build_features(X_dict):
    feats=[np.ones(len(list(X_dict.values())[0]))]; names=["1"]
    for v,x in X_dict.items():
        x=np.asarray(x,float); xc=np.clip(x,-15,15); xp=np.where(x>1e-6,x,1e-6)
        feats+=[x,x**2,x**3,np.exp(xc),np.exp(-xc),np.log(xp),np.sin(x),np.cos(x),np.tanh(x),np.sqrt(np.abs(x)),1/(1+x**2),x*np.exp(xc),x*np.sin(x)]
        names+=[v,v+"^2",v+"^3","exp("+v+")","exp(-"+v+")","ln("+v+")","sin("+v+")","cos("+v+")","tanh("+v+")","sqrt|"+v+"|","1/(1+"+v+"^2)",v+"*exp("+v+")",v+"*sin("+v+")"]
    vlist=list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1,len(vlist)):
            v1,v2=vlist[i],vlist[j]; x1=np.asarray(X_dict[v1],float); x2=np.asarray(X_dict[v2],float)
            feats+=[x1*x2,eml(x1,x2)]; names+=[v1+"*"+v2,"eml("+v1+","+v2+")"]
    return np.column_stack(feats),names

class EMLAdamRegressor:
    def __init__(self,lr=0.05,epochs=1500,l1=1e-3): self.lr,self.epochs,self.l1=lr,epochs,l1; self.w=self.names=None
    def fit(self,X_dict,y):
        F,self.names=build_features(X_dict); y=np.asarray(y,float)
        self.w,_,_,_=np.linalg.lstsq(F,y,rcond=None); opt=Adam(lr=self.lr)
        for _ in range(self.epochs):
            yp=F@self.w; g=(2/len(y))*(F.T@(yp-y))+self.l1*np.sign(self.w); self.w=opt.step(self.w,g)
        return self
    def predict(self,X_dict): F,_=build_features(X_dict); return F@self.w
    def r2(self,X_dict,y):
        yp=self.predict(X_dict); y=np.asarray(y,float)
        ss_r=np.sum((y-yp)**2); ss_t=np.sum((y-np.mean(y))**2)
        return float(1-ss_r/ss_t) if ss_t>0 else 1.0
    def formula(self,thr=0.05):
        terms=[]
        for n,w in zip(self.names,self.w):
            if abs(w)>thr: terms.append(("+%.3f" % w) if n=="1" else ("%.3f*" % w)+n)
        return "y = "+" ".join(terms).lstrip("+").strip() if terms else "y = 0"
    def top_terms(self,n=5):
        idx=np.argsort(np.abs(self.w))[::-1][:n]
        return [{"term":self.names[i],"weight":round(float(self.w[i]),4)} for i in idx if abs(self.w[i])>0.01]

def quick_search(X_dict,y,top_n=8,min_r2=0.5):
    ops={}
    for v,x in X_dict.items():
        ops["exp("+v+")"]  = lambda d,v=v: np.exp(np.clip(d[v],-500,500))
        ops["ln("+v+")"]   = lambda d,v=v: np.log(np.where(d[v]>0,d[v],1e-10))
        ops[v+"^2"]        = lambda d,v=v: d[v]**2
        ops[v+"^3"]        = lambda d,v=v: d[v]**3
        ops["sqrt("+v+")"] = lambda d,v=v: np.sqrt(np.abs(d[v]))
        ops["sin("+v+")"]  = lambda d,v=v: np.sin(d[v])
        ops["cos("+v+")"]  = lambda d,v=v: np.cos(d[v])
        ops[v]             = lambda d,v=v: d[v]
        ops["eml("+v+",1)"]= lambda d,v=v: eml(d[v],np.ones(len(d[v])))
    vlist=list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1,len(vlist)):
            v1,v2=vlist[i],vlist[j]; ops[v1+"*"+v2]=lambda d,v1=v1,v2=v2: d[v1]*d[v2]
    y=np.asarray(y,float); n=len(y); res=[]
    for nm,fn in ops.items():
        try:
            f=np.asarray(fn(X_dict),float)
            if np.any(np.isnan(f))|np.any(np.isinf(f)): continue
            A=np.column_stack([f,np.ones(n)]); c,_,_,_=np.linalg.lstsq(A,y,rcond=None)
            a,b=float(c[0]),float(c[1]); yp=a*f+b
            ss_r=np.sum((y-yp)**2); ss_t=np.sum((y-np.mean(y))**2)
            r2=float(1-ss_r/ss_t) if ss_t>0 else 1.0
            if r2>=min_r2:
                if abs(a-1)<0.005 and abs(b)<0.005: fs="y = "+nm
                elif abs(b)<0.005: fs="y = %.3f*%s" % (a,nm)
                else: fs="y = %.3f*%s + %.3f" % (a,nm,b)
                res.append({"formula":fs,"r2":round(r2,6),"accuracy":"%.4f%%" % (r2*100),
                    "quality":"PERFECT" if r2>0.9999 else "GREAT" if r2>0.99 else "GOOD"})
        except: pass
    res.sort(key=lambda d:d["r2"],reverse=True); return res[:top_n]

def make_chart_b64(X_dict, y_true, model=None):
    """
    Genera grafico base64. Usa solo la prima variabile X sull'asse X.
    Robusto: non crasha mai, restituisce stringa vuota in caso di errore.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vlist = list(X_dict.keys())
        x_vals = np.asarray(X_dict[vlist[0]], float)
        y_arr  = np.asarray(y_true, float)
        sidx   = np.argsort(x_vals)
        xs, ys = x_vals[sidx], y_arr[sidx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0D1B2A")

        # --- Pannello sinistro: Data vs Fit ---
        ax = axes[0]
        ax.set_facecolor("#112233")
        ax.scatter(xs, ys, color="#00e5ff", s=25, alpha=0.7, label="Data", zorder=5)
        if model is not None:
            try:
                yp = model.predict({vlist[0]: xs})
                r2 = model.r2(X_dict, y_arr)
                ax.plot(xs, yp, color="#ff6b6b", lw=2.5,
                        label="Fit R\u00B2=%.4f" % r2, zorder=4)
            except Exception:
                pass
        ax.set_title("Data vs Fit (%s)" % vlist[0], color="white", fontsize=11)
        ax.set_xlabel(vlist[0], color="#888")
        ax.set_ylabel("y", color="#888")
        ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#333355")

        # --- Pannello destro: Residui ---
        ax2 = axes[1]
        ax2.set_facecolor("#112233")
        if model is not None:
            try:
                res2 = ys - model.predict({vlist[0]: xs})
                ax2.bar(range(len(res2)), res2, color="#1B9AAA", alpha=0.7)
                ax2.axhline(0, color="#ff6b6b", lw=1.5, ls="--")
                ax2.set_title("Residuals", color="white", fontsize=11)
            except Exception:
                ax2.set_title("Residuals (unavailable)", color="#888", fontsize=11)
        else:
            ax2.set_title("Residuals (no model)", color="#888", fontsize=11)
        ax2.tick_params(colors="#888")
        for s in ax2.spines.values(): s.set_edgecolor("#333355")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#0D1B2A")
        plt.close(fig)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode()
        buf.close()
        return data

    except Exception as ex:
        # Genera un'immagine di errore minimale invece di restituire ""
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
            plt.savefig(buf2, format="png", dpi=80,
                        bbox_inches="tight", facecolor="#0D1B2A")
            plt.close(fig2)
            buf2.seek(0)
            return base64.b64encode(buf2.read()).decode()
        except Exception:
            return ""


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formula Finder</title>
<style>
:root{--bg:#0D1B2A;--mid:#112233;--card:#162840;--teal:#1B9AAA;--neon:#06D6A0;--amber:#FCD34D;--red:#EF4444;--white:#fff;--silver:#B0C4D8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--white);font-family:'Segoe UI',sans-serif;min-height:100vh}
header{background:var(--mid);border-bottom:2px solid var(--teal);padding:18px 32px;display:flex;align-items:center;gap:16px}
header h1{font-size:1.6rem;letter-spacing:3px}
header span{font-size:.75rem;color:var(--neon);border:1px solid var(--neon);border-radius:20px;padding:3px 12px}
.container{max-width:1100px;margin:0 auto;padding:32px 24px}
.upload-zone{border:2px dashed var(--teal);border-radius:12px;padding:48px;text-align:center;background:var(--card);margin-bottom:28px;transition:.2s}
.upload-zone.dragover{border-color:var(--neon);background:#1a2f4a}
.upload-zone h2{color:var(--teal);margin-bottom:8px}
.upload-zone p{color:var(--silver);font-size:.9rem;margin-top:8px}
#fi{display:none}
.upload-label{display:inline-block;margin-top:18px;background:var(--teal);color:var(--bg);border:none;padding:10px 28px;border-radius:6px;cursor:pointer;font-weight:700;font-size:1rem;transition:.2s}
.upload-label:hover{background:var(--neon)}
.col-select{display:none;background:var(--card);border-radius:12px;padding:24px;margin-bottom:24px;border:1px solid var(--teal)}
.col-select h3{color:var(--teal);margin-bottom:16px;letter-spacing:1px}
.col-row{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:4px}
.col-group label{color:var(--silver);font-size:.82rem;font-weight:600;letter-spacing:.5px}
select{background:var(--mid);color:var(--white);border:1px solid var(--teal);border-radius:6px;padding:8px 12px;font-size:.9rem;min-width:140px}
.hint{font-size:.75rem;color:var(--silver);margin-top:3px;opacity:.7}
.run-btn{background:var(--neon);color:var(--bg);border:none;padding:12px 36px;border-radius:6px;cursor:pointer;font-weight:700;font-size:1rem;margin-top:8px;letter-spacing:1px}
.run-btn:hover{background:var(--teal);color:#fff}
.run-btn:disabled{opacity:.5;cursor:not-allowed}
.spinner{display:none;text-align:center;padding:40px;color:var(--teal);font-size:1.1rem}
.results{display:none}
.best-box{background:linear-gradient(135deg,var(--teal),#0d7a85);border-radius:12px;padding:24px 28px;margin-bottom:24px}
.best-box h2{font-size:.85rem;letter-spacing:3px;color:var(--bg);margin-bottom:8px}
.best-formula{font-size:1.3rem;font-weight:700;color:var(--bg);font-family:monospace;margin-bottom:6px;word-break:break-all}
.best-r2{font-size:.95rem;color:rgba(0,0,0,.65)}
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--card);border:1px solid #1e3a5f;border-radius:10px;padding:18px;transition:.2s}
.card:hover{border-color:var(--teal)}
.card-rank{font-size:.75rem;color:var(--silver);letter-spacing:1px}
.card-formula{font-family:monospace;font-size:.9rem;color:var(--white);margin:6px 0;word-break:break-all}
.card-r2{font-size:.82rem}
.qp{color:var(--neon)}.qg{color:#06D6A0}.qb{color:var(--amber)}
.chart-box{background:var(--card);border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #1e3a5f}
.chart-box h3{color:var(--teal);margin-bottom:14px;letter-spacing:1px}
.chart-box img{width:100%;border-radius:8px;display:block}
.chart-err{color:var(--red);font-size:.9rem;padding:12px 0}
.terms-box{background:var(--card);border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #1e3a5f}
.terms-box h3{color:var(--teal);margin-bottom:14px}
.term-row{display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #1e3a5f}
.term-name{font-family:monospace;color:var(--white);min-width:140px;font-size:.85rem}
.term-bar-wrap{flex:1;margin:0 16px;height:8px;background:#1e3a5f;border-radius:4px;overflow:hidden}
.term-bar{height:100%;background:var(--teal);border-radius:4px;transition:.5s}
.term-w{color:var(--amber);font-size:.82rem;min-width:80px;text-align:right}
.error-box{background:#2a1018;border:1px solid var(--red);border-radius:10px;padding:16px;color:var(--red);margin-bottom:16px}
footer{text-align:center;padding:24px;color:var(--silver);font-size:.8rem;border-top:1px solid #1e3a5f;margin-top:40px}

/* How it works accordion */
.how-box{background:var(--card);border:1px solid #1e3a5f;border-radius:14px;margin-bottom:24px;overflow:hidden}
.how-toggle{width:100%;background:none;border:none;color:var(--teal);font-size:.88rem;font-weight:700;letter-spacing:1px;text-align:left;padding:14px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.how-toggle:hover{background:#1a2f4a}
.how-toggle .arrow{transition:transform .3s;font-style:normal;font-size:.75rem}
.how-toggle.open .arrow{transform:rotate(180deg)}
.how-body{display:none;padding:16px 20px 18px;border-top:1px solid #1e3a5f}
.how-body.open{display:block}
.how-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-bottom:14px}
.how-card{background:var(--mid);border-radius:10px;padding:12px 14px}
.how-card .step{font-size:.68rem;color:var(--neon);font-weight:700;letter-spacing:1px;margin-bottom:4px}
.how-card p{font-size:.8rem;color:var(--silver);line-height:1.5}
.how-card strong{color:var(--white)}
.how-warn{font-size:.78rem;color:var(--amber);background:#1e1a0a;border-left:3px solid var(--amber);padding:8px 12px;border-radius:6px;margin-top:4px;line-height:1.5}
</style>
</head>
<body>
<header>
  <h1>FORMULA FINDER</h1>
  <span>Powered by EML + Adam</span>
</header>
<div class="container">

  <div class="upload-zone" id="dropZone">
    <h2>Drop your CSV file here</h2>
    <p>or click the button below to choose a file</p>
    <input type="file" id="fi" accept=".csv">
    <label for="fi" class="upload-label">&#128193; Choose File</label>
    <p id="fn" style="margin-top:14px;color:var(--neon);font-weight:600;min-height:24px"></p>
  </div>

  <div class="col-select" id="colSel">
    <h3>CONFIGURE COLUMNS</h3>
    <div class="col-row">
      <div class="col-group">
        <label>&#127919; Target Y (dipendente)</label>
        <select id="yCol"></select>
        <span class="hint">La variabile da predire</span>
      </div>
      <div class="col-group">
        <label>&#128200; Variables X (indipendenti)</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Tieni Cmd/Ctrl per selezionarne pi&ugrave;</span>
      </div>
      <div class="col-group">
        <label>&#9881; Method</label>
        <select id="method">
          <option value="both">Both (Quick + Adam)</option>
          <option value="quick">Quick only</option>
          <option value="adam">Adam only</option>
        </select>
        <span class="hint">Both = pi&ugrave; accurato</span>
      </div>
    </div>
    <button type="button" class="run-btn" id="runBtn" onclick="run()">&#128269; FIND FORMULA</button>
  </div>

  <div class="spinner" id="spin">&#9881; Searching... please wait</div>
  <div class="error-box" id="err" style="display:none"></div>

  <div class="results" id="res">
    <div class="best-box">
      <h2>BEST FORMULA FOUND</h2>
      <div class="best-formula" id="bf"></div>
      <div class="best-r2" id="br"></div>
    </div>
    <div class="chart-box">
      <h3>DASHBOARD</h3>
      <div id="chartWrap"></div>
    </div>
    <div class="terms-box">
      <h3>TOP TERMS (Adam weights)</h3>
      <div id="tl"></div>
    </div>
    <div class="cards-grid" id="cg"></div>
  </div>

</div>
<footer>Formula Finder v3.2 &mdash; FormulaFinder S.R.L.</footer>

<script>
var csv = null;
var parsedCols = [];
var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');

// Upload via label nativa (zero JS per aprire il picker)
fi.addEventListener('change', function() {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});

// Drag & Drop
dz.addEventListener('dragover', function(e){ e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function(){ dz.classList.remove('dragover'); });
dz.addEventListener('drop', function(e){
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(f) {
  if (!f) return;
  document.getElementById('fn').textContent = '\u2705 ' + f.name;
  var reader = new FileReader();
  reader.onload = function(ev) { csv = ev.target.result; parseCols(csv); };
  reader.readAsText(f);
}

function parseCols(c) {
  // Rimuove BOM e whitespace, supporta \r\n e \n
  var firstLine = c.replace(/^\uFEFF/, '').trim().split(/\r?\n/)[0];
  parsedCols = firstLine.split(',').map(function(x){ return x.trim().replace(/^"|"$/g,''); });

  var yE = document.getElementById('yCol');
  var xE = document.getElementById('xCols');
  yE.innerHTML = '';
  xE.innerHTML = '';

  // Euristica: l'ultima colonna va in Y, le altre in X
  // MA: se una colonna si chiama 'y' o 'Y' o 'target' o 'output', quella viene messa in Y
  var yHints = ['y','Y','target','output','result','out','label','response'];
  var guessedY = parsedCols.length - 1; // default: ultima colonna
  for (var i = 0; i < parsedCols.length; i++) {
    if (yHints.indexOf(parsedCols[i]) !== -1) { guessedY = i; break; }
  }

  parsedCols.forEach(function(col, i) {
    var optY = document.createElement('option');
    optY.value = col; optY.textContent = col;
    if (i === guessedY) optY.selected = true;
    yE.appendChild(optY);

    var optX = document.createElement('option');
    optX.value = col; optX.textContent = col;
    if (i !== guessedY) optX.selected = true; // pre-seleziona tutte le non-Y
    xE.appendChild(optX);
  });

  document.getElementById('colSel').style.display = 'block';
}

async function run() {
  // Validazione
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  if (!csv) { alert('Carica prima un file CSV!'); return; }
  if (xc.length === 0) { alert('Seleziona almeno una variabile X!'); return; }
  if (xc.indexOf(yc) !== -1) { alert('Attenzione: Y \u00e8 incluso nelle X. Rimuovilo dalla lista X!'); return; }

  document.getElementById('spin').style.display  = 'block';
  document.getElementById('res').style.display   = 'none';
  document.getElementById('err').style.display   = 'none';
  document.getElementById('runBtn').disabled     = true;

  var m = document.getElementById('method').value;

  try {
    var resp = await fetch('/api/find', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({csv: csv, y_col: yc, x_cols: xc, method: m})
    });
    var d = await resp.json();
    if (!d.success) throw new Error(d.error);
    showResults(d);
  } catch(e) {
    document.getElementById('err').style.display  = 'block';
    document.getElementById('err').textContent    = '\u274c Error: ' + e.message;
  } finally {
    document.getElementById('spin').style.display  = 'none';
    document.getElementById('runBtn').disabled     = false;
  }
}

function showResults(d) {
  document.getElementById('res').style.display = 'block';
  var best = (d.quick_results && d.quick_results[0]) || {};
  document.getElementById('bf').textContent = d.adam_formula || best.formula || 'n/a';
  document.getElementById('br').textContent = 'Accuracy: ' +
    (d.adam_r2 ? (d.adam_r2 * 100).toFixed(4) + '%' : best.accuracy || 'n/a');

  // Chart — robusto
  var cw = document.getElementById('chartWrap');
  cw.innerHTML = '';
  if (d.chart_b64 && d.chart_b64.length > 100) {
    var img = document.createElement('img');
    img.src = 'data:image/png;base64,' + d.chart_b64;
    img.alt = 'Dashboard chart';
    img.style.width = '100%';
    img.style.borderRadius = '8px';
    img.onerror = function(){ cw.innerHTML = '<p class="chart-err">\u26a0\ufe0f Grafico non disponibile.</p>'; };
    cw.appendChild(img);
  } else {
    cw.innerHTML = '<p class="chart-err">\u26a0\ufe0f Grafico non disponibile (matplotlib potrebbe non essere installato sul server).</p>';
  }

  // Top terms
  var tl = document.getElementById('tl');
  tl.innerHTML = '';
  var wts = (d.top_terms || []).map(function(t){ return Math.abs(t.weight); });
  var mx  = wts.length ? Math.max.apply(null, wts) : 1;
  if ((d.top_terms || []).length === 0) {
    tl.innerHTML = '<p style="color:var(--silver);font-size:.85rem">Nessun termine significativo trovato.</p>';
  }
  (d.top_terms || []).forEach(function(t) {
    var pct = Math.min(100, Math.abs(t.weight) / mx * 100);
    tl.innerHTML += '<div class="term-row">'
      + '<span class="term-name">' + t.term + '</span>'
      + '<div class="term-bar-wrap"><div class="term-bar" style="width:' + pct + '%"></div></div>'
      + '<span class="term-w">' + t.weight.toFixed(4) + '</span>'
      + '</div>';
  });

  // Cards
  var cg = document.getElementById('cg');
  cg.innerHTML = '';
  if ((d.quick_results || []).length === 0) {
    cg.innerHTML = '<p style="color:var(--silver)">Nessun risultato Quick Search (prova a abbassare il min R\u00B2).</p>';
  }
  (d.quick_results || []).forEach(function(r, i) {
    var cls = r.quality === 'PERFECT' ? 'qp' : r.quality === 'GREAT' ? 'qg' : 'qb';
    cg.innerHTML += '<div class="card">'
      + '<div class="card-rank">#' + (i+1) + ' &nbsp; <span class="' + cls + '">' + r.quality + '</span></div>'
      + '<div class="card-formula">' + r.formula + '</div>'
      + '<div class="card-r2 ' + cls + '">R\u00B2 = ' + r.r2.toFixed(6) + ' &nbsp; (' + r.accuracy + ')</div>'
      + '</div>';
  });
}
function toggleHow() {
  var body   = document.getElementById('howBody');
  var toggle = document.getElementById('howToggle');
  var arrow  = document.getElementById('howArrow');
  var open   = body.classList.toggle('open');
  toggle.classList.toggle('open', open);
}

</script>
</body>
</html>"""


def create_app():
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    @app.route('/')
    def home():
        return Response(HTML_PAGE, mimetype='text/html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'version': '3.2'})

    @app.route('/api/find', methods=['POST'])
    def api_find():
        try:
            body   = request.get_json()
            df     = pd.read_csv(io.StringIO(body['csv']))
            X_dict = {c: df[c].astype(float).values for c in body['x_cols']}
            y_data = df[body['y_col']].astype(float).values
            method = body.get('method', 'both')
            result = {'success': True}
            if method in ('quick', 'both'):
                result['quick_results'] = quick_search(X_dict, y_data, top_n=8)
            if method in ('adam', 'both'):
                model = EMLAdamRegressor(lr=0.05, epochs=1200, l1=5e-4).fit(X_dict, y_data)
                result['adam_formula'] = model.formula(thr=0.05)
                result['adam_r2']      = round(model.r2(X_dict, y_data), 6)
                result['top_terms']    = model.top_terms(8)
                result['chart_b64']    = make_chart_b64(X_dict, y_data, model)
            else:
                result['chart_b64'] = make_chart_b64(X_dict, y_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    @app.route('/api/json', methods=['POST'])
    def api_json():
        try:
            body   = request.get_json()
            X_dict = {k: np.array(v, float) for k, v in body['X'].items()}
            y_data = np.array(body['y'], float)
            method = body.get('method', 'quick')
            result = {'success': True}
            if method in ('quick', 'both'):
                result['quick_results'] = quick_search(X_dict, y_data, top_n=8)
            if method in ('adam', 'both'):
                model = EMLAdamRegressor(lr=0.05, epochs=1000, l1=5e-4).fit(X_dict, y_data)
                result['adam_formula'] = model.formula()
                result['adam_r2']      = round(model.r2(X_dict, y_data), 6)
                result['top_terms']    = model.top_terms(5)
            return jsonify(result)
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
    app = create_app()
    if app:
        app.run(host=args.host, port=args.port, debug=False)
