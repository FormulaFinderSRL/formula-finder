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
            if abs(w)>thr:
                if n=="1":
                    terms.append("%.3f" % w)
                else:
                    terms.append("%.3f*%s" % (w, n))
        if not terms: return "y = 0"
        result = "y = " + terms[0]
        for t in terms[1:]:
            if t.startswith("-"):
                result += " - " + t[1:]
            else:
                result += " + " + t
        return result
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
                if abs(a-1)<0.005 and abs(b)<0.005:
                    fs="y = "+nm
                elif abs(b)<0.005:
                    fs="y = %.3f*%s" % (a,nm)
                elif b<0:
                    fs="y = %.3f*%s - %.3f" % (a,nm,abs(b))
                else:
                    fs="y = %.3f*%s + %.3f" % (a,nm,b)
                res.append({"formula":fs,"r2":round(r2,6),"accuracy":"%.4f%%" % (r2*100),
                    "quality":"PERFECT" if r2>0.9999 else "GREAT" if r2>0.99 else "GOOD"})
        except: pass
    res.sort(key=lambda d:d["r2"],reverse=True); return res[:top_n]

def make_chart_b64(X_dict, y_true, model=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vlist  = list(X_dict.keys())
        x_vals = np.asarray(X_dict[vlist[0]], float)
        y_arr  = np.asarray(y_true, float)
        sidx   = np.argsort(x_vals)
        xs, ys = x_vals[sidx], y_arr[sidx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0D1B2A")

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
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0D1B2A")
        plt.close(fig)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode()
        buf.close()
        return data
    except Exception as ex:
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
header{background:var(--mid);border-bottom:2px solid var(--teal);padding:16px 32px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:1.5rem;letter-spacing:3px}
header span{font-size:.72rem;color:var(--neon);border:1px solid var(--neon);border-radius:20px;padding:3px 12px}
.container{max-width:1100px;margin:0 auto;padding:28px 20px}

/* Upload zone */
.upload-zone{border:2px dashed var(--teal);border-radius:14px;padding:44px 32px;text-align:center;background:var(--card);margin-bottom:24px;transition:.2s}
.upload-zone.dragover{border-color:var(--neon);background:#1a2f4a}
.upload-zone h2{color:var(--teal);margin-bottom:6px;font-size:1.3rem}
.upload-zone p{color:var(--silver);font-size:.88rem;margin-top:6px}
#fi{display:none}
.upload-label{display:inline-block;margin-top:16px;background:var(--teal);color:var(--bg);padding:11px 30px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;transition:.2s;letter-spacing:.5px}
.upload-label:hover{background:var(--neon)}
#fn{margin-top:12px;color:var(--neon);font-weight:600;min-height:22px;font-size:.9rem}

/* Configure columns */
.col-select{display:none;background:var(--card);border-radius:14px;padding:24px;margin-bottom:20px;border:1px solid var(--teal)}
.col-select h3{color:var(--teal);margin-bottom:18px;letter-spacing:1px;font-size:1rem}
.col-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:6px;min-width:150px}
.col-group > label{color:var(--silver);font-size:.8rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase}
select{background:var(--mid);color:var(--white);border:1px solid #1e3a5f;border-radius:8px;padding:9px 12px;font-size:.88rem;width:100%;transition:.2s}
select:focus{outline:none;border-color:var(--teal)}
.hint{font-size:.72rem;color:var(--silver);opacity:.7;margin-top:2px}

/* Inline validation message (sostituisce alert) */
.val-msg{display:none;color:var(--red);font-size:.82rem;margin-top:8px;padding:8px 12px;background:#2a1018;border-radius:6px;border-left:3px solid var(--red)}

.run-btn{background:var(--neon);color:var(--bg);border:none;padding:13px 38px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;margin-top:4px;letter-spacing:1px;transition:.2s}
.run-btn:hover{background:var(--teal);color:#fff}
.run-btn:disabled{opacity:.5;cursor:not-allowed}

/* Spinner */
.spinner{display:none;text-align:center;padding:36px;color:var(--teal);font-size:1rem}
.spinner::after{content:'';display:inline-block;width:20px;height:20px;border:3px solid var(--teal);border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;margin-left:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* Results */
.results{display:none}
.best-box{background:linear-gradient(135deg,var(--teal),#0d7a85);border-radius:14px;padding:24px 28px;margin-bottom:20px}
.best-box h2{font-size:.8rem;letter-spacing:3px;color:rgba(0,0,0,.6);margin-bottom:10px;text-transform:uppercase}
.best-formula{font-size:1.1rem;font-weight:700;color:var(--bg);font-family:monospace;word-break:break-all;line-height:1.6;max-height:120px;overflow:hidden;transition:max-height .4s ease}
.best-formula.expanded{max-height:2000px}
.expand-btn{background:rgba(0,0,0,.15);border:none;color:var(--bg);font-size:.78rem;cursor:pointer;margin-top:8px;padding:4px 12px;border-radius:20px;font-weight:600}
.expand-btn:hover{background:rgba(0,0,0,.25)}
.best-r2{font-size:.92rem;color:rgba(0,0,0,.6);margin-top:8px}

/* Chart */
.chart-box{background:var(--card);border-radius:14px;padding:20px;margin-bottom:20px;border:1px solid #1e3a5f}
.chart-box h3{color:var(--teal);margin-bottom:12px;letter-spacing:1px;font-size:.95rem}
.chart-box img{width:100%;border-radius:8px;display:block}
.chart-err{color:var(--silver);font-size:.85rem;padding:20px;text-align:center;background:var(--mid);border-radius:8px}

/* Top terms */
.terms-box{background:var(--card);border-radius:14px;padding:20px;margin-bottom:20px;border:1px solid #1e3a5f}
.terms-box h3{color:var(--teal);margin-bottom:14px;letter-spacing:1px;font-size:.95rem}
.term-row{display:flex;align-items:center;padding:7px 0;border-bottom:1px solid #1a2f4a}
.term-name{font-family:monospace;color:var(--white);min-width:150px;font-size:.82rem}
.term-bar-wrap{flex:1;margin:0 14px;height:7px;background:#1a2f4a;border-radius:4px;overflow:hidden}
.term-bar{height:100%;background:linear-gradient(90deg,var(--teal),var(--neon));border-radius:4px;transition:width .6s ease}
.term-w{color:var(--amber);font-size:.8rem;min-width:90px;text-align:right;font-family:monospace}

/* Cards */
.cards-section h3{color:var(--teal);margin-bottom:14px;letter-spacing:1px;font-size:.95rem}
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin-bottom:24px}
.card{background:var(--card);border:1px solid #1e3a5f;border-radius:10px;padding:16px;transition:.2s;cursor:default}
.card:hover{border-color:var(--teal);transform:translateY(-2px)}
.card-rank{font-size:.72rem;color:var(--silver);letter-spacing:1px;margin-bottom:4px}
.card-formula{font-family:monospace;font-size:.88rem;color:var(--white);margin:6px 0;word-break:break-all;line-height:1.5}
.card-r2{font-size:.78rem;margin-top:4px}
.qp{color:var(--neon)}.qg{color:#2ecc71}.qb{color:var(--amber)}
.badge{display:inline-block;font-size:.68rem;padding:2px 8px;border-radius:20px;font-weight:700;margin-right:6px}
.badge-p{background:rgba(6,214,160,.15);color:var(--neon)}
.badge-g{background:rgba(46,204,113,.15);color:#2ecc71}
.badge-b{background:rgba(252,211,77,.15);color:var(--amber)}

/* Error */
.error-box{background:#2a1018;border:1px solid var(--red);border-radius:10px;padding:14px 16px;color:var(--red);margin-bottom:16px;font-size:.88rem}

footer{text-align:center;padding:24px;color:var(--silver);font-size:.78rem;border-top:1px solid #1e3a5f;margin-top:40px}

@media(max-width:600px){
  header h1{font-size:1.1rem}
  .col-row{flex-direction:column}
  .col-group{min-width:100%}
  .best-formula{font-size:.95rem}
}
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
    <h2>Drop your CSV file here</h2>
    <p>or click the button below to browse</p>
    <input type="file" id="fi" accept=".csv">
    <label for="fi" class="upload-label">&#128193; Choose File</label>
    <p id="fn"></p>
  </div>

  <!-- Configure Columns -->
  <div class="col-select" id="colSel">
    <h3>CONFIGURE COLUMNS</h3>
    <div class="col-row">
      <div class="col-group">
        <label>&#127919; Target Y</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">La variabile da predire</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>&#128200; Variables X</label>
        <select id="xCols" multiple style="height:100px"></select>
        <span class="hint">Cmd/Ctrl per selezionarne pi&ugrave; di una</span>
      </div>
      <div class="col-group">
        <label>&#9881; Method</label>
        <select id="method">
          <option value="both">Both (Quick + Adam)</option>
          <option value="quick">Quick only</option>
          <option value="adam">Adam only</option>
        </select>
        <span class="hint">Both = massima accuratezza</span>
      </div>
    </div>
    <div class="val-msg" id="valMsg"></div>
    <button type="button" class="run-btn" id="runBtn" onclick="run()">&#128269; FIND FORMULA</button>
  </div>

  <div class="spinner" id="spin">Searching&hellip; please wait</div>
  <div class="error-box" id="err" style="display:none"></div>

  <div class="results" id="res">
    <div class="best-box">
      <h2>Best Formula Found</h2>
      <div class="best-formula" id="bf"></div>
      <button class="expand-btn" id="expandBtn" onclick="toggleFormula()" style="display:none">&#9660; Show full formula</button>
      <div class="best-r2" id="br"></div>
    </div>
    <div class="chart-box">
      <h3>DASHBOARD</h3>
      <div id="chartWrap"></div>
    </div>
    <div class="terms-box">
      <h3>TOP TERMS &mdash; Adam weights</h3>
      <div id="tl"></div>
    </div>
    <div class="cards-section">
      <h3>QUICK SEARCH RESULTS</h3>
      <div class="cards-grid" id="cg"></div>
    </div>
  </div>

</div>
<footer>Formula Finder v3.3 &mdash; FormulaFinder S.R.L.</footer>

<script>
var csv        = null;
var allCols    = [];
var formulaExpanded = false;

var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');

// --- Upload via label nativa ---
fi.addEventListener('change', function() {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});

// --- Drag & Drop ---
dz.addEventListener('dragover',  function(e){ e.preventDefault(); dz.classList.add('dragover'); });
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
  var firstLine = c.replace(/^\uFEFF/, '').trim().split(/\r?\n/)[0];
  allCols = firstLine.split(',').map(function(x){ return x.trim().replace(/^"|"$/g,''); });

  // Euristica: trova la colonna Y più probabile
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

// Sincronizza xCols escludendo automaticamente la colonna Y selezionata
function syncXcols() {
  var yc  = document.getElementById('yCol').value;
  var xE  = document.getElementById('xCols');
  var prevSelected = Array.from(xE.selectedOptions).map(function(o){ return o.value; });

  xE.innerHTML = '';
  allCols.forEach(function(col) {
    if (col === yc) return; // Y NON compare nelle X, punto
    var opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    // Mantieni selezione precedente, o pre-seleziona tutto
    opt.selected = (prevSelected.length === 0 || prevSelected.indexOf(col) !== -1);
    xE.appendChild(opt);
  });
  hideValMsg();
}

function showValMsg(msg) {
  var el = document.getElementById('valMsg');
  el.textContent = '\u26a0\ufe0f ' + msg;
  el.style.display = 'block';
}
function hideValMsg() {
  document.getElementById('valMsg').style.display = 'none';
}

function toggleFormula() {
  var bf  = document.getElementById('bf');
  var btn = document.getElementById('expandBtn');
  formulaExpanded = !formulaExpanded;
  bf.classList.toggle('expanded', formulaExpanded);
  btn.textContent = formulaExpanded ? '\u25b2 Collapse formula' : '\u25bc Show full formula';
}

async function run() {
  hideValMsg();
  if (!csv)        { showValMsg('Carica prima un file CSV!'); return; }
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  if (xc.length === 0) { showValMsg('Seleziona almeno una variabile X.'); return; }

  document.getElementById('spin').style.display  = 'block';
  document.getElementById('res').style.display   = 'none';
  document.getElementById('err').style.display   = 'none';
  document.getElementById('runBtn').disabled      = true;
  formulaExpanded = false;

  var m = document.getElementById('method').value;
  try {
    var resp = await fetch('/api/find', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({csv:csv, y_col:yc, x_cols:xc, method:m})
    });
    var d = await resp.json();
    if (!d.success) throw new Error(d.error);
    showResults(d);
  } catch(e) {
    document.getElementById('err').style.display = 'block';
    document.getElementById('err').textContent   = '\u274c ' + e.message;
  } finally {
    document.getElementById('spin').style.display = 'none';
    document.getElementById('runBtn').disabled    = false;
  }
}

function showResults(d) {
  document.getElementById('res').style.display = 'block';

  // Best formula
  var formula = d.adam_formula || ((d.quick_results && d.quick_results[0]) || {}).formula || 'n/a';
  var bf  = document.getElementById('bf');
  var btn = document.getElementById('expandBtn');
  bf.textContent = formula;
  bf.classList.remove('expanded');
  // Mostra pulsante expand solo se la formula è lunga
  if (formula.length > 120) {
    btn.style.display = 'inline-block';
    btn.textContent   = '\u25bc Show full formula';
  } else {
    btn.style.display = 'none';
  }

  var acc = d.adam_r2
    ? (d.adam_r2 * 100).toFixed(4) + '%'
    : ((d.quick_results && d.quick_results[0]) || {}).accuracy || 'n/a';
  document.getElementById('br').textContent = 'Accuracy: ' + acc;

  // Chart
  var cw = document.getElementById('chartWrap');
  cw.innerHTML = '';
  if (d.chart_b64 && d.chart_b64.length > 200) {
    var img = document.createElement('img');
    img.src   = 'data:image/png;base64,' + d.chart_b64;
    img.alt   = 'Dashboard chart';
    img.onerror = function(){ cw.innerHTML = '<div class="chart-err">&#9888; Grafico non disponibile.</div>'; };
    cw.appendChild(img);
  } else {
    cw.innerHTML = '<div class="chart-err">&#9888; Grafico non disponibile (matplotlib potrebbe non girare sul server).</div>';
  }

  // Top terms
  var tl  = document.getElementById('tl');
  tl.innerHTML = '';
  var wts = (d.top_terms || []).map(function(t){ return Math.abs(t.weight); });
  var mx  = wts.length ? Math.max.apply(null, wts) : 1;
  if (!wts.length) {
    tl.innerHTML = '<p style="color:var(--silver);font-size:.82rem">Nessun termine significativo.</p>';
  }
  (d.top_terms || []).forEach(function(t) {
    var pct = Math.min(100, Math.abs(t.weight) / mx * 100).toFixed(1);
    tl.innerHTML +=
      '<div class="term-row">' +
      '<span class="term-name">' + t.term + '</span>' +
      '<div class="term-bar-wrap"><div class="term-bar" style="width:' + pct + '%"></div></div>' +
      '<span class="term-w">' + (t.weight > 0 ? '+' : '') + t.weight.toFixed(4) + '</span>' +
      '</div>';
  });

  // Cards
  var cg = document.getElementById('cg');
  cg.innerHTML = '';
  if (!(d.quick_results || []).length) {
    cg.innerHTML = '<p style="color:var(--silver);font-size:.85rem">Nessun risultato Quick Search.</p>';
  }
  (d.quick_results || []).forEach(function(r, i) {
    var cls   = r.quality === 'PERFECT' ? 'qp' : r.quality === 'GREAT' ? 'qg' : 'qb';
    var badgeCls = r.quality === 'PERFECT' ? 'badge-p' : r.quality === 'GREAT' ? 'badge-g' : 'badge-b';
    cg.innerHTML +=
      '<div class="card">' +
      '<div class="card-rank">' +
        '<span class="badge ' + badgeCls + '">' + r.quality + '</span>' +
        '#' + (i+1) +
      '</div>' +
      '<div class="card-formula">' + r.formula + '</div>' +
      '<div class="card-r2 ' + cls + '">R\u00B2 = ' + r.r2.toFixed(6) + ' &nbsp;&mdash;&nbsp; ' + r.accuracy + '</div>' +
      '</div>';
  });
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
        return jsonify({'status': 'ok', 'version': '3.3'})

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
