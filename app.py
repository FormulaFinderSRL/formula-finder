#!/usr/bin/env python3
import numpy as np, pandas as pd, base64, io, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── MATH CORE ───────────────────────────

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
                if n=="1": terms.append("%.3f" % w)
                else: terms.append("%.3f*%s" % (w,n))
        if not terms: return "y = 0"
        result="y = "+terms[0]
        for t in terms[1:]:
            result+=(" - "+t[1:]) if t.startswith("-") else (" + "+t)
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
                if abs(a-1)<0.005 and abs(b)<0.005: fs="y = "+nm
                elif abs(b)<0.005: fs="y = %.3f*%s" % (a,nm)
                elif b<0: fs="y = %.3f*%s - %.3f" % (a,nm,abs(b))
                else: fs="y = %.3f*%s + %.3f" % (a,nm,b)
                res.append({"formula":fs,"r2":round(r2,6),"accuracy":"%.4f%%" % (r2*100),
                    "quality":"PERFECT" if r2>0.9999 else "GREAT" if r2>0.99 else "GOOD"})
        except: pass
    res.sort(key=lambda d:d["r2"],reverse=True); return res[:top_n]

def make_chart_b64(X_dict, y_true, model=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        vlist=list(X_dict.keys()); x_vals=np.asarray(X_dict[vlist[0]],float)
        y_arr=np.asarray(y_true,float); sidx=np.argsort(x_vals); xs,ys=x_vals[sidx],y_arr[sidx]
        fig,axes=plt.subplots(1,2,figsize=(12,4)); fig.patch.set_facecolor("#0D1B2A")
        ax=axes[0]; ax.set_facecolor("#112233")
        ax.scatter(xs,ys,color="#00e5ff",s=25,alpha=0.7,label="Data",zorder=5)
        if model is not None:
            try:
                yp=model.predict({vlist[0]:xs}); r2=model.r2(X_dict,y_arr)
                ax.plot(xs,yp,color="#ff6b6b",lw=2.5,label="Fit R\u00B2=%.4f"%r2,zorder=4)
            except: pass
        ax.set_title("Data vs Fit (%s)"%vlist[0],color="white",fontsize=11)
        ax.set_xlabel(vlist[0],color="#888"); ax.set_ylabel("y",color="#888")
        ax.tick_params(colors="#888"); ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#333355")
        ax2=axes[1]; ax2.set_facecolor("#112233")
        if model is not None:
            try:
                res2=ys-model.predict({vlist[0]:xs}); ax2.bar(range(len(res2)),res2,color="#1B9AAA",alpha=0.7)
                ax2.axhline(0,color="#ff6b6b",lw=1.5,ls="--"); ax2.set_title("Residuals",color="white",fontsize=11)
            except: ax2.set_title("Residuals (unavailable)",color="#888",fontsize=11)
        else: ax2.set_title("Residuals (no model)",color="#888",fontsize=11)
        ax2.tick_params(colors="#888")
        for s in ax2.spines.values(): s.set_edgecolor("#333355")
        plt.tight_layout(); buf=io.BytesIO()
        plt.savefig(buf,format="png",dpi=100,bbox_inches="tight",facecolor="#0D1B2A")
        plt.close(fig); buf.seek(0); data=base64.b64encode(buf.read()).decode(); buf.close(); return data
    except Exception as ex:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig2,ax3=plt.subplots(figsize=(6,2)); fig2.patch.set_facecolor("#0D1B2A"); ax3.set_facecolor("#0D1B2A")
            ax3.text(0.5,0.5,"Chart error: %s"%str(ex),color="#EF4444",ha="center",va="center",transform=ax3.transAxes,fontsize=9)
            ax3.axis("off"); buf2=io.BytesIO()
            plt.savefig(buf2,format="png",dpi=80,bbox_inches="tight",facecolor="#0D1B2A")
            plt.close(fig2); buf2.seek(0); return base64.b64encode(buf2.read()).decode()
        except: return ""

# ─────────────────────────── HTML PAGE ───────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formula Finder</title>
<script>
/* Minimal robust CSV parser — handles BOM, quoted fields, auto-detects , ; \t */
var Papa = (function(){
  function detectDelim(str) {
    var s = str.slice(0, 2000);
    var counts = {',': 0, ';': 0, '\t': 0};
    var inQ = false;
    for (var i = 0; i < s.length; i++) {
      if (s[i] === '"') { inQ = !inQ; continue; }
      if (!inQ && counts[s[i]] !== undefined) counts[s[i]]++;
    }
    var best = ',', max = -1;
    Object.keys(counts).forEach(function(k){ if(counts[k]>max){max=counts[k];best=k;} });
    return best;
  }
  function parseRows(str, delim) {
    var rows = [], row = [], cur = '', inQ = false, i = 0;
    while (i < str.length) {
      var c = str[i];
      if (c === '"') {
        if (inQ && str[i+1] === '"') { cur += '"'; i += 2; continue; }
        inQ = !inQ; i++; continue;
      }
      if (!inQ && c === delim) { row.push(cur); cur = ''; i++; continue; }
      if (!inQ && (c === '\n' || (c === '\r' && str[i+1] === '\n'))) {
        row.push(cur); rows.push(row); row = []; cur = '';
        if (c === '\r') i++;
        i++; continue;
      }
      cur += c; i++;
    }
    if (cur || row.length) { row.push(cur); rows.push(row); }
    return rows;
  }
  return {
    parse: function(file, opts) {
      var reader = new FileReader();
      reader.onload = function(e) {
        try {
          var raw = e.target.result.replace(/^\uFEFF/, '');
          var delim = detectDelim(raw);
          var rows = parseRows(raw.trim(), delim);
          if (rows.length < 2) { opts.error && opts.error({message:'No data rows found'}); return; }
          var fields = rows[0].map(function(f){ return f.trim(); });
          var data = [];
          for (var i = 1; i < rows.length; i++) {
            if (rows[i].length === 1 && rows[i][0] === '') continue;
            var obj = {};
            fields.forEach(function(f, j){ obj[f] = (rows[i][j] || '').trim(); });
            data.push(obj);
          }
          opts.complete && opts.complete({data:data, meta:{fields:fields, delimiter:delim}});
        } catch(ex) { opts.error && opts.error({message: String(ex)}); }
      };
      reader.onerror = function(){ opts.error && opts.error({message:'FileReader error'}); };
      reader.readAsText(file);
    },
    unparse: function(data) {
      if (!data || !data.length) return '';
      var fields = Object.keys(data[0]);
      var lines = [fields.join(',')];
      data.forEach(function(r){ lines.push(fields.map(function(f){ return r[f]||''; }).join(',')); });
      return lines.join('\n');
    }
  };
})();
</script>
<style>
:root{--bg:#0D1B2A;--mid:#112233;--card:#162840;--teal:#1B9AAA;--neon:#06D6A0;--amber:#FCD34D;--red:#EF4444;--white:#fff;--silver:#B0C4D8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--white);font-family:'Segoe UI',sans-serif;min-height:100vh}
header{background:var(--mid);border-bottom:2px solid var(--teal);padding:16px 32px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:1.5rem;letter-spacing:3px}
header span{font-size:.72rem;color:var(--neon);border:1px solid var(--neon);border-radius:20px;padding:3px 12px}
.container{max-width:1100px;margin:0 auto;padding:28px 20px}

/* ── How it works accordion ── */
.how-box{background:var(--card);border:1px solid #1e3a5f;border-radius:14px;margin-bottom:24px;overflow:hidden}
.how-toggle{width:100%;background:none;border:none;color:var(--teal);font-size:.88rem;font-weight:700;letter-spacing:1px;text-align:left;padding:14px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.how-toggle:hover{background:#1a2f4a}
.how-arrow{transition:transform .3s;font-size:.75rem;display:inline-block}
.how-body{display:none;padding:16px 20px 18px;border-top:1px solid #1e3a5f}
.how-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-bottom:14px}
.how-card{background:var(--mid);border-radius:10px;padding:12px 14px}
.how-step{font-size:.68rem;color:var(--neon);font-weight:700;letter-spacing:1px;margin-bottom:5px}
.how-card p{font-size:.8rem;color:var(--silver);line-height:1.55}
.how-card strong{color:var(--white)}
.how-card code{background:#0d1b2a;color:var(--neon);padding:1px 5px;border-radius:3px;font-size:.78rem}
.how-warn{font-size:.78rem;color:var(--amber);background:#1e1a0a;border-left:3px solid var(--amber);padding:10px 14px;border-radius:6px;margin-top:4px;line-height:1.6}

/* ── Upload zone ── */
.upload-zone{border:2px dashed var(--teal);border-radius:14px;padding:44px 32px;text-align:center;background:var(--card);margin-bottom:24px;transition:.2s}
.upload-zone.dragover{border-color:var(--neon);background:#1a2f4a}
.upload-zone h2{color:var(--teal);margin-bottom:6px;font-size:1.3rem}
.upload-zone p{color:var(--silver);font-size:.88rem;margin-top:6px}
#fi{display:none}
.upload-label{display:inline-block;margin-top:16px;background:var(--teal);color:var(--bg);padding:11px 30px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;transition:.2s;letter-spacing:.5px}
.upload-label:hover{background:var(--neon)}
#fn{margin-top:12px;color:var(--neon);font-weight:600;min-height:22px;font-size:.9rem}

/* ── CSV mini-preview ── */
.preview-box{display:none;background:var(--mid);border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:20px;overflow-x:auto}
.preview-box h4{color:var(--teal);font-size:.78rem;letter-spacing:1px;margin-bottom:10px}
.preview-table{border-collapse:collapse;font-size:.75rem;width:100%}
.preview-table th{background:#0d1b2a;color:var(--neon);padding:5px 10px;text-align:left;border-bottom:1px solid #1e3a5f}
.preview-table td{color:var(--silver);padding:4px 10px;border-bottom:1px solid #162840}
.preview-table tr:last-child td{border-bottom:none}
.preview-meta{font-size:.72rem;color:var(--silver);margin-top:8px;opacity:.7}

/* ── Configure columns ── */
.col-select{display:none;background:var(--card);border-radius:14px;padding:24px;margin-bottom:20px;border:1px solid var(--teal)}
.col-select h3{color:var(--teal);margin-bottom:18px;letter-spacing:1px;font-size:1rem}
.col-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-start}
.col-group{display:flex;flex-direction:column;gap:6px;min-width:150px}
.col-group > label{color:var(--silver);font-size:.8rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase}
select{background:var(--mid);color:var(--white);border:1px solid #1e3a5f;border-radius:8px;padding:9px 12px;font-size:.88rem;width:100%;transition:.2s}
select:focus{outline:none;border-color:var(--teal)}
.hint{font-size:.72rem;color:var(--silver);opacity:.7;margin-top:2px}

/* ── Live formula preview ── */
.formula-preview{font-size:.8rem;color:var(--silver);background:var(--mid);border-radius:6px;padding:7px 12px;margin-top:10px;margin-bottom:4px;min-height:22px;font-family:monospace;border-left:3px solid var(--teal)}

/* ── Validation msg ── */
.val-msg{display:none;color:var(--red);font-size:.82rem;margin-top:8px;padding:8px 12px;background:#2a1018;border-radius:6px;border-left:3px solid var(--red)}

.run-btn{background:var(--neon);color:var(--bg);border:none;padding:13px 38px;border-radius:8px;cursor:pointer;font-weight:700;font-size:.95rem;margin-top:4px;letter-spacing:1px;transition:.2s}
.run-btn:hover{background:var(--teal);color:#fff}
.run-btn:disabled{opacity:.5;cursor:not-allowed}

/* ── Spinner ── */
.spinner{display:none;text-align:center;padding:36px;color:var(--teal);font-size:1rem}
.spinner::after{content:'';display:inline-block;width:20px;height:20px;border:3px solid var(--teal);border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;margin-left:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Results ── */
.results{display:none}
.best-box{background:linear-gradient(135deg,var(--teal),#0d7a85);border-radius:14px;padding:24px 28px;margin-bottom:20px}
.best-box h2{font-size:.8rem;letter-spacing:3px;color:rgba(0,0,0,.6);margin-bottom:10px;text-transform:uppercase}
.best-formula{font-size:1.1rem;font-weight:700;color:var(--bg);font-family:monospace;word-break:break-all;line-height:1.6;max-height:120px;overflow:hidden;transition:max-height .4s ease}
.best-formula.expanded{max-height:2000px}
.expand-btn{background:rgba(0,0,0,.15);border:none;color:var(--bg);font-size:.78rem;cursor:pointer;margin-top:8px;padding:4px 12px;border-radius:20px;font-weight:600}
.expand-btn:hover{background:rgba(0,0,0,.25)}
.best-r2{font-size:.92rem;color:rgba(0,0,0,.6);margin-top:8px}

/* Copy + Export buttons */
.result-actions{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
.action-btn{background:rgba(0,0,0,.18);border:1px solid rgba(0,0,0,.25);color:var(--bg);font-size:.78rem;font-weight:700;padding:6px 16px;border-radius:20px;cursor:pointer;transition:.2s;letter-spacing:.4px}
.action-btn:hover{background:rgba(0,0,0,.32)}
.action-btn.copied{background:rgba(0,0,0,.35)}

/* ── Chart ── */
.chart-box{background:var(--card);border-radius:14px;padding:20px;margin-bottom:20px;border:1px solid #1e3a5f}
.chart-box h3{color:var(--teal);margin-bottom:12px;letter-spacing:1px;font-size:.95rem}
.chart-box img{width:100%;border-radius:8px;display:block}
.chart-err{color:var(--silver);font-size:.85rem;padding:20px;text-align:center;background:var(--mid);border-radius:8px}

/* ── Top terms ── */
.terms-box{background:var(--card);border-radius:14px;padding:20px;margin-bottom:20px;border:1px solid #1e3a5f}
.terms-box h3{color:var(--teal);margin-bottom:14px;letter-spacing:1px;font-size:.95rem}
.term-row{display:flex;align-items:center;padding:7px 0;border-bottom:1px solid #1a2f4a}
.term-name{font-family:monospace;color:var(--white);min-width:150px;font-size:.82rem}
.term-bar-wrap{flex:1;margin:0 14px;height:7px;background:#1a2f4a;border-radius:4px;overflow:hidden}
.term-bar{height:100%;background:linear-gradient(90deg,var(--teal),var(--neon));border-radius:4px;transition:width .6s ease}
.term-w{color:var(--amber);font-size:.8rem;min-width:90px;text-align:right;font-family:monospace}

/* ── Cards ── */
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

/* ── Error ── */
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

  <!-- ── How it works accordion ── -->
  <div class="how-box">
    <button class="how-toggle" id="howToggle" onclick="toggleHow()">
      HOW TO USE FORMULA FINDER
      <span class="how-arrow" id="howArrow">&#9660;</span>
    </button>
    <div class="how-body" id="howBody">
      <div class="how-grid">
        <div class="how-card">
          <div class="how-step">STEP 1 &mdash; PREPARE YOUR FILE</div>
          <p>Save your spreadsheet as a <strong>.csv file</strong>, not .xlsx.<br>
          In Excel: <em>File &rarr; Save As &rarr; CSV UTF-8</em>.<br>
          The <strong>first row must be column names</strong>. All data must be <strong>numeric</strong>.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 2 &mdash; CHOOSE TARGET Y</div>
          <p><strong>Y</strong> is the value you want to <strong>predict or explain</strong><br>
          (e.g. price, temperature, energy output).<br>
          The app auto-detects it if your column is named <code>Y</code>, <code>target</code> or <code>output</code>.<br>
          <strong>Y will not appear in the X list</strong> &mdash; that is expected.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 3 &mdash; SELECT X VARIABLES</div>
          <p><strong>X columns</strong> are the inputs that <strong>drive Y</strong><br>
          (e.g. time, pressure, quantity).<br>
          Hold <strong>Cmd &#8984;</strong> (Mac) or <strong>Ctrl</strong> (Windows) to pick multiple columns.</p>
        </div>
        <div class="how-card">
          <div class="how-step">STEP 4 &mdash; METHOD &amp; RUN</div>
          <p><strong>Both</strong> = maximum accuracy (recommended).<br>
          <strong>Quick</strong> = fast scan of common formulas.<br>
          <strong>Adam</strong> = deep nonlinear optimizer.<br>
          Press <strong>FIND FORMULA</strong> and wait a few seconds.</p>
        </div>
      </div>
      <div class="how-warn">
        <strong>Excel &amp; semicolons:</strong> if your language settings use <strong>;</strong> as separator (Italian, German, French…), Excel may export CSV with semicolons instead of commas. The app will then see all data as a single column. Fix: open the CSV in a text editor, replace <code>;</code> with <code>,</code> &mdash; or choose <em>CSV UTF-8 (comma delimited)</em> when saving.<br><br>
        <strong>Column named &ldquo;Y&rdquo;:</strong> if one of your input variables is called <code>Y</code> (e.g. a geometric Y coordinate), rename it to something like <code>coord_y</code> before uploading, otherwise it will be auto-selected as the target variable.
      </div>
    </div>
  </div>

  <!-- ── Upload ── -->
  <div class="upload-zone" id="dropZone">
    <h2>Drop your CSV file here</h2>
    <p>or click the button below to browse</p>
    <input type="file" id="fi" accept=".csv">
    <label for="fi" class="upload-label">Choose File</label>
    <p id="fn"></p>
  </div>

  <!-- ── CSV mini-preview ── -->
  <div class="preview-box" id="previewBox">
    <h4>FILE PREVIEW</h4>
    <div id="previewTable"></div>
    <div class="preview-meta" id="previewMeta"></div>
  </div>

  <!-- ── Configure Columns ── -->
  <div class="col-select" id="colSel">
    <h3>CONFIGURE COLUMNS</h3>
    <div class="col-row">
      <div class="col-group">
        <label>Target Y</label>
        <select id="yCol" onchange="syncXcols()"></select>
        <span class="hint">The variable you want to predict</span>
      </div>
      <div class="col-group" style="flex:1;min-width:180px">
        <label>Variables X</label>
        <select id="xCols" multiple style="height:110px"></select>
        <span class="hint">Cmd/Ctrl to select multiple &nbsp;&bull;&nbsp; Y is automatically excluded</span>
      </div>
      <div class="col-group">
        <label>Method</label>
        <select id="method" onchange="updatePreviewLabel()">
          <option value="both">Both (Quick + Adam)</option>
          <option value="quick">Quick only</option>
          <option value="adam">Adam only</option>
        </select>
        <span class="hint">Both = maximum accuracy</span>
      </div>
    </div>
    <!-- Live selection preview -->
    <div class="formula-preview" id="selPreview">Select Y and X columns above to preview your setup.</div>
    <div class="val-msg" id="valMsg"></div>
    <button type="button" class="run-btn" id="runBtn" onclick="run()">&#128269; FIND FORMULA</button>
  </div>

  <div class="spinner" id="spin">Searching&hellip; please wait</div>
  <div class="error-box" id="err" style="display:none"></div>

  <!-- ── Results ── -->
  <div class="results" id="res">
    <div class="best-box">
      <h2>Best Formula Found</h2>
      <div class="best-formula" id="bf"></div>
      <button class="expand-btn" id="expandBtn" onclick="toggleFormula()" style="display:none">&#9660; Show full formula</button>
      <div class="best-r2" id="br"></div>
      <div class="result-actions">
        <button class="action-btn" id="copyBtn" onclick="copyFormula()">Copy formula</button>
        <button class="action-btn" onclick="exportCSV()">Download results CSV</button>
      </div>
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
<footer>FormulaFinder &mdash; undoubtedly created by surely not A.G.</footer>

<script>
var csv            = null;
var allCols        = [];
var parsedData     = null;   // PapaParse result
var lastResults    = null;
var formulaExpanded = false;

var fi = document.getElementById('fi');
var dz = document.getElementById('dropZone');

// ── Upload via file input ──
fi.addEventListener('change', function() {
  if (fi.files && fi.files.length > 0) handleFile(fi.files[0]);
});

// ── Drag & Drop ──
dz.addEventListener('dragover',  function(e){ e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function(){ dz.classList.remove('dragover'); });
dz.addEventListener('drop', function(e){
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(f) {
  if (!f) return;
  document.getElementById('fn').textContent = f.name;

  // Use PapaParse for robust CSV parsing (handles BOM, quotes, auto-detects separator)
  Papa.parse(f, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    complete: function(result) {
      if (!result.data || result.data.length === 0) {
        document.getElementById('fn').textContent = 'Could not parse file — check format.';
        return;
      }
      parsedData = result;
      // Rebuild raw CSV string from PapaParse output (always comma-separated, clean)
      csv = Papa.unparse(result.data);
      allCols = result.meta.fields || [];
      showPreview(result);
      parseCols();
    },
    error: function(err) {
      document.getElementById('fn').textContent = 'Parse error: ' + err.message;
    }
  });
}

function showPreview(result) {
  var box   = document.getElementById('previewBox');
  var wrap  = document.getElementById('previewTable');
  var meta  = document.getElementById('previewMeta');
  var cols  = result.meta.fields || [];
  var rows  = result.data.slice(0, 4);  // show first 4 rows

  var html  = '<table class="preview-table"><thead><tr>';
  cols.forEach(function(c){ html += '<th>' + escHtml(c) + '</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(function(r){
    html += '<tr>';
    cols.forEach(function(c){ html += '<td>' + escHtml(String(r[c] !== undefined ? r[c] : '')) + '</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
  meta.textContent = result.data.length + ' rows \u00B7 ' + cols.length + ' columns detected' +
    (result.meta.delimiter !== ',' ? '  \u26a0\ufe0f Separator detected: "' + result.meta.delimiter + '" (auto-fixed)' : '');
  box.style.display = 'block';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
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
  var yc  = document.getElementById('yCol').value;
  var xE  = document.getElementById('xCols');
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
  var yc  = document.getElementById('yCol').value;
  var xc  = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  var el  = document.getElementById('selPreview');
  if (!yc || xc.length === 0) {
    el.textContent = 'Select Y and X columns above to preview your setup.';
    return;
  }
  el.textContent = 'Ready: predicting  Y = ' + yc + '  from  X = [' + xc.join(', ') + ']';
}

// ── Run ──
async function run() {
  hideValMsg();
  if (!csv) { showValMsg('Please upload a CSV file first!'); return; }
  var yc = document.getElementById('yCol').value;
  var xc = Array.from(document.getElementById('xCols').selectedOptions).map(function(o){ return o.value; });
  if (xc.length === 0) { showValMsg('Select at least one X variable.'); return; }

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
    lastResults = d;
    lastResults._y_col = yc;
    lastResults._x_cols = xc;
    showResults(d);
  } catch(e) {
    document.getElementById('err').style.display = 'block';
    document.getElementById('err').textContent   = e.message;
  } finally {
    document.getElementById('spin').style.display = 'none';
    document.getElementById('runBtn').disabled    = false;
  }
}

function showResults(d) {
  document.getElementById('res').style.display = 'block';

  var formula = d.adam_formula || ((d.quick_results && d.quick_results[0]) || {}).formula || 'n/a';
  var bf  = document.getElementById('bf');
  var btn = document.getElementById('expandBtn');
  bf.textContent = formula;
  bf.classList.remove('expanded');
  btn.style.display = formula.length > 120 ? 'inline-block' : 'none';
  if (formula.length > 120) btn.textContent = '\u25bc Show full formula';

  var acc = d.adam_r2
    ? (d.adam_r2 * 100).toFixed(4) + '%'
    : ((d.quick_results && d.quick_results[0]) || {}).accuracy || 'n/a';
  document.getElementById('br').textContent = 'Accuracy (R\u00B2): ' + acc;

  // Chart
  var cw = document.getElementById('chartWrap');
  cw.innerHTML = '';
  if (d.chart_b64 && d.chart_b64.length > 200) {
    var img = document.createElement('img');
    img.src   = 'data:image/png;base64,' + d.chart_b64;
    img.alt   = 'Dashboard chart';
    img.onerror = function(){ cw.innerHTML = '<div class="chart-err">\u26a0 Chart not available.</div>'; };
    cw.appendChild(img);
  } else {
    cw.innerHTML = '<div class="chart-err">\u26a0 Chart not available.</div>';
  }

  // Top terms
  var tl  = document.getElementById('tl');
  tl.innerHTML = '';
  var wts = (d.top_terms || []).map(function(t){ return Math.abs(t.weight); });
  var mx  = wts.length ? Math.max.apply(null, wts) : 1;
  if (!wts.length) {
    tl.innerHTML = '<p style="color:var(--silver);font-size:.82rem">No significant terms found.</p>';
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
    cg.innerHTML = '<p style="color:var(--silver);font-size:.85rem">No Quick Search results.</p>';
  }
  (d.quick_results || []).forEach(function(r, i) {
    var cls      = r.quality === 'PERFECT' ? 'qp' : r.quality === 'GREAT' ? 'qg' : 'qb';
    var badgeCls = r.quality === 'PERFECT' ? 'badge-p' : r.quality === 'GREAT' ? 'badge-g' : 'badge-b';
    cg.innerHTML +=
      '<div class="card">' +
      '<div class="card-rank"><span class="badge ' + badgeCls + '">' + r.quality + '</span>#' + (i+1) + '</div>' +
      '<div class="card-formula">' + r.formula + '</div>' +
      '<div class="card-r2 ' + cls + '">R\u00B2 = ' + r.r2.toFixed(6) + ' &nbsp;&mdash;&nbsp; ' + r.accuracy + '</div>' +
      '</div>';
  });
}

// ── Copy formula ──
function copyFormula() {
  var formula = document.getElementById('bf').textContent;
  if (!formula || formula === 'n/a') return;
  navigator.clipboard.writeText(formula).then(function() {
    var btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    setTimeout(function(){ btn.textContent = 'Copy formula'; }, 1800);
  }).catch(function() {
    // fallback
    var ta = document.createElement('textarea');
    ta.value = formula; document.body.appendChild(ta); ta.select(); document.execCommand('copy');
    document.body.removeChild(ta);
  });
}

// ── Export results as CSV ──
function exportCSV() {
  if (!lastResults) return;
  var rows = [['rank','quality','formula','r2','accuracy']];
  (lastResults.quick_results || []).forEach(function(r, i){
    rows.push([i+1, r.quality, '"'+r.formula+'"', r.r2, r.accuracy]);
  });
  if (lastResults.adam_formula) {
    rows.push(['ADAM','—','"'+lastResults.adam_formula+'"', lastResults.adam_r2 || '', '']);
  }
  var csv_out = rows.map(function(r){ return r.join(','); }).join('\n');
  var blob = new Blob([csv_out], {type:'text/csv'});
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href = url; a.download = 'formula_finder_results.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Helpers ──
function toggleHow() {
  var body   = document.getElementById('howBody');
  var arrow  = document.getElementById('howArrow');
  var open   = body.style.display === 'block';
  body.style.display  = open ? 'none' : 'block';
  arrow.style.transform = open ? '' : 'rotate(180deg)';
}

function toggleFormula() {
  var bf  = document.getElementById('bf');
  var btn = document.getElementById('expandBtn');
  formulaExpanded = !formulaExpanded;
  bf.classList.toggle('expanded', formulaExpanded);
  btn.textContent = formulaExpanded ? '\u25b2 Collapse formula' : '\u25bc Show full formula';
}

function showValMsg(msg) {
  var el = document.getElementById('valMsg');
  el.textContent = msg;
  el.style.display = 'block';
}
function hideValMsg() {
  document.getElementById('valMsg').style.display = 'none';
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
        return jsonify({'status': 'ok', 'version': '4.0'})

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
