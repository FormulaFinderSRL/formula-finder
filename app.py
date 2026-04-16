#!/usr/bin/env python3
"""
FORMULA FINDER v3.0 — Complete App
Livelli 3+4+5+6: Adam + API REST + Web UI + Dashboard
"""
import numpy as np, pandas as pd, json, base64, io, warnings, os
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
        names+=[v,f"{v}^2",f"{v}^3",f"exp({v})",f"exp(-{v})",f"ln({v})",f"sin({v})",f"cos({v})",f"tanh({v})",f"sqrt|{v}|",f"1/(1+{v}^2)",f"{v}*exp({v})",f"{v}*sin({v})"]
    vlist=list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1,len(vlist)):
            v1,v2=vlist[i],vlist[j]; x1=np.asarray(X_dict[v1],float); x2=np.asarray(X_dict[v2],float)
            feats+=[x1*x2,eml(x1,x2)]; names+=[f"{v1}*{v2}",f"eml({v1},{v2})"]
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
        terms=[]; 
        for n,w in zip(self.names,self.w):
            if abs(w)>thr: terms.append(f"{w:+.3f}" if n=="1" else f"{w:+.3f}*{n}")
        return "y = "+" ".join(terms).lstrip("+").strip() if terms else "y = 0"
    def top_terms(self,n=5):
        idx=np.argsort(np.abs(self.w))[::-1][:n]
        return [{"term":self.names[i],"weight":round(float(self.w[i]),4)} for i in idx if abs(self.w[i])>0.01]

def quick_search(X_dict,y,top_n=8,min_r2=0.5):
    ops={}
    for v,x in X_dict.items():
        x=np.asarray(x,float); xc=np.clip(x,-500,500); xp=np.where(x>0,x,1e-10)
        ops[f"exp({v})"]=lambda d,v=v: np.exp(np.clip(d[v],-500,500))
        ops[f"ln({v})"]=lambda d,v=v: np.log(np.where(d[v]>0,d[v],1e-10))
        ops[f"{v}^2"]=lambda d,v=v: d[v]**2; ops[f"{v}^3"]=lambda d,v=v: d[v]**3
        ops[f"sqrt({v})"]=lambda d,v=v: np.sqrt(np.abs(d[v]))
        ops[f"sin({v})"]=lambda d,v=v: np.sin(d[v]); ops[f"cos({v})"]=lambda d,v=v: np.cos(d[v])
        ops[v]=lambda d,v=v: d[v]; ops[f"eml({v},1)"]=lambda d,v=v: eml(d[v],np.ones(len(d[v])))
    vlist=list(X_dict.keys())
    for i in range(len(vlist)):
        for j in range(i+1,len(vlist)):
            v1,v2=vlist[i],vlist[j]; ops[f"{v1}*{v2}"]=lambda d,v1=v1,v2=v2: d[v1]*d[v2]
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
                if abs(a-1)<0.005 and abs(b)<0.005: fs=f"y = {nm}"
                elif abs(b)<0.005: fs=f"y = {a:.3f}*{nm}"
                else: fs=f"y = {a:.3f}*{nm} + {b:.3f}"
                res.append({"formula":fs,"r2":round(r2,6),"accuracy":f"{r2*100:.4f}%",
                    "quality":"PERFECT ✅" if r2>0.9999 else "GREAT 🟢" if r2>0.99 else "GOOD 🟡"})
        except: pass
    res.sort(key=lambda d:d["r2"],reverse=True); return res[:top_n]

def make_chart_b64(X_dict,y_true,model=None):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4)); fig.patch.set_facecolor("#0D1B2A")
        vlist=list(X_dict.keys()); x_vals=np.asarray(X_dict[vlist[0]],float); y_true=np.asarray(y_true,float)
        idx=np.argsort(x_vals); xs,ys=x_vals[idx],y_true[idx]
        ax=axes[0]; ax.set_facecolor("#112233")
        ax.scatter(xs,ys,color="#00e5ff",s=25,alpha=0.7,label="Data",zorder=5)
        if model:
            yp=model.predict({vlist[0]:xs}); r2=model.r2(X_dict,y_true)
            ax.plot(xs,yp,color="#ff6b6b",lw=2.5,label=f"Fit  R²={r2:.4f}",zorder=4)
        ax.set_title("Data vs Fit",color="white",fontsize=11); ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#333355")
        ax2=axes[1]; ax2.set_facecolor("#112233")
        if model:
            res2=ys-model.predict({vlist[0]:xs}); ax2.bar(range(len(res2)),res2,color="#1B9AAA",alpha=0.7)
            ax2.axhline(0,color="#ff6b6b",lw=1.5,ls="--"); ax2.set_title("Residuals",color="white",fontsize=11)
        ax2.tick_params(colors="#888")
        for s in ax2.spines.values(): s.set_edgecolor("#333355")
        plt.tight_layout(); buf=io.BytesIO()
        plt.savefig(buf,format="png",dpi=100,bbox_inches="tight",facecolor="#0D1B2A")
        plt.close(); buf.seek(0); return base64.b64encode(buf.read()).decode()
    except: return ""

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formula Finder</title>
<style>
:root{--bg:#0D1B2A;--mid:#112233;--card:#162840;--teal:#1B9AAA;--neon:#06D6A0;--amber:#FCD34D;--red:#EF4444;--white:#fff;--silver:#B0C4D8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--white);font-family:'Segoe UI',sans-serif;min-height:100vh}
header{background:var(--mid);border-bottom:2px solid var(--teal);padding:18px 32px;display:flex;align-items:center;gap:16px}
header h1{font-size:1.6rem;letter-spacing:3px}
header span{font-size:.75rem;color:var(--neon);border:1px solid var(--neon);border-radius:20px;padding:3px 12px}
.container{max-width:1100px;margin:0 auto;padding:32px 24px}
.upload-zone{border:2px dashed var(--teal);border-radius:12px;padding:48px;text-align:center;cursor:pointer;background:var(--card);margin-bottom:28px;transition:.2s}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--neon);background:#1a2f4a}
.upload-zone h2{color:var(--teal);margin-bottom:8px}
.upload-zone p{color:var(--silver);font-size:.9rem}
.upload-zone input[type=file]{display:none}
.upload-btn{display:inline-block;margin-top:18px;background:var(--teal);color:var(--bg);border:none;padding:10px 28px;border-radius:6px;cursor:pointer;font-weight:700}
.upload-btn:hover{background:var(--neon)}
.col-select{display:none;background:var(--card);border-radius:12px;padding:24px;margin-bottom:24px;border:1px solid var(--teal)}
.col-select h3{color:var(--teal);margin-bottom:16px;letter-spacing:1px}
.col-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px}
label{color:var(--silver);font-size:.85rem;display:block;margin-bottom:4px}
select,input[type=text]{background:var(--mid);color:var(--white);border:1px solid var(--teal);border-radius:6px;padding:8px 12px;font-size:.9rem}
.run-btn{background:var(--neon);color:var(--bg);border:none;padding:12px 36px;border-radius:6px;cursor:pointer;font-weight:700;font-size:1rem;margin-top:12px;letter-spacing:1px}
.run-btn:hover{background:var(--teal);color:#fff}
.run-btn:disabled{opacity:.5;cursor:not-allowed}
.spinner{display:none;text-align:center;padding:40px;color:var(--teal);font-size:1.1rem}
.results{display:none}
.best-box{background:linear-gradient(135deg,var(--teal),#0d7a85);border-radius:12px;padding:24px 28px;margin-bottom:24px}
.best-box h2{font-size:.85rem;letter-spacing:3px;color:var(--bg);margin-bottom:8px}
.best-formula{font-size:1.4rem;font-weight:700;color:var(--bg);font-family:monospace;margin-bottom:6px;word-break:break-all}
.best-r2{font-size:.95rem;color:rgba(0,0,0,.65)}
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--card);border:1px solid #1e3a5f;border-radius:10px;padding:18px;transition:.2s}
.card:hover{border-color:var(--teal)}
.card-rank{font-size:.75rem;color:var(--silver);letter-spacing:1px}
.card-formula{font-family:monospace;font-size:.95rem;color:var(--white);margin:6px 0;word-break:break-all}
.card-r2{font-size:.85rem}
.qp{color:var(--neon)} .qg{color:#06D6A0} .qb{color:var(--amber)}
.chart-box{background:var(--card);border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #1e3a5f}
.chart-box h3{color:var(--teal);margin-bottom:14px;letter-spacing:1px}
.chart-box img{width:100%;border-radius:8px}
.terms-box{background:var(--card);border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #1e3a5f}
.terms-box h3{color:var(--teal);margin-bottom:14px}
.term-row{display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #1e3a5f}
.term-name{font-family:monospace;color:var(--white);min-width:120px}
.term-bar-wrap{flex:1;margin:0 16px;height:8px;background:#1e3a5f;border-radius:4px;overflow:hidden}
.term-bar{height:100%;background:var(--teal);border-radius:4px;transition:.5s}
.term-w{color:var(--amber);font-size:.85rem;min-width:70px;text-align:right}
.error-box{background:#2a1018;border:1px solid var(--red);border-radius:10px;padding:16px;color:var(--red);margin-bottom:16px}
footer{text-align:center;padding:24px;color:var(--silver);font-size:.8rem;border-top:1px solid #1e3a5f;margin-top:40px}
</style></head>
<body>
<header><h1>FORMULA FINDER</h1><span>Powered by EML + Adam</span></header>
<div class="container">
<div class="upload-zone" id="dropZone">
  <h2>Drop your CSV file here</h2>
  <p>or click to browse your files</p>
  <label for="fi" class="upload-btn" style="cursor:pointer;display:inline-block;margin-top:18px;">Choose File</label>
  <input type="file" id="fi" accept=".csv" style="display:none">
  <p id="fn" style="margin-top:12px;color:var(--neon);font-weight:600"></p>
</div>
<div class="col-select" id="colSel">
  <h3>CONFIGURE COLUMNS</h3>
  <div class="col-row">
    <div><label>Target (Y)</label><select id="yCol"></select></div>
    <div><label>Variables (X) — hold Ctrl for multiple</label><select id="xCols" multiple style="height:90px"></select></div>
    <div><label>Method</label><select id="method"><option value="both">Both (Quick + Adam)</option><option value="quick">Quick only</option><option value="adam">Adam only</option></select></div>
  </div>
  <button class="run-btn" id="runBtn" onclick="run()">FIND FORMULA</button>
</div>
<div class="spinner" id="spin">Searching for the formula... please wait</div>
<div class="error-box" id="err" style="display:none"></div>
<div class="results" id="res">
  <div class="best-box"><h2>BEST FORMULA FOUND</h2><div class="best-formula" id="bf"></div><div class="best-r2" id="br"></div></div>
  <div class="chart-box"><h3>DASHBOARD — Data vs Fit &amp; Residuals</h3><img id="ci" src="" alt="chart"></div>
  <div class="terms-box"><h3>TOP CONTRIBUTING TERMS (Adam)</h3><div id="tl"></div></div>
  <div class="cards-grid" id="cg"></div>
</div>
</div>
<footer>Formula Finder v3.0 | EML Operator — Odrzywołek, arXiv 2026</footer>
<script>
let csv=null;
const dz=document.getElementById('dropZone');
dz.addEventListener('click',function(e){if(e.target.tagName!=='LABEL'&&e.target.tagName!=='INPUT'){document.getElementById('fi').click();}});
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');handle(e.dataTransfer.files[0])});
document.getElementById('fi').addEventListener('change',e=>handle(e.target.files[0]));
function handle(f){if(!f)return;document.getElementById('fn').textContent='✅ '+f.name;const r=new FileReader();r.onload=e=>{csv=e.target.result;parseCols(csv)};r.readAsText(f)}
function parseCols(c){const h=c.trim().split('\n')[0].split(',').map(x=>x.trim().replace(/"/g,''));const y=document.getElementById('yCol');const x=document.getElementById('xCols');y.innerHTML='';x.innerHTML='';h.forEach((v,i)=>{y.innerHTML+=`<option value="${v}" ${i===h.length-1?'selected':''}>${v}</option>`;x.innerHTML+=`<option value="${v}" ${i<h.length-1?'selected':''}>${v}</option>`});document.getElementById('colSel').style.display='block'}
async function run(){
  document.getElementById('spin').style.display='block';
  document.getElementById('res').style.display='none';
  document.getElementById('err').style.display='none';
  document.getElementById('runBtn').disabled=true;
  const yc=document.getElementById('yCol').value;
  const xc=Array.from(document.getElementById('xCols').selectedOptions).map(o=>o.value);
  const m=document.getElementById('method').value;
  try{
    const r=await fetch('/api/find',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csv,y_col:yc,x_cols:xc,method:m})});
    const d=await r.json();
    if(!d.success)throw new Error(d.error);
    show(d);
  }catch(e){document.getElementById('err').style.display='block';document.getElementById('err').textContent='Error: '+e.message}
  finally{document.getElementById('spin').style.display='none';document.getElementById('runBtn').disabled=false}
}
function show(d){
  document.getElementById('res').style.display='block';
  const b=d.quick_results&&d.quick_results[0]||{};
  document.getElementById('bf').textContent=d.adam_formula||b.formula||'n/a';
  document.getElementById('br').textContent='Accuracy: '+(d.adam_r2?(d.adam_r2*100).toFixed(4)+'%':b.accuracy||'n/a');
  if(d.chart_b64)document.getElementById('ci').src='data:image/png;base64,'+d.chart_b64;
  const tl=document.getElementById('tl');tl.innerHTML='';
  const mx=Math.max(...(d.top_terms||[]).map(t=>Math.abs(t.weight)));
  (d.top_terms||[]).forEach(t=>{const p=Math.min(100,Math.abs(t.weight)/mx*100);tl.innerHTML+=`<div class="term-row"><span class="term-name">${t.term}</span><div class="term-bar-wrap"><div class="term-bar" style="width:${p}%"></div></div><span class="term-w">${t.weight.toFixed(4)}</span></div>`});
  const cg=document.getElementById('cg');cg.innerHTML='';
  (d.quick_results||[]).forEach((r,i)=>{const qc=r.quality.includes('PERFECT')?'qp':r.quality.includes('GREAT')?'qg':'qb';cg.innerHTML+=`<div class="card"><div class="card-rank">#${i+1}</div><div class="card-formula">${r.formula}</div><div class="card-r2 ${qc}">${r.quality} &nbsp; R²=${r.r2.toFixed(6)} &nbsp; ${r.accuracy}</div></div>`});
}
</script></body></html>"""

def create_app():
    try:
        from flask import Flask, request, jsonify, Response
    except ImportError:
        print("Install Flask: pip install flask"); return None
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50*1024*1024

    @app.route("/")
    def home(): return Response(HTML_PAGE, mimetype="text/html")

    @app.route("/health")
    def health(): return jsonify({"status":"ok","version":"3.0"})

    @app.route("/api/find", methods=["POST"])
    def api_find():
        try:
            body=request.get_json()
            df=pd.read_csv(io.StringIO(body["csv"]))
            X_dict={c:df[c].astype(float).values for c in body["x_cols"]}
            y_data=df[body["y_col"]].astype(float).values
            method=body.get("method","both")
            result={"success":True}
            if method in ("quick","both"): result["quick_results"]=quick_search(X_dict,y_data,top_n=8)
            if method in ("adam","both"):
                model=EMLAdamRegressor(lr=0.05,epochs=1200,l1=5e-4).fit(X_dict,y_data)
                result["adam_formula"]=model.formula(thr=0.05)
                result["adam_r2"]=round(model.r2(X_dict,y_data),6)
                result["top_terms"]=model.top_terms(8)
                result["chart_b64"]=make_chart_b64(X_dict,y_data,model)
            else: result["chart_b64"]=make_chart_b64(X_dict,y_data)
            return jsonify(result)
        except Exception as e: return jsonify({"success":False,"error":str(e)}),400

    @app.route("/api/json", methods=["POST"])
    def api_json():
        try:
            body=request.get_json()
            X_dict={k:np.array(v,float) for k,v in body["X"].items()}
            y_data=np.array(body["y"],float)
            method=body.get("method","quick")
            result={"success":True}
            if method in ("quick","both"): result["quick_results"]=quick_search(X_dict,y_data,top_n=8)
            if method in ("adam","both"):
                model=EMLAdamRegressor(lr=0.05,epochs=1000,l1=5e-4).fit(X_dict,y_data)
                result["adam_formula"]=model.formula()
                result["adam_r2"]=round(model.r2(X_dict,y_data),6)
                result["top_terms"]=model.top_terms(5)
            return jsonify(result)
        except Exception as e: return jsonify({"success":False,"error":str(e)}),400

    return app


# Render/gunicorn compatible app instance
app = create_app()

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=5000)
    p.add_argument("--host",default="0.0.0.0")
    p.add_argument("--demo",action="store_true")
    args=p.parse_args()
    if args.demo:
        x=np.linspace(0.5,3,50); y=np.exp(x)+np.random.normal(0,0.05,50)
        for r in quick_search({"x":x},y,top_n=3): print(f"  {r['formula']}  R²={r['r2']}")
    else:
        app=create_app()
        if app:
            print(f"""
╔══════════════════════════════════════════╗
║   Formula Finder v3.0 — RUNNING          ║
║   Open: http://localhost:{args.port}          ║
╚══════════════════════════════════════════╝""")
            app.run(host=args.host,port=args.port,debug=False)
