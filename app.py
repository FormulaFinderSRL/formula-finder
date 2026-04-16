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
