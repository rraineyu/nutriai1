"""
benchmark.py - compares the TWO BAX-423 techniques and measures impact.
T1: Embeddings + FAISS nearest-neighbor retrieval (embeddings/ANN lecture)
T2: DNN scoring + adaptive feedback (neural ranking/recommendation)
Metrics per persona on the same candidate pool: goal_fit, diversity, latency_ms.
Writes data/benchmark_results.csv. Run: python code/benchmark.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from engine import NutriEngine, default_personas, NUTRIENTS

def goal_fit(top, p):
    if len(top)==0: return 0.0
    s=[np.clip(top["protein_g"].mean()/25,0,1),
       np.clip(top["fiber_g"].mean()/8,0,1),
       1-np.clip(top["saturated_fat_g"].mean()/10,0,1)]
    if "T2_Diabetes" in p.conditions: s.append(1-np.clip(top["sugar_g"].mean()/10,0,1))
    if "Hypertension" in p.conditions: s.append(1-np.clip(top["sodium_mg"].mean()/300,0,1))
    for m in p.micro_priority:
        if m in top: s.append(np.clip(top[m].mean()/(top[m].max()+1e-9),0,1))
    return float(np.mean(s))

def diversity(top):
    if len(top)<2: return 0.0
    X=top[NUTRIENTS].to_numpy(float); X=(X-X.mean(0))/(X.std(0)+1e-9)
    sims=cosine_similarity(X); n=len(X)
    return float(1-(sims.sum()-n)/(n*n-n))

def run(k=20):
    eng=NutriEngine(); rows=[]
    for p in default_personas():
        cand,_=eng.candidates(p)
        for tech in ["faiss","dnn"]:
            t=time.time(); ranked=eng.rank(cand,p,tech); lat=(time.time()-t)*1000
            top=ranked.head(k)
            rows.append({"persona":p.name.split(" - ")[0],"technique":tech,
                "goal_fit":round(goal_fit(top,p),3),"diversity":round(diversity(top),3),
                "latency_ms":round(lat,1)})
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(__file__),"..","data","benchmark_results.csv"),index=False)
    print("\n=== Per-persona / per-technique ==="); print(df.to_string(index=False))
    summ=df.groupby("technique")[["goal_fit","diversity","latency_ms"]].mean().round(3)
    print("\n=== Technique averages ==="); print(summ.to_string())
    lift=(summ.loc["dnn","goal_fit"]-summ.loc["faiss","goal_fit"])/max(summ.loc["faiss","goal_fit"],1e-9)*100
    print(f"\nIMPACT: DNN vs FAISS goal_fit lift {lift:+.1f}%")
    return df

if __name__=="__main__": run()
