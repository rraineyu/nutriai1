"""
test_personas.py - validates plans against EACH persona's spec pass-criteria and
the 6 core capabilities, producing the pass/fail table for the brief.
Run: python code/test_personas.py   (exit 0 if all pass)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from engine import (NutriEngine, default_personas, rda_for, GERD_TRIGGERS, MICROS)

def foods_of(plan):
    return [f for d in plan for meal in d["meals"].values() for f in meal]

def per_day_totals(plan):
    days=[]
    from engine import NUTRIENTS
    for d in plan:
        t={n:0.0 for n in NUTRIENTS}
        for meal,foods in d["meals"].items():
            for f in foods:
                s=f["_scale"]
                for n in NUTRIENTS: t[n]+=float(f.get(n,0))*s
        days.append(t)
    return days

def check(eng, p):
    plan,log,an=eng.build_plan(p,"dnn")
    foods=foods_of(plan); days=per_day_totals(plan)
    rda=rda_for(p.age,p.sex); fails=[]

    # capability checks per persona spec
    if "IBS" in p.conditions:
        if any(str(f["fodmap"]).upper()=="H" for f in foods): fails.append("FODMAP trigger present")
    if p.lactose_free and any(str(f.get("lactose")).upper()=="Y" for f in foods):
        fails.append("dairy/lactose present")
    if p.diet=="vegetarian" and any("vegetarian" not in str(f["diet_tags"]) for f in foods):
        fails.append("non-vegetarian item")
    if "GERD" in p.conditions:
        if any(any(t in str(f["gerd_trigger"]) for t in GERD_TRIGGERS) for f in foods):
            fails.append("GERD trigger present")
    if p.gluten_free and any(str(f["gluten"]).upper()=="Y" for f in foods):
        fails.append("gluten present")
    if p.gi_max:
        if any(float(f.get("glycemic_index",0))>p.gi_max for f in foods):
            fails.append(f"GI>{p.gi_max} present")
    if p.diet=="vegan" and any("vegan" not in str(f["diet_tags"]) for f in foods):
        fails.append("non-vegan item")
    for a in p.allergens:
        if any(a in str(f["allergens"]).split("|") for f in foods):
            fails.append(f"allergen {a} present")
    if p.diet=="pescatarian":
        nf=sum(1 for f in foods if str(f.get("fish_seafood")).upper()=="Y")
        if nf < p.require_fish_meals: fails.append(f"only {nf} fish meals (<{p.require_fish_meals})")
    if p.sodium_daily_cap:
        if any(d["sodium_mg"]>p.sodium_daily_cap for d in days):
            mx=max(d['sodium_mg'] for d in days); fails.append(f"sodium {mx:.0f}>{p.sodium_daily_cap} some day")

    # ---- exact documented pass-criteria micronutrient gate (per persona) ----
    # Priya: iron >=80% RDA daily | Ravi: B12 >=80% | Mei: fibre >=25g/day
    # James: potassium >=80% RDA daily
    name=p.name.split(" - ")[0]
    gate={"Priya":("iron_mg",0.8),"Ravi":("vitamin_b12_ug",0.8),
          "James":("potassium_mg",0.8)}.get(name)
    if gate:
        m,frac=gate; worst=min(d[m] for d in days)
        if rda.get(m) and worst < frac*rda[m]:
            fails.append(f"{m} {worst:.1f}<{int(frac*100)}% RDA")
    if name=="Mei":
        if min(d["fiber_g"] for d in days) < 25: fails.append("fibre <25g/day")
    # diversity & timing
    if an["diversity_score"]<0.7: fails.append(f"diversity {an['diversity_score']}<0.7")
    return {"persona":p.name.split(" - ")[0],"pass":len(fails)==0,
            "kcal":round(an["daily_avg"]["energy_kcal"]),
            "diversity":an["diversity_score"],"fish":an["fish_meals"],
            "reason":"OK" if not fails else "; ".join(fails[:3])}

def run(personas=None):
    eng=NutriEngine(); personas=personas or default_personas()
    res=[check(eng,p) for p in personas]
    df=pd.DataFrame(res)
    print("\n=== PERSONA PASS/FAIL ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(os.path.dirname(__file__),"..","data","persona_results.csv"),index=False)
    n=int(df["pass"].sum()); print(f"\n{n}/{len(df)} passed")
    return df, n==len(df)

if __name__=="__main__":
    _,ok=run(); sys.exit(0 if ok else 1)
