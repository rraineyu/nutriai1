"""
app.py - NutriAI Streamlit application (BAX-423 final, full spec)
Six core capabilities: clinical filtering, allergy exclusion, dietary preference,
diversity engine, macro/micro RDA analysis, sub-60s generation. Plus explain
feature, adaptive learning, PDF/CSV export, and a technique benchmark.
Run: streamlit run code/app.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import streamlit as st
from engine import (NutriEngine, Persona, default_personas, NUTRIENTS, MACROS,
                    MICROS, rda_for, ALLERGEN_CHOICES)
from export import plan_to_csv, plan_to_pdf

st.set_page_config(page_title="NutriAI - Diet Plan Builder", page_icon="🥗", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Nunito:wght@400;500;600;700&display=swap');

:root{
  --cream:#fcf8f3; --cream2:#f6efe6;
  --sage:#a8c8a0; --sage-deep:#6fa07e; --sage-bg:#e8f1e4;
  --peach:#f6c9a8; --peach-bg:#fbe6d6; --peach-deep:#d98e5f;
  --lavender:#cdc1e8; --lavender-bg:#ece7f7;
  --butter:#f6e2a0; --butter-bg:#fbf3d4;
  --sky:#bcd6e8; --sky-bg:#e4f0f7;
  --ink:#4a4034; --ink-soft:#7a6f60; --line:#ece2d2;
}
html, body, [class*="css"] { font-family:'Nunito',system-ui,sans-serif; color:var(--ink); }
h1,h2,h3,h4 { font-family:'Fraunces',Georgia,serif !important; letter-spacing:-0.01em; color:var(--ink); }
.stApp { background:
   radial-gradient(900px 500px at 12% -8%, var(--peach-bg) 0%, transparent 55%),
   radial-gradient(800px 480px at 92% 4%, var(--lavender-bg) 0%, transparent 50%),
   radial-gradient(700px 600px at 70% 100%, var(--sky-bg) 0%, transparent 55%),
   linear-gradient(180deg,var(--cream) 0%, var(--cream2) 100%);
   background-attachment:fixed; }
.block-container { padding-top:2.2rem; max-width:1180px; }

/* big title */
h1 { font-size:2.7rem !important; }

/* metric cards - soft pastel tints, rounded, gentle shadow */
.metric-card{
  background:linear-gradient(160deg,#ffffff 0%, var(--sage-bg) 130%);
  border:1px solid #e3ecdd; border-radius:20px; padding:16px 18px;
  box-shadow:0 6px 18px -10px rgba(111,160,126,.45); }

/* meal cards - friendly rounded, colored left accent, hover lift */
.meal-card{
  background:#fffdfa; border:1px solid var(--line); border-left:5px solid var(--sage);
  border-radius:16px; padding:13px 16px; margin-bottom:6px;
  box-shadow:0 4px 14px -10px rgba(74,64,52,.35);
  transition:transform .15s ease, box-shadow .15s ease; }
.meal-card:hover{ transform:translateY(-2px);
  box-shadow:0 10px 22px -12px rgba(217,142,95,.5); }
.meal-card small{ color:var(--ink-soft); }

/* day chip - pill, pastel */
.daychip{display:inline-block;
  background:linear-gradient(135deg,var(--sage) 0%, var(--sage-deep) 100%);
  color:#fff;border-radius:999px;padding:6px 20px;
  font-family:'Fraunces',serif;font-weight:600;font-size:1.05rem;
  box-shadow:0 6px 16px -8px rgba(111,160,126,.7); margin:6px 0; }

.excl{color:var(--peach-deep);font-size:0.94em;}

/* buttons - rounded pastel, friendly hover */
.stButton>button{
  border-radius:14px; border:1.5px solid var(--sage);
  background:#fff; color:var(--sage-deep); font-weight:600;
  transition:all .15s ease; }
.stButton>button:hover{
  background:var(--sage-bg); border-color:var(--sage-deep); color:var(--ink);
  transform:translateY(-1px); }
/* primary generate button */
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--sage) 0%, var(--sage-deep) 100%);
  color:#fff; border:none;
  box-shadow:0 8px 20px -8px rgba(111,160,126,.8); }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.04); transform:translateY(-1px); }

/* download buttons */
.stDownloadButton>button{
  border-radius:14px; border:1.5px solid var(--peach); background:var(--peach-bg);
  color:var(--peach-deep); font-weight:700; }
.stDownloadButton>button:hover{ background:var(--peach); color:#fff; }

/* tabs - pill style */
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{
  background:#fff; border:1px solid var(--line); border-radius:999px;
  padding:6px 16px; font-weight:600; color:var(--ink-soft); }
.stTabs [aria-selected="true"]{
  background:var(--sage-bg) !important; color:var(--ink) !important;
  border-color:var(--sage) !important; }

/* sidebar */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#fff 0%, var(--cream2) 100%);
  border-right:1px solid var(--line); }

/* metric native */
[data-testid="stMetricValue"]{ font-family:'Fraunces',serif; color:var(--sage-deep); }
hr{ border-color:var(--line); }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_engine(): return NutriEngine()
for k,v in {"dnn_feedback":None,"plan":None,"feedback_log":[]}.items():
    if k not in st.session_state: st.session_state[k]=v
eng=get_engine()
if st.session_state.dnn_feedback: eng.dnn_feedback=dict(st.session_state.dnn_feedback)

st.markdown("# 🍽️ NutriAI")
st.markdown("##### Tell us your health needs and NutriAI will create a safe 7-day meal plan with nutrition analysis and explainable exclusions.")

CONDS=["IBS","GERD","T2_Diabetes","Hypertension"]

TECHNIQUE_LABELS = {
    "dnn": "DNN nutrition model",
    "faiss": "Fast match"
}
TECHNIQUE_HELP = {
    "dnn": "DNN nutrition model uses a feed-forward neural network scorer from the project.",
    "faiss": "Fast match uses the embeddings + FAISS retrieval baseline from the project."
}

def _clean_food_name(food):
    return str(food.get("name", "")).split("(")[0].strip()

def meal_title(meal, foods):
    """Create more specific, ingredient-aware meal names.

    The earlier version used broad labels like "Protein Grain Plate," which made
    many meals sound repeated even when the ingredients were different. This
    version names meals from the actual ingredients so the weekly plan feels more
    varied and more like a real meal-planning product.
    """
    names = " ".join(_clean_food_name(f).lower() for f in foods)

    def has(*words):
        return any(w in names for w in words)

    # ----- Breakfast-style names -----
    if meal == "Breakfast":
        if has("oat milk") and has("blueberr"):
            return "Blueberry Oat Milk Morning Bowl"
        if has("oat milk") and has("banana"):
            return "Banana Oat Milk Sunrise Bowl"
        if has("oat milk"):
            return "Creamy Oat Milk Breakfast"
        if has("soy milk") and has("kiwi"):
            return "Tropical Kiwi Soy Breakfast"
        if has("soy milk") and has("strawberr"):
            return "Strawberry Soy Protein Start"
        if has("soy milk"):
            return "Soy Protein Morning Bowl"
        if has("almond milk"):
            return "Light Almond Morning Bowl"
        if has("lactose-free milk"):
            return "Lactose-Free Morning Plate"
        if has("oat") and has("avocado"):
            return "Creamy Avocado Oat Bowl"
        if has("oat") and has("banana"):
            return "Banana Oat Energy Bowl"
        if has("oat"):
            return "Fiber-Rich Oat Breakfast"
        if has("quinoa") and has("lemon"):
            return "Bright Quinoa Citrus Bowl"
        if has("buckwheat") and has("avocado"):
            return "Nutty Buckwheat Avocado Bowl"
        if has("buckwheat"):
            return "Warm Buckwheat Breakfast Bowl"
        if has("rice") and has("banana"):
            return "Golden Banana Rice Bowl"
        if has("blueberr"):
            return "Blueberry Bright Start Bowl"
        if has("kiwi"):
            return "Kiwi Morning Refresh Plate"
        if has("strawberr"):
            return "Strawberry Sunrise Bowl"
        if has("banana"):
            return "Banana Sunrise Breakfast"
        if has("orange", "lemon"):
            return "Citrus Morning Plate"
        return "Bright Start Breakfast Plate"

    # ----- Ingredient-specific lunch/dinner names -----
    if has("tofu") and has("rice"):
        return "Garden Tofu Rice Bowl"
    if has("tofu") and has("broccoli"):
        return "Broccoli Tofu Power Bowl"
    if has("tofu") and has("tomato"):
        return "Savory Tofu Tomato Plate"
    if has("tofu"):
        return "Plant Protein Tofu Plate"

    if has("tempeh") and has("broccoli"):
        return "Tempeh Broccoli Power Plate"
    if has("tempeh") and has("kale"):
        return "Tempeh Kale Harvest Bowl"
    if has("tempeh") and has("rice"):
        return "Tempeh Rice Protein Bowl"
    if has("tempeh"):
        return "Savory Tempeh Harvest Plate"

    if has("edamame") and has("sweet potato"):
        return "Edamame Sweet Potato Bowl"
    if has("edamame") and has("rice"):
        return "Edamame Rice Power Bowl"
    if has("edamame") and has("spinach"):
        return "Green Edamame Protein Bowl"
    if has("edamame"):
        return "Edamame Power Plate"

    if has("egg") and has("quinoa"):
        return "Protein-Packed Quinoa Egg Plate"
    if has("egg") and has("oat"):
        return "Savory Egg Oat Protein Plate"
    if has("egg") and has("green beans"):
        return "Egg & Green Bean Protein Plate"
    if has("egg"):
        return "Savory Egg Protein Plate"

    if has("pumpkin") and has("tortilla"):
        return "Crunchy Southwest Seed Plate"
    if has("pumpkin") and has("kale"):
        return "Crunchy Kale Seed Bowl"
    if has("pumpkin") and has("spinach"):
        return "Spinach Pumpkin Seed Power Bowl"
    if has("pumpkin"):
        return "Mineral-Rich Pumpkin Seed Plate"

    if has("almond") and has("buckwheat"):
        return "Almond Buckwheat Crunch Bowl"
    if has("almond") and has("broccoli"):
        return "Almond Broccoli Garden Plate"
    if has("almond") and has("spinach"):
        return "Almond Spinach Power Plate"
    if has("almond"):
        return "Almond Crunch Protein Plate"

    if has("walnut") and has("buckwheat"):
        return "Walnut Buckwheat Harvest Bowl"
    if has("walnut") and has("zucchini"):
        return "Walnut Zucchini Garden Plate"
    if has("walnut") and has("sweet potato"):
        return "Walnut Sweet Potato Plate"
    if has("walnut"):
        return "Toasty Walnut Energy Plate"

    if has("quinoa") and has("kale"):
        return "Quinoa Kale Wellness Bowl"
    if has("quinoa"):
        return "Quinoa Garden Bowl"
    if has("buckwheat") and has("sweet potato"):
        return "Buckwheat Sweet Potato Bowl"
    if has("buckwheat"):
        return "Nutty Buckwheat Grain Bowl"
    if has("rice") and has("spinach"):
        return "Spinach Rice Garden Bowl"
    if has("rice") and has("tomato"):
        return "Tomato Rice Garden Plate"
    if has("rice"):
        return "Simple Rice Garden Bowl"
    if has("corn tortilla"):
        return "Warm Tortilla Protein Plate"
    if has("sweet potato"):
        return "Harvest Sweet Potato Plate"

    if has("broccoli") and has("green beans"):
        return "Green Garden Vegetable Plate"
    if has("broccoli"):
        return "Broccoli Wellness Plate"
    if has("kale") and has("spinach"):
        return "Leafy Greens Power Bowl"
    if has("kale"):
        return "Kale Harvest Bowl"
    if has("spinach"):
        return "Spinach Garden Bowl"
    if has("zucchini"):
        return "Zucchini Garden Plate"
    if has("tomato"):
        return "Tomato Garden Plate"

    # Meal-type fallback, still more natural than generic repeated labels.
    if meal == "Lunch":
        return "Midday Harvest Plate"
    if meal == "Dinner":
        return "Evening Garden Bowl"
    return "NutriAI Signature Meal"
def plan_fit_summary(persona, analysis):
    """Short personalized summary shown after generation."""
    parts = []
    if persona.get("conditions"):
        readable = {
            "IBS": "IBS-sensitive triggers",
            "GERD": "GERD reflux triggers",
            "T2_Diabetes": "high-glycemic foods",
            "Hypertension": "higher-sodium foods",
        }
        parts.append("avoids " + ", ".join(readable.get(c, c) for c in persona.get("conditions", [])))
    if persona.get("allergens"):
        parts.append("removes " + ", ".join(persona.get("allergens", [])) + " allergens")
    if persona.get("lactose_free"):
        parts.append("keeps meals lactose-free")
    if persona.get("gluten_free"):
        parts.append("keeps meals gluten-free")
    if persona.get("diet") and persona.get("diet") != "nonveg":
        parts.append(f"stays {persona.get('diet')}")
    kcal = analysis.get("daily_avg", {}).get("energy_kcal", 0)
    diversity = analysis.get("diversity_score", "-")
    base = "This plan " + "; ".join(parts) + "." if parts else "This plan is built around your profile and nutrition targets."
    return f"{base} Daily average is about {kcal:.0f} calories with a diversity score of {diversity}."

def ingredient_line(foods):
    return ", ".join(_clean_food_name(f) for f in foods)

with st.sidebar:
    st.markdown("## Build your plan")
    st.caption("Start with a sample persona or enter your own needs.")

    presets={p.name:p for p in default_personas()}
    sel=st.selectbox("Start from",["Custom"]+list(presets.keys()))

    if sel!="Custom":
        b=presets[sel]
        d=dict(conditions=b.conditions,allergens=b.allergens,diet=b.diet,kcal=b.kcal_target,
               age=b.age,sex=b.sex,lact=b.lactose_free,pork=b.no_pork,gf=b.gluten_free,
               fish=b.require_fish_meals,na=b.sodium_daily_cap,gi=b.gi_max)
    else:
        d=dict(conditions=[],allergens=[],diet="nonveg",kcal=2000,age=35,sex="female",
               lact=False,pork=False,gf=False,fish=0,na=0,gi=0)

    st.markdown("### 1. About you")
    c1,c2=st.columns(2)
    age=c1.number_input("Age",10,100,d["age"])
    sex=c2.selectbox("Sex",["female","male"],index=0 if d["sex"]=="female" else 1)
    kcal=st.slider("Daily calorie goal",1200,3000,d["kcal"],50)

    st.markdown("### 2. Food preferences")
    diet=st.selectbox("Diet",["nonveg","pescatarian","vegetarian","vegan"],
                      index=["nonveg","pescatarian","vegetarian","vegan"].index(d["diet"]))
    allergens=st.multiselect("Allergies to avoid",ALLERGEN_CHOICES,default=d["allergens"])
    cc1,cc2,cc3=st.columns(3)
    lact=cc1.checkbox("Lactose-free",d["lact"])
    pork=cc2.checkbox("No pork",d["pork"])
    gf=cc3.checkbox("Gluten-free",d["gf"])

    st.markdown("### 3. Health needs")
    conditions=st.multiselect("Conditions",CONDS,default=d["conditions"])

    with st.expander("Advanced settings"):
        st.caption("Most users can leave these as-is. They are included for the class demo and clinical constraints.")
        na=st.slider("Sodium daily cap (mg, 0=off)",0,2500,d["na"],50)
        gi=st.slider("Max glycemic index (0=off)",0,100,d["gi"],5)
        fish=st.slider("Minimum fish meals/week",0,7,d["fish"])
        plan_style=st.radio("Plan style",["dnn","faiss"],
            format_func=lambda t: TECHNIQUE_LABELS[t],
            help="DNN nutrition model = neural network ranking. Fast match = embeddings + FAISS.")
        st.caption(TECHNIQUE_HELP[plan_style])
    technique=plan_style

    go=st.button("Create my 7-day plan",type="primary",use_container_width=True)

persona=Persona(name=sel if sel!="Custom" else "Custom profile",conditions=conditions,
    allergens=allergens,diet=diet,kcal_target=kcal,age=age,sex=sex,lactose_free=lact,
    no_pork=pork,gluten_free=gf,require_fish_meals=fish,sodium_daily_cap=na,gi_max=gi,
    micro_priority=presets[sel].micro_priority if sel!="Custom" else [])

if go:
    with st.spinner("Filtering foods -> ranking -> assigning meals..."):
        t0=time.time()
      
        plan,log,an=eng.build_plan(persona,technique)
      
        gt=time.time()-t0
    if plan is None:
        st.error("No foods remain after filtering. Loosen a constraint and retry.")
    else:
        st.session_state.plan={"plan":plan,"log":log.__dict__,"analysis":an,
            "persona":persona.__dict__,"gt":gt,"technique":technique}

tab_plan,tab_an,tab_excl,tab_bench=st.tabs(
    ["My plan 📝 ","Nutrition 📊","Why excluded 🚫","Class benchmark 🔬"])
state=st.session_state.plan

with tab_plan:
    if not state: st.info("Set your profile and click **Generate 7-day plan**.")
    else:
        plan=state["plan"]; an=state["analysis"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Generation time",f"{state['gt']:.2f}s","under 60s")
        c2.metric("Diversity score",an["diversity_score"],"target >=0.7")
        c3.metric("Foods (deduped)",f"{eng.n_dedup:,}")
        c4.metric("Fish meals",an.get("fish_meals",0))
        st.success(plan_fit_summary(state["persona"], an))
        st.caption("Save or replace meals to help NutriAI adjust future recommendations when you regenerate the plan.")
        st.markdown("---")
        csvb=plan_to_csv(plan); pdfb=plan_to_pdf(plan,state["persona"],an)
        e1,e2=st.columns(2)
        e1.download_button("⬇ Download CSV",csvb,"nutriai_plan.csv","text/csv",use_container_width=True)
        e2.download_button("⬇ Download PDF",pdfb,"nutriai_plan.pdf","application/pdf",use_container_width=True)
        st.markdown("---")
        for day in plan:
            st.markdown(f'<span class="daychip">Day {day["day"]}</span>',unsafe_allow_html=True)
            cols=st.columns(3)
            meal_style={"Breakfast":("🥣","var(--peach)"),
                        "Lunch":("🥗","var(--sage)"),
                        "Dinner":("🍲","var(--lavender)")}
            for col,(meal,foods) in zip(cols,day["meals"].items()):
                with col:
                    emoji,accent=meal_style.get(meal,("🍽️","var(--sage)"))
                    title = meal_title(meal, foods)
                    total_kcal = sum(float(f["energy_kcal"])*float(f["_scale"]) for f in foods)
                    total_protein = sum(float(f["protein_g"])*float(f["_scale"]) for f in foods)
                    avg_gi = sum(float(f.get("glycemic_index",0) or 0) for f in foods) / max(len(foods),1)
                    st.markdown(f"#### {emoji} {meal}")
                    st.markdown(
                        f'<div class="meal-card" style="border-left-color:{accent}">'
                        f'<b style="font-size:1.05rem">{title}</b><br>'
                        f'<small><b>Includes:</b> {ingredient_line(foods)}</small><br>'
                        f'<small>{total_kcal:.0f} kcal · {total_protein:.0f}g protein · avg GI {avg_gi:.0f}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    b1, b2 = st.columns(2)

                    with b1:
                        if st.button(
                            "⭐ Save meal",
                            key=f"save_d{day['day']}_{meal}",
                            use_container_width=True
                        ):
                            for f in foods:
                                eng.apply_feedback(f, True)
                            st.session_state.dnn_feedback = dict(eng.dnn_feedback)
                            st.session_state.feedback_log.append((title, "saved meal"))
                            st.toast(f"Saved: {title}")

                    with b2:
                        if st.button(
                            "🔄 Replace meal",
                            key=f"replace_d{day['day']}_{meal}",
                            use_container_width=True
                        ):
                            for f in foods:
                                eng.apply_feedback(f, False)
                            st.session_state.dnn_feedback = dict(eng.dnn_feedback)
                            st.session_state.feedback_log.append((title, "replace requested"))
                            st.toast("Alternative meal feature coming soon")
            st.markdown("")
        if st.session_state.feedback_log:
            with st.expander("🧠 Adaptive learning log"):
                st.caption("Your ratings update the adaptive recommender. Regenerate to see the plan shift toward liked foods.")
                for nm,act in st.session_state.feedback_log[-12:]: st.write(f"• {act}: {nm}")

with tab_an:
    if not state: st.info("Generate a plan to see nutrient analysis.")
    else:
        an=state["analysis"]; da=an["daily_avg"]; rda=an["rda"]
        st.markdown("### Daily macros")
        cols=st.columns(5)
        for col,m in zip(cols,MACROS):
            label={"energy_kcal":"Calories","protein_g":"Protein g","carb_g":"Carb g",
                   "fat_g":"Fat g","fiber_g":"Fiber g"}[m]
            col.markdown(f'<div class="metric-card"><small>{label}</small><br>'
                f'<span style="font-size:1.5em;font-family:Fraunces">{da[m]:.0f}</span></div>',
                unsafe_allow_html=True)
        st.markdown("### Micronutrients vs RDA (tailored by age/sex)")
        mrows=[]
        
        sex = str(state["persona"].get("sex","")).lower()
        
        target_overrides = {
            "sodium_mg": 1500,
            "potassium_mg": 3400 if state["persona"]["sex"] == "male" else 2600,
            "magnesium_mg": 400 if state["persona"]["sex"] == "male" else 310,
        }
        display_micros = MICROS + ["sodium_mg", "potassium_mg", "magnesium_mg"]
        for m in display_micros:
            tgt = target_overrides.get(m, rda.get(m, 0))

            if m == "sodium_mg":
                pct = 100 * da[m] / tgt if tgt else 0
                status = "✅" if da[m] <= tgt else "⚠️ > target"
            else:
                pct = 100 * da[m] / tgt if tgt else 0
                status = "✅" if pct >= 80 else "⚠️ <80%"
            mrows.append({"Micronutrient":m,"Daily avg":round(da[m],2),"RDA":tgt,
                          "% RDA":round(pct),"Status":"✅" if pct>=80 else "⚠️ <80%"})
        st.dataframe(pd.DataFrame(mrows),use_container_width=True,hide_index=True)
        flags=an.get("rda_flags",[])
        st.markdown(f"### RDA gap flags ({len(flags)} day-nutrient below 80%)")
        if flags:
            st.dataframe(pd.DataFrame(flags),use_container_width=True,hide_index=True)
            st.caption("Note: B12 and vitamin D are commonly supplemented, especially on "
                       "vegan/lactose-free diets - flags here are informational.")
        else: st.success("All tracked nutrients meet >=80% RDA every day.")

with tab_excl:
    if not state: st.info("Generate a plan to see exclusion explanations.")
    else:
        log=state["log"]
        st.markdown("### Why foods were excluded")
        st.caption("This explains what the app removed before building your plan. Showing up to 25 examples per stage.")
        for stage,label in [("clinical","🩺 Clinical condition"),("allergy","⚠️ Allergen / intolerance"),
                            ("diet","🥬 Diet preference")]:
            dct=log.get(stage,{})
            with st.expander(f"{label} - {len(dct):,} foods excluded"):
                if not dct: st.write("None.")
                for nm,reason in list(dct.items())[:25]:
                    st.markdown(f'<span class="excl">• <b>{nm.split("(")[0].strip()}</b> - {reason}</span>',
                                unsafe_allow_html=True)

with tab_bench:
    st.markdown("### Class benchmark")
    st.caption("This tab is mainly for the BAX-423 project evaluation. Normal users do not need to change these settings.")
    bp=os.path.join(os.path.dirname(__file__),"..","data","benchmark_results.csv")
    if os.path.exists(bp):
        bdf=pd.read_csv(bp); st.dataframe(bdf,use_container_width=True,hide_index=True)
        summ=bdf.groupby("technique")[["goal_fit","diversity","latency_ms"]].mean().round(3)
        st.dataframe(summ,use_container_width=True)
        st.markdown("- **Embeddings + FAISS** nearest-neighbor retrieval (Lecture 5, Embeddings & Vectors)")
        st.markdown("- **Reinforcement learning - Thompson bandit** over food groups (Lectures 8-9, RL)")
        st.markdown("- **DNN ranking** - feed-forward neural network scorer + re-rank (Lecture 7/ML Ranking)")
    else: st.info("Run `python code/benchmark.py` to populate results.")

st.markdown("---")
st.caption("NutriAI is an educational project, not medical advice. Consult a registered "
           "dietitian or physician for clinical nutrition decisions.")
