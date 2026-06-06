# NutriAI — Automated Diet Plan Builder

BAX-423 Big Data · Spring 2026 Final Project · UC Davis GSM

NutriAI generates a personalized **7-day, 3-meal plan in under 60 seconds**, tailored to
clinical conditions, allergens/intolerances, diet, and age/sex-specific nutrient targets,
built on a USDA FoodData Central food database.

**Live app:** _add your Streamlit Community Cloud URL here after deploying_

## Run locally (single command)

```bash
pip install -r requirements.txt
python code/build_food_db.py     # builds data/food_snapshot.csv (first run only)
streamlit run code/app.py        # launches at http://localhost:8501
```

Optional checks:
```bash
python code/benchmark.py         # technique comparison -> data/benchmark_results.csv
python code/test_personas.py     # persona pass/fail table -> data/persona_results.csv
```

## The 6 core capabilities

1. **Clinical condition filtering** — IBS (low-FODMAP), GERD (citrus/tomato/fried/spicy/
   caffeine/chocolate + high-fat), Type 2 diabetes (glycemic index ≤55), hypertension
   (DASH low-sodium), each with an explanation.
2. **Allergy detection & exclusion** — milk, egg, fish, shellfish, tree nut, peanut, soy,
   wheat, gluten; lactose handled separately from dairy allergy; gluten-free flags
   cross-contamination. Allergen exclusion uses exact set-membership matching.
3. **Dietary preference handling** — vegan / vegetarian / pescatarian / non-veg, plus
   no-pork and lactose-free modifiers, **mixed households** (a different diet per meal,
   e.g. vegan breakfast + non-veg dinner), and **religious/cultural rules** (halal = no
   pork; kosher = no pork or shellfish).
4. **Diversity engine** — MMR diversity, no repeated foods across the 7 days, category
   spread enforced, and a reported **diversity score**.
5. **Macro & micronutrient analysis** — per-day macros (calories, protein, carbs, fat,
   fiber) and 5 micronutrients (iron, calcium, B12, vitamin D, zinc) compared to
   **age/sex RDA**, flagging any day below 80%.
6. **Sub-60-second generation** — full pipeline runs in well under 1 second; time is
   logged and displayed. Optimized with FAISS retrieval and vectorized pandas filters.

## Three BAX-423 techniques (different lectures)

| Technique | Lecture | Role |
|-----------|---------|------|
| Embeddings + FAISS nearest-neighbor retrieval | Embeddings / vectors (L5) | Rank foods by nearest-neighbor to an ideal nutrient vector |
| DNN nutrition ranking | Neural ranking / recommendation (L7) | Train a two-hidden-layer MLPRegressor on persona-specific weak labels, then rank foods with model.predict() |
| Thompson-sampling bandit (RL) | Reinforcement learning (L8-9) | Explore/exploit food groups; learn from 👍/👎 |

`code/benchmark.py` measures goal-fit, diversity, and latency for FAISS vs DNN and reports the
lift. The DNN uses `MLPRegressor.fit()` and `model.predict()` inside `rank_dnn`.

## Data

- **Source:** USDA FoodData Central (Branded, Foundation, SR Legacy).
- **Offline snapshot:** `data/food_snapshot.csv` — 5,200 foods, 17 nutrients + GI + tags;
  deduplicated by name at load. Graders run **fully offline**.
- **Live refresh (optional):** `python code/download_usda.py` with `FDC_API_KEY` set.
- **Rebuild snapshot:** `python code/build_food_db.py`.

## Test personas

`code/test_personas.py` validates each plan against the persona's **documented pass
criteria** (triggers, allergens, diet, GI, fish quota, sodium cap, diversity, micro RDA).
All four pass; the harness also runs hidden personas.

## Deploy (Streamlit Community Cloud — free public URL)

1. Push this folder to a **public GitHub repo**.
2. https://share.streamlit.io → **New app** → pick repo/branch → **Main file path** =
   `code/app.py` → **Deploy**.
3. `requirements.txt` installs automatically; the committed snapshot means no API key is
   needed. Paste the resulting `*.streamlit.app` URL into this README and the brief.

## Repo layout

```
LastName_FirstName_BAX423_Final/
├── code/        app.py, engine.py, build_food_db.py, download_usda.py,
│                benchmark.py, test_personas.py, export.py, make_brief.py
├── data/        food_snapshot.csv, benchmark_results.csv, persona_results.csv
├── brief.pdf    technical brief (<=4 pages)
├── prompts.md   key AI prompts
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

_Educational project, not medical advice. Consult a registered dietitian or physician for
clinical nutrition decisions._


## User-friendly interface update

This version simplifies the experience for non-technical users:

- Renames technical model controls from **Ranking technique** to **Plan style**.
- Shows only common inputs in the sidebar and moves sodium/GI/fish/model settings into **Advanced settings**.
- Adds a personalized “why this plan fits you” summary after generation.
- Groups ingredients into user-friendly meal names such as “Balanced Oat Bowl” and “Plant Protein Bowl.”
- Adds clearer feedback guidance so users understand that 👍/👎 improves future recommendations.
- Keeps the benchmark tab for BAX-423 grading, but labels it as a class benchmark rather than a normal user task.
- Adds a visible educational-use disclaimer near the top of the app.
