# prompts.md — Key AI prompts used (BAX-423 NutriAI)

AI tools were used as encouraged by the assignment. One sentence per prompt on what
the output was used for and how it was modified. All code was reviewed and can be
walked through in the live demo.

1. **Pipeline architecture** — "Design a modular Python pipeline for a clinical diet
   planner with ordered stages (intake → dedup → clinical filter → allergy exclusion →
   diet filter → rank → diversity → 7×3 meal assignment → RDA analysis → feedback),
   recording a reason for every excluded food." — Used as the skeleton of `engine.py`;
   rewrote the filters to be vectorized in pandas for speed.

2. **Two benchmarked techniques** — "Implement embeddings + FAISS nearest-neighbor
   retrieval and a Deep Neural Network (DNN) ranking model with adaptive feedback, then benchmark
   goal-fit, diversity, and latency across personas." — Became `rank_faiss`,
   `rank_dnn`, and `benchmark.py`; benchmarked the DNN ranking model against FAISS retrieval
   and tuned the nutrition scoring features to improve goal-fit performance across personas.

3. **MMR diversity** — "My planner repeated the same high-fiber seeds every meal; add
   Maximal Marginal Relevance and diversify within each food group." — Fixed the
   repetition; tuned lambda and per-group k, and added a reported diversity score.

4. **Spec-exact personas** — "Encode Priya/Ravi/Mei/James with their exact constraints
   (lactose-free vs dairy-allergic, no-pork, gluten-free cross-contamination, GI≤55,
   ≥3 fish meals, sodium ≤1500/day) and test against each one's documented pass
   criteria." — Built `default_personas()` and the criteria-specific checks in
   `test_personas.py`; corrected an over-strict micro gate that wasn't in the spec.

5. **RDA analysis by age/sex** — "Add NIH RDA tables tailored by age and sex and flag any
   day below 80% for tracked micronutrients." — Implemented `RDA`, `rda_for`, and the
   `rda_flags` logic; cross-checked values against published DRIs.

6. **Realistic portions** — "Foods are per-100g; make each meal a composite of
   complementary groups and choose grams to hit each meal's calorie budget, with sane
   per-group caps." — Brought all personas within ~±10% of calorie targets.

7. **Streamlit UI & exports** — "Build a Streamlit app exposing all six capabilities with
   demographic intake, RDA flagging, a diversity metric, per-food 👍/👎 that re-weights the
   ranker, and CSV+PDF export via fpdf2." — Implemented `app.py` / `export.py`; fixed an
   fpdf2 cell-width crash and confirmed the app serves headless without errors.

8. **RL bandit for adaptive learning** — "Replace the gradient weight-nudge with a
    Thompson-sampling multi-armed bandit over food groups (Beta arms, explore/exploit),
    updated from 👍/👎, matching the Lecture 8 bandit material." — Implemented
    `ThompsonBandit` in `engine.py`; wired it into group selection and `apply_feedback`.

9. **Constraint-solver meal assembly** — "Pose each meal slot as a constraint-
    satisfaction problem with python-constraint: calorie-window and no-repeat
    constraints, choosing the highest-ranked feasible food, with a greedy fallback." —
    Became `pick_csp` (selectable via `assembler='csp'`, default); verified all personas
    still pass and diversity improved.

10. **User-friendly interface improvements** — "Redesign the app so it feels like 
      a real meal-planning product rather than a technical demo." — Simplified the 
      sidebar, improved labels, added guidance text, and reduced visual clutter.

11. **Meal naming and presentation** — "Convert ingredient lists into meal names that
      are easier for users to understand." — Added meal names such as "Balanced Oat Bowl"
      and "Protein Grain Plate" while still showing the underlying ingredients.

12. **Personalized plan summaries** — "Generate a short explanation of why the meal plan
      fits the user's dietary needs and health conditions." — Added a personalized summary
      after plan generation.

13. **PDF meal planner redesign** — "Redesign the PDF export to look more like a weekly
      meal planner and less like a technical report." — Updated the PDF layout to 
      organize meals by day and meal type in a more readable format.

14. **Recommendation feedback improvements** — "Improve how users provide feedback
      on recommendations." — Simplified meal feedback controls and improved recommendation 
      explanations.

15. **Mixed households & religious diets** — "Add per-meal diets for mixed households
      (e.g. vegan breakfast + non-veg dinner) and halal/kosher rules (halal = no pork;
      kosher = no pork or shellfish)." — Added `_diet_mask`, `_religious_mask`, `meal_diets`
      and `religious` to the engine and sidebar controls; verified personas still pass.


16. **Clinical exclusion refinement (GERD & IBS)** — "Strengthen the exclusion engine so GERD 
      removes citrus, tomatoes, fried foods, caffeine, chocolate, spicy foods, garlic, and onion; 
      IBS removes high-FODMAP foods plus garlic, onion, and wheat with explainable logging." — 
      Updated the clinical filtering stage and exclusion audit trail so persona-specific triggers 
      are explicitly removed and documented.

17. **Glycemic index integration** — "Use published glycemic-index references and enforce GI-aware
      meal planning for the diabetes persona." — Added GI-aware filtering and reporting, surfaced 
      GI values in meal cards/PDF exports, and verified all selected foods for the diabetes persona 
      remained within the required GI threshold.

18. **DASH hypertension optimization** — "Apply DASH diet principles for the hypertension persona 
      while preserving pescatarian and soy-allergy constraints." — Added sodium caps, potassium/magnesium
      monitoring, fish-meal requirements, and DASH-oriented food selection logic.

19. **Large-scale food database expansion** — "Expand the food catalog and benchmark performance 
      on a larger recommendation space." — Increased the deduplicated food inventory from approximately
      5,200 to 10,400 foods and revalidated latency, diversity, and persona pass criteria.

20. **Diversity and no-repeat enforcement** — "Prevent repeated meals across the week and increase
      category diversity while maintaining calorie targets." — Refined reuse penalties, food-group
      balancing, and diversity scoring to better satisfy the ≥0.7 diversity benchmark and reduce repeated meal patterns.

21. **Calorie-target correction and meal scaling** — "Ensure generated plans satisfy persona calorie
      requirements while remaining clinically compliant." — Added meal-budget scaling and post-selection
      calibration so generated plans stay close to specified calorie targets (e.g., 1,600, 1,800, 
      2,000, and 2,200 kcal/day personas).

22. **Explainable nutrition benchmarking** — "Expose generation time, diversity score, food-catalog size,
      fish-meal counts, and nutrient compliance directly in the UI." — Added benchmark metrics and
      validation views used to evaluate recommendation quality, performance, and constraint
      satisfaction across personas.
