"""export.py - CSV and PDF export (capability: output). Uses fpdf2 (no system deps)."""
import io, csv
from fpdf import FPDF

def plan_to_csv(plan) -> bytes:
    buf=io.StringIO(); w=csv.writer(buf)
    w.writerow(["day","meal","food","portion_g","kcal","protein_g","carb_g","fat_g",
                "fiber_g","sugar_g","sodium_mg","iron_mg","calcium_mg","vitamin_b12_ug",
                "vitamin_d_ug","zinc_mg","glycemic_index"])
    for d in plan:
        for meal,foods in d["meals"].items():
            for f in foods:
                s=f["_scale"]
                w.writerow([d["day"],meal,f["name"],f["portion_g"],
                    round(f["energy_kcal"]*s,1),round(f["protein_g"]*s,1),
                    round(f["carb_g"]*s,1),round(f["fat_g"]*s,1),round(f["fiber_g"]*s,1),
                    round(f["sugar_g"]*s,1),round(f["sodium_mg"]*s,1),
                    round(f["iron_mg"]*s,2),round(f["calcium_mg"]*s,1),
                    round(f["vitamin_b12_ug"]*s,2),round(f["vitamin_d_ug"]*s,2),
                    round(f["zinc_mg"]*s,2),f.get("glycemic_index","")])
    return buf.getvalue().encode("utf-8")

def _c(t): return str(t).encode("latin-1","replace").decode("latin-1")

def plan_to_pdf(plan, persona, analysis) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    def clean_name(food):
        return str(food.get("name", "")).split("(")[0].strip()

    def meal_text(foods):
        lines = []
        for f in foods:
            s = float(f.get("_scale", 1))
            kcal = float(f.get("energy_kcal", 0)) * s
            protein = float(f.get("protein_g", 0)) * s
            gi = f.get("glycemic_index", "-")
            lines.append(f"{clean_name(f)} ({kcal:.0f} kcal," f"{protein:.0f}g protein," f"GI {gi:.0f})")
        return "\n".join(lines)

    # Title
    pdf.set_font("Helvetica", "B", 34)
    pdf.cell(0, 16, "MEAL PLANNER", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(47, 107, 79)
    pdf.cell(0, 7, "Personalized 7-day NutriAI meal plan", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Profile summary
    cond = ", ".join(persona.get("conditions") or []) or "none"
    allg = ", ".join(persona.get("allergens") or []) or "none"
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        0,
        5,
        _c(
            f"Profile: {persona.get('name','Custom')} | Diet: {persona.get('diet')} | "
            f"Age/Sex: {persona.get('age')}/{persona.get('sex')} | "
            f"Conditions: {cond} | Allergens: {allg}"
        ),
    )
    pdf.ln(2)

    # Table dimensions
    left = 10
    top = pdf.get_y()
    day_w = 35
    meal_w = 82
    header_h = 12
    row_h = 24

    # Header row
    pdf.set_xy(left, top)
    pdf.set_fill_color(47, 107, 79)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)

    pdf.cell(day_w, header_h, "DAY", border=1, align="C", fill=True)
    pdf.cell(meal_w, header_h, "BREAKFAST", border=1, align="C", fill=True)
    pdf.cell(meal_w, header_h, "LUNCH", border=1, align="C", fill=True)
    pdf.cell(meal_w, header_h, "DINNER", border=1, align="C", fill=True)

    y = top + header_h
    pdf.set_text_color(0, 0, 0)

    for d in plan:
        # New page if table gets too low
        if y + row_h > 190:
            pdf.add_page()
            y = 20

            pdf.set_xy(left, y)
            pdf.set_fill_color(47, 107, 79)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(day_w, header_h, "DAY", border=1, align="C", fill=True)
            pdf.cell(meal_w, header_h, "BREAKFAST", border=1, align="C", fill=True)
            pdf.cell(meal_w, header_h, "LUNCH", border=1, align="C", fill=True)
            pdf.cell(meal_w, header_h, "DINNER", border=1, align="C", fill=True)
            y += header_h
            pdf.set_text_color(0, 0, 0)

        # Day cell
        pdf.set_xy(left, y)
        pdf.set_fill_color(232, 241, 228)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(day_w, row_h, f"DAY {d['day']}", border=1, align="C", fill=True)

        # Meal cells
        x = left + day_w
        for meal in ["Breakfast", "Lunch", "Dinner"]:
            foods = d["meals"].get(meal, [])

            # Draw empty cell border
            pdf.set_xy(x, y)
            pdf.cell(meal_w, row_h, "", border=1)

            # Add text inside cell
            pdf.set_xy(x + 2, y + 3)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.multi_cell(meal_w - 4, 4, _c(meal_text(foods)))

            x += meal_w

        y += row_h

    # Nutrition summary under table
    pdf.ln(5)
    pdf.set_y(y + 4)
    da = analysis["daily_avg"]
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(
        0,
        5,
        _c(
            f"Daily average: {da['energy_kcal']:.0f} kcal, "
            f"{da['protein_g']:.0f}g protein, {da['carb_g']:.0f}g carbs, "
            f"{da['fat_g']:.0f}g fat, {da['fiber_g']:.0f}g fiber. "
            "Educational project only, not medical advice."
        ),
    )

    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")
