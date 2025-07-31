"""Meal‑plan utilities – read/write from slot‑memory shared with Manager."""
from __future__ import annotations
import json
import datetime as dt, re
from typing import List, Dict
from langchain_core.tools import tool
from tools.manager_tools import call_pantry, call_cuisine, memory  # slot memory

# ------------ helpers ------------------------------------------------------
def _today() -> dt.date:      
    return dt.date.today()

def _date_range(start: dt.date, days: int) -> List[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(days)]

def _short(d: dt.date) -> str:    # “Mon 8 Jul”
    return d.strftime("%a %-d %b")

def load_plan_dict() -> dict[str, list[dict]]:
    """Return the raw JSON meal-plan dictionary (day → list[ {dish, meal} ])."""
    import json, os
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, "data", "meal_plan.json"))
    if os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_plan_dict(plan: dict[str, list[dict]]):
    """Overwrite meal_plan.json with *plan*."""
    import json, os, pathlib
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, "data", "meal_plan.json"))
    pathlib.Path(PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

# A regex to detect pure ISO dates:
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _to_iso(date_str: str) -> str:
    """
    Convert 'today', 'tomorrow', natural phrases, OR MM/DD/YYYY → ISO YYYY-MM-DD.
    If it’s already ISO, just return it.
    """
    s = date_str.strip()

    # 0) Already ISO?
    if _ISO_RE.fullmatch(s):
        return s

    # 1) US MM/DD/YYYY → ISO
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"

    # 2) “today” / “tomorrow”
    today = dt.date.today()
    if s.lower() in {"today", "tod"}:
        return today.isoformat()
    if s.lower() in {"tomorrow", "tmr"}:
        return (today + dt.timedelta(days=1)).isoformat()

    # 3) “next monday” / etc.
    m2 = re.match(r"next\s+(\w+)", s, flags=re.I)
    if m2:
        weekdays = ["monday","tuesday","wednesday","thursday",
                    "friday","saturday","sunday"]
        try:
            target = weekdays.index(m2.group(1).lower())
            days_ahead = (target - today.weekday() + 7) % 7 or 7
            return (today + dt.timedelta(days=days_ahead)).isoformat()
        except ValueError:
            pass

    # 4) Give up — return original
    return date_str

                      

# ────────────────────────────────────────────────────────────────────────────
# CRUD helpers for individual meals
# ────────────────────────────────────────────────────────────────────────────
def _ensure_date(plan: dict[str, list], date: str):
    """Guarantee the given ISO date key exists in *plan*."""
    if date not in plan:
        plan[date] = []

@tool
def add_meal(date: str, meal: str, dish: str) -> str:
    """
    Add or replace one *meal* (“breakfast” / “lunch” / “dinner”)
    on the given ISO *date* with the specified *dish*.
    """
    plan = load_plan_dict()
    _ensure_date(plan, date)

    # replace if same meal already present
    plan[date] = [rec for rec in plan[date] if rec["meal"] != meal]
    plan[date].append({"meal": meal, "dish": dish})
    save_plan_dict(plan)
    return f"✅ Saved **{meal}** = *{dish.title()}* for {date}."

@tool
def delete_meal(date: str, meal: str) -> str:
    """
    Remove one meal entry (by meal name) from the given ISO *date*.
    """
    plan = load_plan_dict()
    if date not in plan:
        return "⚠️ No meals planned for that date."

    before = len(plan[date])
    plan[date] = [rec for rec in plan[date] if rec["meal"] != meal]
    if len(plan[date]) == before:
        return f"⚠️ No '{meal}' entry found on {date}."
    if not plan[date]:                 # clean up empty day
        del plan[date]

    save_plan_dict(plan)
    return f"🗑️ Deleted **{meal}** on {date}."

@tool
def list_meal_plan(start: str | None = None,
                   end: str | None = None) -> str:
    """
    Pretty‑print the stored meal plan.
    • Optional *start* and *end* (YYYY‑MM‑DD) to filter a date range.
    """
    plan = load_plan_dict()
    if not plan:
        return "📭 No meals planned yet."

    if start:
        plan = {d: v for d, v in plan.items() if d >= start}
    if end:
        plan = {d: v for d, v in plan.items() if d <= end}
    if not plan:
        return "📭 No meals in that period."

    lines: list[str] = []
    for date in sorted(plan):
        lines.append(f"### {date}")
        for rec in sorted(plan[date], key=lambda r: r['meal']):
            lines.append(f"- **{rec['meal'].title()}**: {rec['dish'].title()}")
        lines.append("")              # blank line between days
    return "\n".join(lines).strip()

@tool
def plan_meals(tool_input: str) -> str:
    """
    Draft a daily meal plan starting *start_date* for *days* days.

    • start_date may be ISO (YYYY-MM-DD), US (MM/DD/YYYY), or words
      (“today”, “tomorrow”, “next Monday”).
    • days: number of days to plan (default 7).
    • meals_per_day: 1–5 (breakfast, lunch, dinner, snack, supper).
    • diet: e.g. “veg”, “non-veg”, “mixed”.

    Uses only ingredients in your pantry (via call_pantry), stops
    on the first missing-ingredient, writes each accepted meal
    via add_meal(), and returns a markdown plan + shopping list.
    """
    # 1) parse JSON
    data = json.loads(tool_input)
    start_date    = data["start_date"]
    days          = int(data.get("days", 7))
    meals_per_day = int(data.get("meals_per_day", 1))
    diet          = data.get("diet")

    # 1) normalize date
    iso = _to_iso(start_date)
    try:
        start = dt.date.fromisoformat(iso)
    except ValueError:
        return "⚠️ start_date must be YYYY-MM-DD (or today/tomorrow/next Monday)."

    # 2) snapshot pantry
    inv_txt = call_pantry("list pantry")
    pantry_items = {
        re.split(r"\s*\(", line)[0].strip().lower()
        for line in inv_txt.splitlines() if ":" in line
    }

    plan: Dict[str, List[str]] = {}
    shortages: Dict[str, int] = {}
    slots = ["breakfast", "lunch", "dinner", "snack", "supper"][:meals_per_day]

    # 3) day-by-day
    for i in range(days):
        day = start + dt.timedelta(days=i)
        label = _short(day)
        plan[label] = []

        for slot in slots:
            # find one fully-cookable recipe
            cand = call_cuisine(
                f'find_recipes_by_items items={",".join(pantry_items)} k=1 diet={diet or ""}'
            )
            if "No recipes" in cand:
                break
            recipe = re.sub(r"^-+\s*", "", cand.splitlines()[0]).split(" (")[0]

            # get ingredients
            ing_txt = call_cuisine(f"get_recipe {recipe}")
            need = {
                re.sub(r"^\s*-\s*\d+\s+\w*\s+", "", ln).split()[-1].lower()
                for ln in ing_txt.splitlines() if ln.lstrip().startswith("-")
            }

            missing = need - pantry_items
            if missing:
                for m in missing:
                    shortages[m] = shortages.get(m, 0) + 1
                break

            # persist & consume
            add_meal(day.isoformat(), slot, recipe)
            plan[label].append(f"**{slot.title()}**: *{recipe}*")
            pantry_items -= need

        if not plan[label]:
            del plan[label]
            break

    # 4) stash in slot-memory
    memory.memories["meal_plan"] = plan
    memory.memories["shortages"] = shortages

    # 5) format output
    if not plan:
        return "📭 Cannot build even a one-day plan with current pantry."

    lines = [f"### Meal Plan ({len(plan)} day{'s' if len(plan)>1 else ''})"]
    for d, dishes in plan.items():
        lines.append(f"- {d}:")
        lines += [f"  - {dish}" for dish in dishes]

    if shortages:
        lines += ["", "### Needed Groceries"]
        lines += [f"- {item}: {qty}" for item, qty in shortages.items()]
    else:
        lines += ["", "You’re fully stocked for the whole period 🎉"]

    return "\n".join(lines)


@tool
def shopping_list_for_plan() -> str:
    """
    Return the consolidated shopping list for the last plan_meals call.
    """
    shortages = memory.memories.get("shortages", {})
    if not shortages:
        return "🎉 No outstanding shortages."

    return "\n".join(f"- {item}: {qty}" for item, qty in shortages.items())
