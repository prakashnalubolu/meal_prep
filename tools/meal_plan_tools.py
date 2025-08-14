from __future__ import annotations
import json, os, datetime, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from langchain_core.tools import tool
from langchain.memory import SimpleMemory
from tools.cuisine_tools import _load as _load_recipes
from tools.manager_tools import call_pantry as _mgr_call_pantry

##############################################################################
# Shared memory object – survives for the life of the Streamlit session      #
##############################################################################
memory: SimpleMemory = SimpleMemory(memories={})  # injected into agent via import

# Where we persist finished plans ------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PLAN_DIR = os.path.join(ROOT_DIR, "plans")
os.makedirs(PLAN_DIR, exist_ok=True)

##############################################################################
# 1 · call_manager – turn a structured JSON query into a prompt for ManagerAgent
##############################################################################
from agents import manager_agent


# Default planning mode if not set by UI
DEFAULT_MODE = "pantry-first"   # or "user-choice"

def _get_mode() -> str:
    return (memory.memories.get("mode") or DEFAULT_MODE).strip().lower()

def _fmt_prompt(payload: Dict[str, Any]) -> str:
    """Turn the structured query into a natural-language prompt for Manager."""
    diet      = payload.get("diet", "any")
    meal_type = payload.get("meal_type", "meal")
    max_time  = payload.get("max_cook_time")
    exclude   = payload.get("exclude", []) or []
    top_k     = payload.get("top_k", 5)
    cuisine   = payload.get("cuisine")

    # If caller didn’t specify prefer_pantry, derive from current mode
    if "prefer_pantry" in payload:
        prefer_pantry = bool(payload["prefer_pantry"])
    else:
        prefer_pantry = (_get_mode() == "pantry-first")

    lines = []
    if prefer_pantry:
        lines.append("What can I cook with what's in my pantry?")
    lines.append(f"Please suggest up to {top_k} {diet} recipes suitable for {meal_type}.")
    if cuisine:
        lines.append(f"The cuisine must be {cuisine}.")
    if max_time:
        lines.append(f"They should require no more than {max_time} minutes total time.")
    if exclude:
        lines.append("Do NOT include these dishes: " + ", ".join(exclude) + ".")
    lines.append("If you pass a diet filter to tools, use codes: 'veg', 'eggtarian', or 'non-veg'.")
    lines.append("Return ONE recipe name per line—no extra text.")
    return " ".join(lines)

@tool
def get_planner_mode(_: str | None = None) -> str:
    """Return current planning mode: 'pantry-first' or 'user-choice'."""
    return _get_mode()

@tool
def set_planner_mode(mode: str) -> str:
    """Set planning mode: 'pantry-first' or 'user-choice'."""
    m = (mode or "").strip().lower()
    if m not in ("pantry-first", "user-choice"):
        return "Error: mode must be 'pantry-first' or 'user-choice'."
    memory.memories["mode"] = m
    return f"OK, mode set to {m}."

@tool
def call_manager(query: Dict[str, Any] | str | None = None) -> str:
    """Ask ManagerAgent for recipe options.

    Args:
        query: dict with keys like {diet, cuisine, meal_type, max_cook_time, exclude, top_k}
               or a JSON string containing the same fields.

    Returns:
        Raw text reply from ManagerAgent (expected: one recipe name per line).
    """
    if isinstance(query, str):
        try:
            payload: Dict[str, Any] = json.loads(query)
        except Exception:
            payload = {}
    elif isinstance(query, dict):
        payload = query
    else:
        payload = {}

    prompt = _fmt_prompt(payload)
    return manager_agent.chat(prompt)


##############################################################################
# 2 · update_plan – mutate planner_state
##############################################################################
@tool
def update_plan(payload: Dict[str, Any] | str | None = None) -> str:
    """Write a recipe into the plan for a given slot, and record a calc entry.

    Accepts either a dict or a JSON string:
      {"day":"Day1","meal":"Breakfast","recipe_name":"Palak Paneer", "reason":"top pantry coverage 67%"}
    """
    # tolerate quoted JSON from the model
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return "Error: update_plan expects a JSON object with keys day, meal, recipe_name."

    if not isinstance(payload, dict):
        return "Error: update_plan payload must be an object."

    day  = payload.get("day") or payload.get("slot", {}).get("day")
    meal = payload.get("meal") or payload.get("slot", {}).get("meal")
    recipe_name = payload.get("recipe_name") or payload.get("dish") or payload.get("recipe")
    reason = payload.get("reason")  # optional human reason (e.g., coverage line)

    if not (day and meal and recipe_name):
        return "Error: need 'day', 'meal', and 'recipe_name'."

    # 1) write the slot
    plan: Dict[str, Dict[str, str]] = memory.memories.setdefault("plan", {})
    plan.setdefault(day, {})[meal] = recipe_name
    memory.memories["last_query"] = json.dumps({"day": day, "meal": meal})

    # 2) simulate consumption in SHADOW pantry for pantry-first calculation view
    #    (real pantry is only touched by cook_meal)
    recipe = _load_recipe_by_name(recipe_name)
    shadow = _ensure_shadow_pantry()

    virtual_deducted, still_missing = [], []
    if recipe:
        for ing in recipe.get("ingredients", []):
            item = ing.get("item", "")
            qty  = int(ing.get("quantity", 0) or 0)
            unit = _normalize_unit(ing.get("unit") or "count")
            if not item or qty <= 0:
                continue
            key = _find_matching_key(shadow, item, unit)
            if not key:
                still_missing.append(f"{qty} {unit} {item}")
                continue
            have = int(shadow.get(key, 0))
            use  = min(have, qty)
            shadow[key] = have - use
            virtual_deducted.append(f"{use} {unit} {item}")
            if use < qty:
                still_missing.append(f"{qty - use} {unit} {item}")

    memory.memories["shadow_pantry"] = shadow  # save back

    # 3) record a structured calc entry
    calc_log = memory.memories.get("calc_log", [])
    if not isinstance(calc_log, list):
        calc_log = []
    entry = {
        "slot": f"{day} » {meal}",
        "dish": recipe_name,
        "virtual_deducted": virtual_deducted,
        "still_missing": still_missing,
    }
    if reason:
        entry["reason"] = reason
    calc_log.append(entry)
    memory.memories["calc_log"] = calc_log

    return f"Set {day} » {meal} to {recipe_name}."
##############################################################################
# 3 · missing_ingredients – re-export Manager’s tool so Planner can call it
##############################################################################
from tools.manager_tools import missing_ingredients  # already decorated with @tool

##############################################################################
# 4 · Pantry helpers (normalize names/units, load/save)
##############################################################################
PANTRY_JSON_PATH = os.path.join(ROOT_DIR, "data", "pantry.json")

def _load_pantry() -> Dict[str, int]:
    try:
        with open(PANTRY_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    
def _ensure_shadow_pantry() -> Dict[str, int]:
    """Create a shadow pantry in memory if missing (planning simulation only)."""
    shadow = memory.memories.get("shadow_pantry")
    if not isinstance(shadow, dict):
        shadow = dict(_load_pantry())  # start from real pantry
        memory.memories["shadow_pantry"] = shadow
    return shadow

def _normalise(name: str) -> str:
    """Lower-case and strip very simple plurals (onions → onion)."""
    n = (name or "").strip().lower()
    if n.endswith("ies"):
        n = n[:-3] + "y"
    elif n.endswith("s") and len(n) > 3:
        n = n[:-1]
    return n

# --- unit normalization ---
def _normalize_unit(u: str | None) -> str:
    """Map many spellings to {'g','ml','count'}."""
    if not u:
        return "count"
    s = str(u).strip().lower()
    aliases = {
        "g": "g", "gram": "g", "grams": "g",
        "kg": "g", "kilogram": "g", "kilograms": "g",
        "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
        "l": "ml", "liter": "ml", "liters": "ml", "litre": "ml", "litres": "ml",
        "count": "count", "piece": "count", "pieces": "count", "pc": "count", "pcs": "count",
    }
    mapped = aliases.get(s, s)
    # If kilograms or liters, convert later by treating pantry keys as base unit amounts.
    return mapped

_name_unit_re = re.compile(r"^\s*(.*?)\s*\(([^)]+)\)\s*$")

def _split_pantry_key(key: str) -> Tuple[str, str]:
    """'tomato (count)' -> ('tomato','count'), 'rice (g)' -> ('rice','g')"""
    m = _name_unit_re.match(key)
    if not m:
        base = key.split("(")[0]
        return _normalise(base), "count"
    return _normalise(m.group(1)), _normalize_unit(m.group(2))

def _find_matching_key(pantry: Dict[str, int], item: str, unit: str) -> str | None:
    base = _normalise(item)
    unit = _normalize_unit(unit)
    for k in pantry.keys():
        b, u = _split_pantry_key(k)
        if b == base and u == unit:
            return k
    return None

def _load_recipe_by_name(name: str) -> Dict[str, Any] | None:
    name_l = (name or "").strip().lower()
    for r in _load_recipes():
        if r["name"].strip().lower() == name_l:
            return r
    return None

##############################################################################
# 5 · save_plan – write plan + quantity shopping list to disk
##############################################################################
def _collect_plan_requirements(plan: Dict[str, Dict[str, str]]) -> Dict[Tuple[str,str], int]:
    """Sum required qty per (item,unit) across the whole plan."""
    need: Dict[Tuple[str,str], int] = {}
    recipes = {r["name"].lower(): r for r in _load_recipes()}
    for day_dict in plan.values():
        for dish in day_dict.values():
            rec = recipes.get(dish.lower())
            if not rec:
                continue
            for ing in rec.get("ingredients", []):
                item = _normalise(ing.get("item",""))
                unit = _normalize_unit(ing.get("unit") or "count")
                qty  = int(ing.get("quantity") or 0)
                if qty <= 0 or not item:
                    continue
                need[(item, unit)] = need.get((item, unit), 0) + qty
    return need

def _quantity_shopping_deficits(plan: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Compare plan needs to pantry and return deficits with quantities."""
    pantry = _load_pantry()
    needs = _collect_plan_requirements(plan)
    deficits: List[Dict[str, Any]] = []
    for (item, unit), need_qty in needs.items():
        key = _find_matching_key(pantry, item, unit)
        have = int(pantry.get(key, 0)) if key else 0
        buy = max(0, need_qty - have)
        if buy > 0:
            deficits.append({"item": item, "unit": unit, "need": need_qty, "have": have, "buy": buy})
    # nice stable sort
    deficits.sort(key=lambda d: (d["unit"], d["item"]))
    return deficits

def _format_deficits(deficits: List[Dict[str, Any]]) -> str:
    if not deficits:
        return "🛒 Shopping list is empty — you have everything needed for the plan."
    lines = []
    for d in deficits:
        lines.append(f"- {d['buy']} {d['unit']} {d['item']}  (need {d['need']}, have {d['have']})")
    return "\n".join(lines)

@tool
def get_shopping_list(_: str | None = None) -> str:
    """Return a quantity-aware shopping list computed from the current plan."""
    plan = memory.memories.get("plan", {})
    if not plan:
        return "No plan in memory."
    deficits = _quantity_shopping_deficits(plan)
    # keep a copy in memory for UI if desired
    memory.memories["shopping_list"] = deficits
    return _format_deficits(deficits)

@tool
def save_plan(payload: Dict[str, Any] | str | None = None) -> str:
    """Persist the current plan (with constraints & shopping list) to /plans.
    Accepts either {"file_name": "..."} or a plain string name, or None."""
    plan        = memory.memories.get("plan", {})
    constraints = memory.memories.get("constraints", {})
    if not plan:
        return "No plan in memory to save."

    # quantity-aware shopping list
    deficits = _quantity_shopping_deficits(plan)
    data = {"constraints": constraints, "plan": plan, "shopping_list": deficits}

    file_name = None
    if isinstance(payload, dict):
        file_name = payload.get("file_name")
    elif isinstance(payload, str):
        file_name = payload

    if not file_name:
        file_name = f"plan_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')}"
    safe_name = file_name.replace(" ", "_")

    path = Path(PLAN_DIR) / f"{safe_name}.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    return f"Saved plan to {path.relative_to(ROOT_DIR)}"

##############################################################################
# 6 · cook_meal – mark a slot/dish cooked and consume ingredients from pantry
##############################################################################
@tool
def cook_meal(payload: Dict[str, Any] | str) -> str:
    """
    Mark a meal cooked and subtract ingredients from pantry.

    Accepts either:
      {"day":"Day1","meal":"Lunch"}   -> look up dish in planner_state["plan"]
      {"dish":"Palak Paneer"}         -> use dish directly
    """
    # Parse payload (allow JSON string)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"dish": payload}

    if not isinstance(payload, dict):
        return "Error: cook_meal expects an object or dish name."

    # Resolve the dish name
    dish = payload.get("dish")
    if not dish:
        if not (payload.get("day") and payload.get("meal")):
            return "Error: provide {'dish':name} or {'day':..,'meal':..}."
        plan = memory.memories.get("plan", {})
        day_plan = plan.get(payload["day"], {})
        dish = day_plan.get(payload["meal"])
        if not dish:
            return f"Error: no dish set for {payload['day']} » {payload['meal']}."

    recipe = _load_recipe_by_name(dish)
    if not recipe:
        return f"Error: recipe '{dish}' not found."

    pantry = _load_pantry()
    if pantry is None:
        return "Error: could not read pantry."

    deducted, missing = [], []

    for ing in recipe.get("ingredients", []):
        item = ing.get("item", "")
        need_qty = int(ing.get("quantity", 0) or 0)
        unit = (ing.get("unit") or "count").lower()

        if need_qty <= 0 or not item:
            continue

        key = _find_matching_key(pantry, item, unit)
        if not key:
            missing.append(f"{need_qty} {unit} {item}")
            continue

        have = int(pantry.get(key, 0))
        use  = min(have, need_qty)
        pantry[key] = have - use
        deducted.append(f"{use} {unit} {item}")
        if use < need_qty:
            missing.append(f"{need_qty - use} {unit} {item}")

    # Save pantry back to disk
    with open(PANTRY_JSON_PATH, "w", encoding="utf-8") as fp:
        json.dump(pantry, fp, indent=2, ensure_ascii=False)

    # Refresh Manager’s inventory snapshot so future calls see the new state
    try:
        _mgr_call_pantry("list pantry")
    except Exception:
        pass

    # Log this cook event for UI diagnostics
    log = memory.memories.setdefault("planner_log", [])
    log.append({
        "event": "cooked",
        "dish": dish,
        "deducted": deducted,   # list of "N unit item"
        "missing": missing,     # list of "N unit item"
    })
    memory.memories["planner_log"] = log

    parts = [f"✅ Marked cooked: {dish.title()}."]
    if deducted:
        parts.append("Consumed: " + ", ".join(deducted) + ".")
    if missing:
        parts.append("Still needed (not deducted): " + ", ".join(missing) + ".")
    if not deducted and not missing:
        parts.append("No ingredient lines were found in the recipe.")
    return " ".join(parts)