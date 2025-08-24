from __future__ import annotations
import json, os, datetime, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.tools import tool
from langchain.memory import SimpleMemory
from tools.cuisine_tools import _load as _load_recipes
from tools.textnorm import canonical_key as _canon, canonical_and_unit as _canon_and_unit



# -----------------------------------------------------------------------------
# Optional legacy import: call_pantry (no-op stub if not present)
# -----------------------------------------------------------------------------
try:
    from tools.manager_tools import call_pantry as _mgr_call_pantry  # legacy refresh
except Exception:
    def _mgr_call_pantry(_: str) -> str:
        return ""

##############################################################################
# Shared memory object – survives for the life of the Streamlit session
##############################################################################
memory: SimpleMemory = SimpleMemory(memories={})  # injected into agent via import

# Where we persist finished plans ------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PLAN_DIR = os.path.join(ROOT_DIR, "plans")
os.makedirs(PLAN_DIR, exist_ok=True)

##############################################################################
# Prefer KitchenAgent; fallback to ManagerAgent if present
##############################################################################
_kitchen_chat = None
_manager_agent_chat = None
try:
    from agents.kitchen_agent import chat as _kitchen_chat  # preferred
except Exception:
    _kitchen_chat = None

try:
    from agents import manager_agent as _manager_agent_mod  # legacy fallback
    _manager_agent_chat = _manager_agent_mod.chat
except Exception:
    _manager_agent_chat = None

# Default planning mode if not set by UI
DEFAULT_MODE = "pantry-first"   # or "user-choice"

def _get_mode() -> str:
    return (memory.memories.get("mode") or DEFAULT_MODE).strip().lower()

# ---------------- Constraints (single source of truth) ----------------
DEFAULT_CONSTRAINTS = {
    "mode": "pantry-first-strict",   # or "freeform"
    "allow_repeats": True,
    "cuisine": None,
    "diet": None,                    # "veg" | "eggtarian" | "non-veg" | None
    "max_time": None,                # int minutes or None
    "sub_policy": "100%-coverage",   # label only; strict means exact coverage
    "allow_subs": False,             # when True we allow prep/subs to reach 100%
}


def _get_constraints() -> Dict[str, Any]:
    c = memory.memories.get("constraints") or {}
    out = {**DEFAULT_CONSTRAINTS, **c}
    memory.memories["constraints"] = out
    return out

def _normalize_constraints(upd: Dict[str, Any]) -> Dict[str, Any]:
    c = _get_constraints()
    mode = (upd.get("mode") or "").strip().lower()
    if mode in ("pantry-first", "pantry-first-strict", "strict"):
        c["mode"] = "pantry-first-strict"
    elif mode in ("freeform", "user-choice", "personal-choice"):
        c["mode"] = "freeform"
    if "allow_repeats" in upd:
        c["allow_repeats"] = bool(upd["allow_repeats"])
        # NEW: prep/substitutions toggle
    if "allow_subs" in upd:
        c["allow_subs"] = bool(upd["allow_subs"])
    # accept a friendly alias too
    if "include_subs" in upd:
        c["allow_subs"] = bool(upd["include_subs"])
    if "cuisine" in upd:
        val = upd["cuisine"]
        c["cuisine"] = (val.strip().lower() or None) if isinstance(val, str) else None
    if "diet" in upd:
        val = (upd["diet"] or "").strip().lower()
        c["diet"] = val if val in ("veg", "eggtarian", "non-veg") else None
    if "max_time" in upd:
        try:
            c["max_time"] = int(upd["max_time"])
        except Exception:
            c["max_time"] = None
    if "sub_policy" in upd:
        c["sub_policy"] = str(upd["sub_policy"]).strip().lower() or "100%-coverage"
    memory.memories["constraints"] = c
    return c

@tool
def get_constraints(_: str | None = None) -> str:
    """Return the current planning constraints as JSON."""
    return json.dumps(_get_constraints())

@tool
def set_constraints(payload: Dict[str, Any] | str) -> str:
    """Update planning constraints (mode, allow_repeats, cuisine, diet, max_time, sub_policy)."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            # allow shorthand like "pantry-first" or "freeform"
            p = (payload or "").strip().lower()
            if p in ("pantry-first", "pantry-first-strict", "strict"):
                payload = {"mode": "pantry-first-strict"}
            elif p in ("freeform", "user-choice", "personal-choice"):
                payload = {"mode": "freeform"}
            else:
                return "Error: set_constraints expects a JSON object or a known mode keyword."
    if not isinstance(payload, dict):
        return "Error: set_constraints expects an object."
    c = _normalize_constraints(payload)
    nice_mode = "Pantry-first (strict)" if c["mode"] == "pantry-first-strict" else "Freeform"
    return f"OK. Mode: {nice_mode}, repeats: {c['allow_repeats']}, cuisine: {c['cuisine'] or 'any'}, diet: {c['diet'] or 'any'}, max_time: {c['max_time'] or 'any'}."

def _fmt_prompt(payload: Dict[str, Any]) -> str:
    """Turn the structured query into a natural-language prompt for Kitchen/Manager."""
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
    """Ask KitchenAgent (preferred) or ManagerAgent (fallback) for recipe options.

    Args:
        query: dict with keys like {diet, cuisine, meal_type, max_cook_time, exclude, top_k}
               or a JSON string containing the same fields.

    Returns:
        Raw text reply (expected: one recipe name per line).
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

    if _kitchen_chat is not None:
        return _kitchen_chat(prompt)
    if _manager_agent_chat is not None:
        return _manager_agent_chat(prompt)

    return ("Error: no routing agent available (KitchenAgent/ManagerAgent not found). "
            "You can still use cuisine_tools.find_recipes_by_items directly.")

from tools import pantry_tools as _pt  

def _canon_name_unit(item: str, unit: str) -> tuple[str, str]:
    return _canon_and_unit(item, unit)


def _recipe_eligible_by_filters(rec: Dict[str, Any], c: Dict[str, Any]) -> bool:
    # cuisine
    if c.get("cuisine"):
        if (rec.get("cuisine") or "").strip().lower() != c["cuisine"]:
            return False
    # diet
    want = c.get("diet")
    if want and (rec.get("diet") or "").strip().lower() not in (want, "any", ""):
        return False
    # time
    if c.get("max_time"):
        total = int(rec.get("prep_time_min", 0)) + int(rec.get("cook_time_min", 0))
        if total > int(c["max_time"]):
            return False
    return True

def _full_coverage_and_usage(rec: Dict[str, Any], shadow: Dict[str, int]) -> tuple[bool, List[str]]:
    """
    Return (ok, usage_lines). ok=True iff every ingredient can be met from shadow pantry
    with exact canonical (name,unit) key. usage_lines are human-readable like '200 g chicken'.
    """
    usage: List[str] = []
    for ing in rec.get("ingredients", []):
        item = (ing.get("item") or "").strip()
        qty  = int(ing.get("quantity") or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not item or qty <= 0:
            continue
        name_c, unit_n = _canon_name_unit(item, unit)
        key = f"{name_c} ({unit_n})"
        have = int(shadow.get(key, 0))
        if have < qty:
            return False, []
        usage.append(f"{qty} {unit_n} {name_c}")
    return True, usage

def _can_fulfill_strict(rec: Dict[str, Any], shadow: Dict[str, int]) -> bool:
    for ing in rec.get("ingredients", []):
        item = (ing.get("item") or "").strip()
        qty  = int(ing.get("quantity") or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not item or qty <= 0:
            continue
        name_c, unit_n = _canon_name_unit(item, unit)
        key = f"{name_c} ({unit_n})"
        if shadow.get(key, 0) < qty:
            return False
    return True

def _apply_deduction(rec: Dict[str, Any], shadow: Dict[str, int]) -> None:
    """
    Subtract each ingredient qty from the shadow pantry and return usage lines
    like '200 g chicken' for logging.
    """
    for ing in rec.get("ingredients", []):
        item = (ing.get("item") or "").strip()
        qty  = int(ing.get("quantity") or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not item or qty <= 0:
            continue
        name_c, unit_n = _canon_name_unit(item, unit)
        key = f"{name_c} ({unit_n})"
        shadow[key] = max(0, int(shadow.get(key, 0)) - qty)



def _eligible_recipes(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in _load_recipes() if _recipe_eligible_by_filters(r, c)]

def _slot_names(meals: Any) -> List[str]:
    if isinstance(meals, list) and all(isinstance(m, str) for m in meals):
        return meals
    if isinstance(meals, int):
        return ["Breakfast", "Lunch"] if meals == 2 else (["Dinner"] if meals == 1 else ["Breakfast", "Lunch", "Dinner"])
    return ["Breakfast", "Lunch", "Dinner"]

def _can_fulfill_with_prep(rec: Dict[str, Any], shadow: Dict[str, int]) -> tuple[bool, List[str]]:
    """
    Return (ok, notes). ok=True iff every ingredient is coverable either:
      • exactly from shadow (same name+unit), or
      • via simple prep equivalents (e.g., 'cooked rice (g)' covered by 'rice (g)').
    We do NOT mutate/deduct the shadow here — planning remains non-destructive.
    """
    notes: List[str] = []
    for ing in rec.get("ingredients", []):
        raw_name = (ing.get("item") or "").strip()
        qty = int(ing.get("quantity") or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not raw_name or qty <= 0:
            continue

        name_c, unit_n = _canon_name_unit(raw_name, unit)
        exact_key = f"{name_c} ({unit_n})"
        have_exact = int(shadow.get(exact_key, 0))

        if have_exact >= qty:
            # exact coverage — fine
            continue

        # ---- simple prep equivalents (keep this list tiny & conservative)
        # cooked rice ← rice (assume 1:1 coverage for planning purposes)
        if name_c in ("cooked rice", "steamed rice", "rice (cooked)"):
            base_key = f"rice ({unit_n})"
            if int(shadow.get(base_key, 0)) >= qty:
                notes.append("cooked rice from rice")
                continue

        # Add other tiny prep mappings here later (e.g., "boiled egg" ← "egg (count)")

        # if neither exact nor prep-equivalent covers this ingredient
        return False, notes

    return True, notes

def _shadow_pantry_snapshot_canon() -> dict[tuple[str, str], int]:
    """
    Build a shadow pantry map with canonical names:
      (canonical_name, unit_family) -> quantity
    """
    # read the single source-of-truth pantry DB
    try:
        items = dict(_pt._db.items)  # e.g., {'spinach (g)': 260, 'paneer (g)': 300, ...}
    except Exception:
        # extremely rare fallback: empty shadow
        items = {}
    shadow: dict[tuple[str, str], int] = {}
    for k, v in items.items():
        base_raw, unit_raw = _split_pantry_key(k)
        name_c, unit_n = _canon_and_unit(base_raw, unit_raw)
        shadow[(name_c, unit_n)] = shadow.get((name_c, unit_n), 0) + int(v or 0)
    return shadow

def _recipe_requirements_canon(rec: dict) -> list[tuple[str, str, int]]:
    """
    Return [(canonical_name, unit_family, qty), ...] for a recipe.
    """
    out: list[tuple[str, str, int]] = []
    for ing in rec.get("ingredients", []):
        item = (ing.get("item") or "").strip()
        qty  = int(ing.get("quantity") or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not item or qty <= 0:
            continue
        name_c, unit_n = _canon_and_unit(item, unit)
        out.append((name_c, unit_n, qty))
    return out

def _can_fulfill_strict_canon(rec: dict, shadow: dict[tuple[str, str], int]) -> bool:
    """
    True iff every canonical ingredient qty can be met from 'shadow'.
    """
    for name_c, unit_n, qty in _recipe_requirements_canon(rec):
        if shadow.get((name_c, unit_n), 0) < qty:
            return False
    return True

def _apply_deduction_canon(rec: dict, shadow: dict[tuple[str, str], int]) -> None:
    """
    Subtract each canonical ingredient qty from the shadow pantry.
    """
    for name_c, unit_n, qty in _recipe_requirements_canon(rec):
        key = (name_c, unit_n)
        shadow[key] = max(0, int(shadow.get(key, 0)) - qty)
        
def _tightness_key(rec: Dict[str, Any], shadow0: dict[tuple[str, str], int]) -> tuple:
    """
    Lower (tighter) first: recipes whose required lines are closest to pantry limits.
    Encourages placing scarce/bottleneck dishes before they get blocked by earlier picks.
    """
    mins = []
    for name_c, unit_n, need in _recipe_requirements_canon(rec):
        have = float(shadow0.get((name_c, unit_n), 0))
        if need <= 0:
            continue
        ratio = have / float(need) if need else float("inf")
        mins.append(ratio)
    min_ratio = min(mins) if mins else float("inf")
    total_time = int(rec.get("prep_time_min", 0)) + int(rec.get("cook_time_min", 0))
    return (min_ratio, total_time, (rec.get("name") or "").lower())


def _coverable_once_sorted(candidates: List[Dict[str, Any]],
                           shadow0: dict[tuple[str, str], int]) -> List[Dict[str, Any]]:
    """
    Return the list of recipes that are 100% coverable from the *initial* shadow pantry,
    sorted by 'tightness' so scarce recipes are scheduled first.
    """
    coverable = [r for r in candidates if _can_fulfill_strict_canon(r, shadow0)]
    coverable.sort(key=lambda r: _tightness_key(r, shadow0))
    return coverable

@tool
def auto_plan(payload: Dict[str, Any] | str | None = None) -> str:
    """
    Fill Day×Meals according to constraints.
    payload: {"days": int, "meals": int|[names], "continue": bool?}

    Pantry-first (strict):
      • Only recipes fully satisfied by the *canonicalized* shadow pantry.
      • Simulate deductions between slots.
      • Pass 1: place each dish that is 100% coverable from the initial pantry at least once (unseen-first).
      • Pass 2: if no unseen fits now, pick any coverable (still respects no-consecutive).
      • Stop the moment a slot cannot be filled.

    Freeform:
      • Pick eligible recipes (respect cuisine/diet/time); no coverage check.

    Repeat policy:
      • allow_repeats=False ⇒ avoid consecutive repeats (not global uniqueness).
    """
    # Parse payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload or "{}")
        except Exception:
            payload = {}
    payload = payload or {}
    days  = int(payload.get("days") or 3)
    meals = _slot_names(payload.get("meals"))
    cont  = bool(payload.get("continue") or False)

    c = _get_constraints()
    plan: Dict[str, Dict[str, str]] = memory.memories.get("plan", {}) if cont else {}
    memory.memories["plan"] = plan  # ensure it exists

    # Build slot list to fill in order
    start_at = 1
    if cont and plan:
        existing_ns = [int(re.sub(r"\D", "", d) or "0") for d in plan.keys()]
        start_at = (max(existing_ns) + 1) if existing_ns else 1
    target_days = list(range(start_at, start_at + days))

    # Candidate pool (filtered by cuisine/diet/time); deterministic order as tie-breaker
    candidates = _eligible_recipes(c)
    candidates.sort(key=lambda r: ((r.get("name") or "").lower(), (r.get("cuisine") or "").lower()))

    # Shadow pantry for strict mode (canonicalized)
    shadow = _shadow_pantry_snapshot_canon() if c["mode"] == "pantry-first-strict" else {}

    filled = 0
    total_slots = len(target_days) * len(meals)

    calc_log = memory.memories.get("calc_log", [])
    if not isinstance(calc_log, list):
        calc_log = []

    prev_dish_lower: Optional[str] = None

    # ---- NEW: compute once-coverable set (from the initial pantry), sorted by tightness
    initial_shadow = dict(shadow)
    once_list = _coverable_once_sorted(candidates, initial_shadow) if shadow else []
    once_names_left: set[str] = { (r.get("name") or "").strip().lower() for r in once_list }

    for day_i in target_days:
        day_key = f"Day{day_i}"
        day_row = plan.setdefault(day_key, {})

        for meal in meals:
            # Skip if already set (continue mode)
            if cont and day_row.get(meal):
                prev_dish_lower = (day_row.get(meal) or "").strip().lower() or prev_dish_lower
                continue

            pick = None
            pick_reason = None

            if c["mode"] == "pantry-first-strict":
                # ---------- PASS 1: prefer dishes not yet placed from the initial 100%-coverable set ----------
                if once_names_left:
                    for r in once_list:
                        name = (r.get("name") or "").strip()
                        name_l = name.lower()
                        if name_l not in once_names_left:
                            continue
                        if (not c.get("allow_repeats", True)) and prev_dish_lower and name_l == prev_dish_lower:
                            continue
                        if _can_fulfill_strict_canon(r, shadow):
                            pick = r
                            pick_reason = "100% pantry coverage (once-each pass)"
                            _apply_deduction_canon(pick, shadow)
                            once_names_left.discard(name_l)
                            break

                # ---------- PASS 2: any coverable recipe now (still respects no-consecutive) ----------
                if pick is None:
                    for r in candidates:
                        name = (r.get("name") or "").strip()
                        name_l = name.lower()
                        if (not c.get("allow_repeats", True)) and prev_dish_lower and name_l == prev_dish_lower:
                            continue
                        if _can_fulfill_strict_canon(r, shadow):
                            pick = r
                            pick_reason = "100% pantry coverage"
                            _apply_deduction_canon(pick, shadow)
                            # If it was also in once_list but we got to it only now, clear it
                            once_names_left.discard(name_l)
                            break

            else:
                # Freeform: any eligible (avoid consecutive if requested)
                for r in candidates:
                    name = (r.get("name") or "").strip()
                    name_l = name.lower()
                    if (not c.get("allow_repeats", True)) and prev_dish_lower and name_l == prev_dish_lower:
                        continue
                    pick = r
                    pick_reason = "freeform pick"
                    break

            # ---- assign or stop
            if not pick:
                day_row.setdefault(meal, "")
                # Summarize attempted part and exit
                memory.memories["plan"] = plan
                memory.memories["calc_log"] = calc_log
                nice_mode = "Pantry-first (strict)" if c["mode"] == "pantry-first-strict" else "Freeform"

                attempted_keys = [f"Day{n}" for n in range(start_at, day_i + 1)]
                lines = []
                for d in attempted_keys:
                    row = plan.get(d, {})
                    parts = [row.get(m, "—") for m in meals]
                    lines.append(f"{d}: " + ", ".join(parts))

                msg = [f"Mode: {nice_mode}. Filled {filled}/{len(attempted_keys)*len(meals)} slots."]
                if lines:
                    msg.append(" " + " ".join(lines[:min(4, len(lines))]))
                if c["mode"] == "pantry-first-strict":
                    msg.append(" I paused when your pantry couldn’t fully cover the next dish. Say \"allow repeats\", \"relax cuisine/diet/time\", or \"switch to freeform\".")
                return "".join(msg)

            dish = (pick.get("name") or "").strip()
            day_row[meal] = dish
            prev_dish_lower = dish.lower()
            filled += 1

            calc_log.append({
                "slot": f"{day_key} » {meal}",
                "dish": dish,
                "virtual_deducted": [],
                "still_missing": [],
                "reason": pick_reason or "",
            })

    # Completed all slots
    memory.memories["plan"] = plan
    memory.memories["calc_log"] = calc_log
    nice_mode = "Pantry-first (strict)" if c["mode"] == "pantry-first-strict" else "Freeform"

    # Summary (compact)
    lines = []
    for d in [f"Day{n}" for n in target_days]:
        row = plan.get(d, {})
        parts = [row.get(m, "—") for m in meals]
        lines.append(f"{d}: " + ", ".join(parts))

    msg = [f"Mode: {nice_mode}. Filled {filled}/{total_slots} slots."]
    if lines:
        msg.append(" " + " ".join(lines[:min(4, len(lines))]))
    return "".join(msg)

##############################################################################
# 2 · update_plan – mutate planner_state (no shadow-pantry simulation)
##############################################################################
@tool
def update_plan(payload: Dict[str, Any] | str | None = None) -> str:
    """Write a recipe into the plan for a given slot, and record a calc entry.

    Accepts either a dict or a JSON string:
      {"day":"Day1","meal":"Breakfast","recipe_name":"Palak Paneer", "reason":"top pantry coverage 67%"}
    """
    print("DEBUG update_plan type:", type(payload))

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
    reason = payload.get("reason")  # optional human reason

    if not (day and meal and recipe_name):
        return "Error: need 'day', 'meal', and 'recipe_name'."

    # 1) write the slot
    plan: Dict[str, Dict[str, str]] = memory.memories.setdefault("plan", {})
    plan.setdefault(day, {})[meal] = recipe_name
    memory.memories["last_query"] = json.dumps({"day": day, "meal": meal})

    # 2) record a structured calc entry (no shadow/virtual deduction)
    calc_log = memory.memories.get("calc_log", [])
    if not isinstance(calc_log, list):
        calc_log = []
    entry = {
        "slot": f"{day} » {meal}",
        "dish": recipe_name,
        "virtual_deducted": [],   # kept for UI compatibility
        "still_missing": [],      # kept for UI compatibility
    }
    if reason:
        entry["reason"] = reason
    calc_log.append(entry)
    memory.memories["calc_log"] = calc_log

    return f"Set {day} » {meal} to {recipe_name}."

##############################################################################
# 3 · missing_ingredients – import with fallback
##############################################################################
try:
    # If your manager_tools defines this @tool, reuse it.
    from tools.manager_tools import missing_ingredients  # type: ignore
except Exception:
    # Fallback: quick local implementation without quantities
    @tool
    def missing_ingredients(dish: str) -> str:
        """Tell the user which ingredients for *dish* are not in their pantry (simple fallback)."""
        recipe = _load_recipe_by_name(dish)
        if not recipe:
            return f"⚠️ Recipe '{dish}' not found."

        pantry_names = {_normalise(k.split("(")[0]) for k in _load_pantry().keys()}
        need = {_normalise(ing.get("item","")) for ing in recipe.get("ingredients", []) if ing.get("item")}
        missing = sorted(n for n in need if n and n not in pantry_names)

        if not missing:
            return f"You already have every ingredient for {dish.title()}!"
        if len(missing) == 1:
            return f"You'll still need {missing[0]} to cook {dish.title()}."
        *rest, last = missing
        return f"You'll still need {', '.join(rest)} and {last} to cook {dish.title()}."

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

def _normalise(name: str) -> str:
    """Lower-case and strip very simple plurals (onions → onion)."""
    n = (name or "").strip().lower()
    if n.endswith("ies"):
        n = n[:-3] + "y"
    elif n.endswith("s") and len(n) > 3:
        n = n[:-1]
    return n

# --- unit normalization ---
def _normalize_unit(u: Optional[str]) -> str:
    """Map many spellings to {'g','ml','count'}."""
    if not u:
        return "count"
    s = str(u).strip().lower()
    aliases = {
        "g": "g", "gram": "g", "grams": "g", "gms": "g",
        "kg": "g", "kilogram": "g", "kilograms": "g",
        "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
        "l": "ml", "liter": "ml", "liters": "ml", "litre": "ml", "litres": "ml",
        "count": "count", "piece": "count", "pieces": "count", "pc": "count", "pcs": "count",
    }
    return aliases.get(s, s)

_name_unit_re = re.compile(r"^\s*(.*?)\s*\(([^)]+)\)\s*$")

def _split_pantry_key(key: str) -> Tuple[str, str]:
    """'tomato (count)' -> ('tomato','count'), 'rice (g)' -> ('rice','g')"""
    m = _name_unit_re.match(key)
    if not m:
        base = key.split("(")[0]
        return _normalise(base), "count"
    return _normalise(m.group(1)), _normalize_unit(m.group(2))

def _find_matching_key(pantry, item, unit):
    name_c, unit_n = _canon_and_unit(item, unit or "count")
    for k in pantry.keys():
        b, u = _split_pantry_key(k)
        if _canon(b) == name_c and _normalize_unit(u) == unit_n:
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
                item = _canon(ing.get("item",""))
                _, unit = _canon_and_unit(item, ing.get("unit") or "count")  # re-normalize unit family

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
    # keep a copy in memory for UI (your sidebar reads this)
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
def _deduct_one(item: str, qty: int, unit: str) -> None:
    # Normalize exactly like pantry tools do
    name = _pt._canon_item(item)
    u    = _pt._norm_unit(unit or "count")
    _pt._db.remove(name, int(qty), u)

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
        plan = memory.memories.get("plan", {}) or {}
        day_plan = plan.get(payload["day"], {}) or {}
        dish = day_plan.get(payload["meal"])
        if not dish:
            return f"Error: no dish set for {payload['day']} » {payload['meal']}."

    recipe = _load_recipe_by_name(dish)
    if not recipe:
        return f"Error: recipe '{dish}' not found."

    # Build deducted/missing summaries by comparing before/after around the single
    # source-of-truth pantry DB (_pt._db). DO NOT write the JSON file here.
    deducted, missing = [], []

    for ing in recipe.get("ingredients", []):
        item = (ing.get("item") or "").strip()
        need_qty = int(ing.get("quantity", 0) or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if not item or need_qty <= 0:
            continue

        # Canonical key for "before" snapshot
        name_c = _pt._canon_item(item)
        unit_n = _pt._norm_unit(unit)
        key = f"{name_c} ({unit_n})"
        before = int(_pt._db.items.get(key, 0))

        # Deduct via pantry DB (this also mirrors alt units!)
        _deduct_one(item, need_qty, unit)

        after = int(_pt._db.items.get(key, 0))
        used = min(before, need_qty)

        if used > 0:
            deducted.append(f"{used} {unit_n} {item}")
        if used < need_qty:
            missing.append(f"{need_qty - used} {unit_n} {item}")

    # (No direct file writes; _pt._db already saved.)

    # Log for UI
    log = memory.memories.setdefault("planner_log", [])
    log.append({
        "event": "cooked",
        "dish": dish,
        "deducted": deducted,
        "missing": missing,
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
