"""
tools/cuisine_tools.py  –  CRUD + query helpers for recipes.json
"""
import json, os, re
from typing import List, Optional, Dict
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DATA_DIR  = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "recipe.json"))
os.makedirs(DATA_DIR, exist_ok=True)

# ── low-level storage helpers ──────────────────────────────────────────────
def _load() -> List[Dict]:
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return []

def _save(recipes: List[Dict]):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2)

def _match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()

def _find(name: str) -> Optional[Dict]:
    return next((r for r in _load() if _match(name, r["name"])), None)


def _normalize(name: str) -> str:
    """Return the head noun for loose matching."""
    return name.lower().split()[-1]       # last word

def _normalise_diet(label: str | None) -> str:
    """Map user/recipe diet labels to canonical codes: veg, eggtarian, non-veg."""
    if not label:
        return ""
    s = str(label).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    aliases = {
        # vegetarian
        "veg": "veg",
        "vegetarian": "veg",
        "veggie": "veg",
        # eggtarian / ovo-vegetarian
        "eggtarian": "eggtarian",
        "eggetarian": "eggtarian",
        "ovo-vegetarian": "eggtarian",
        "ovo": "eggtarian",
        "egg": "eggtarian",
        # non-vegetarian
        "non-veg": "non-veg",
        "nonveg": "non-veg",
        "non-vegetarian": "non-veg",
        "nonvegetarian": "non-veg",
        "meat": "non-veg",
    }
    return aliases.get(s, s)

def diet_ok(recipe_diet, wanted):
    """Allow veg ⊂ eggtarian ⊂ non-veg (i.e., higher code is more permissive)."""
    r = _normalise_diet(recipe_diet)
    w = _normalise_diet(wanted)
    order = {"veg": 0, "eggtarian": 1, "non-veg": 2}
    if not w:
        # No user filter -> all ok
        return True
    if r not in order or w not in order:
        # Unknown labels: fall back to exact-match to be safe
        return r == w
    return order[r] <= order[w]


# ── pretty ingredient formatter ────────────────────────────────────────────
_plural_re = re.compile(r"([^aeiou]y|[sxz]|ch|sh)$", re.I)
def _plural(word: str) -> str:
    if _plural_re.search(word): return word + "es"
    return word + "s"

def _fmt_ing(item: str, qty: int | float, unit: str) -> str:
    if unit == "count":
        name = item if qty == 1 else _plural(item)
        return f"- {qty} {name}"
    return f"- {qty} {unit} {item}"

# ── LangChain tools ────────────────────────────────────────────────────────
@tool
def get_recipe(name: str) -> str:
    """Return one full recipe (ingredients & steps) or an error."""
    r = _find(name)
    if not r:
        return f"⚠️ Recipe '{name}' not found."
    header = f"🍽 **{r['name'].title()}** ({r['cuisine']}) – " \
             f"Prep {r['prep_time_min']} min · Cook {r['cook_time_min']} min"
    ings   = [_fmt_ing(i["item"], i["quantity"], i["unit"]) for i in r["ingredients"]]
    steps  = [f"{i+1}. {s}" for i, s in enumerate(r["steps"])]
    return "\n".join([header, "", "### Ingredients"] + ings +
                     ["", "### Steps"] + steps)

@tool
def list_recipes(cuisine: Optional[str] = None,
                 max_time: Optional[int] = None) -> str:
    """
    List recipe names. Optional filters:
      • cuisine = "italian", "indian", …
      • max_time = total time in minutes
    """
    items = _load()
    if cuisine:
        items = [r for r in items if r["cuisine"].lower() == cuisine.lower()]
    if max_time is not None:
        items = [r for r in items
                 if r["prep_time_min"] + r["cook_time_min"] <= max_time]
    if not items:
        return "📭 No recipes found with those filters."
    return "\n".join(f"- {r['name'].title()} ({r['cuisine']})" for r in items)

@tool
def add_recipe(recipe_json: Dict) -> str:
    """Add an entire recipe (dict). Fails if name already exists."""
    db = _load()
    if _find(recipe_json["name"]):
        return f"⚠️ Recipe '{recipe_json['name']}' already exists."
    db.append(recipe_json)
    _save(db)
    return f"✅ Added recipe '{recipe_json['name']}'."

@tool
def delete_recipe(name: str) -> str:
    """Delete a recipe by exact name."""
    db = _load()
    new_db = [r for r in db if not _match(name, r["name"])]
    if len(new_db) == len(db):
        return f"⚠️ Recipe '{name}' not found."
    _save(new_db)
    return f"🗑️ Deleted recipe '{name}'."

@tool
def find_recipes_by_items(
    items: Optional[List[str]] = None,
    cuisine: Optional[str] = None,
    max_time: Optional[int] = None,
    diet: Optional[str] = None,
    k: int = 5,
) -> str:
    """
    Suggest up to *k* recipes that best match the given *items* list.

    • Scoring = (# ingredients present) / (total ingredients)
    • Returns top-k with non-zero score, sorted desc.
    • If *items* is empty/None, fall back to filtered top-k by time.
    """
    items = [s.strip().lower() for s in (items or []) if s and s.strip()]

    recipes = _load()
    if diet:
        recipes = [r for r in recipes if diet_ok(r["diet"], diet)]
    if cuisine:
        recipes = [r for r in recipes if r["cuisine"].lower() == cuisine.lower()]
    if max_time is not None:
        recipes = [
            r for r in recipes
            if r["prep_time_min"] + r["cook_time_min"] <= max_time
        ]

    # Fallback: no items provided → return filtered list by time
    if not items:
        recipes_sorted = sorted(
            recipes,
            key=lambda r: (r["prep_time_min"] + r["cook_time_min"], r["name"])
        )[:k]
        return "\n".join(f"- {r['name'].title()} ({r['cuisine']})" for r in recipes_sorted) or "📭 No recipes match those filters."

    # Score by overlap with provided items
    pantry_norm = {_normalize(i) for i in items}
    scored: list[tuple[float, Dict]] = []
    for r in recipes:
        recipe_items = {_normalize(i["item"]) for i in r["ingredients"]}
        score = len(recipe_items & pantry_norm) / len(recipe_items)
        if score > 0:
            scored.append((score, r))

    if not scored:
        return "📭 No recipes match those items."

    from heapq import nlargest
    best = nlargest(
        k, scored,
        key=lambda t: (
            t[0],  # higher score first
            -(t[1]["prep_time_min"] + t[1]["cook_time_min"]),  # shorter total time
            t[1]["name"],
        ),
    )

    return "\n".join(
        f"- {r['name'].title()} ({r['cuisine']}) — {round(score*100):>3}% ingredients covered"
        for score, r in best
    )
