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
