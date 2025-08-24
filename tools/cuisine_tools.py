"""
tools/cuisine_tools.py  –  CRUD + query helpers for recipes.json
"""
import json, os, re, difflib
from typing import List, Optional, Dict
from dotenv import load_dotenv
from langchain_core.tools import tool
from tools.textnorm import canonical_key, canonicalize_many


load_dotenv()

DATA_DIR  = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "recipe.json"))
os.makedirs(DATA_DIR, exist_ok=True)

# ── low-level storage helpers ──────────────────────────────────────────────
def _load() -> List[Dict]:
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)  # let JSON errors raise
    return []

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

_plural_re = re.compile(r"([^aeiou]y|[sxz]|ch|sh)$", re.I)
def _plural(word: str) -> str:
    if _plural_re.search(word): return word + "es"
    return word + "s"

def _fmt_ing(item: str, qty: int | float, unit: str) -> str:
    if unit == "count":
        name = item if qty == 1 else _plural(item)
        return f"- {qty} {name}"
    return f"- {qty} {unit} {item}"

def _clean_name(s: str) -> str:
    s = str(s or "").strip()
    # strip outer matching quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    # strip any stray leading/trailing quotes and collapse spaces
    s = s.strip('\'"')
    s = re.sub(r"\s+", " ", s)
    return s

def _match(a: str, b: str) -> bool:
    return _clean_name(a).lower() == _clean_name(b).lower()

def _find(name: str) -> Optional[Dict]:
    """Exact match first; if not found, fuzzy fallback for common typos."""
    want = _clean_name(name)
    db = _load()
    for r in db:
        if _match(r["name"], want):
            return r
    # fuzzy fallback (handles e.g. "palak pannerr", "kungpao chicken")
    names = [r["name"] for r in db]
    hit = difflib.get_close_matches(want, names, n=1, cutoff=0.85)
    if hit:
        return next((r for r in db if _match(r["name"], hit[0])), None)
    return None

def _coerce_payload(payload):
    if isinstance(payload, dict):
        return payload
    s = str(payload or "").strip()
    import json
    try:
        return json.loads(s)
    except Exception:
        # salvage the first {...}
        start, end = s.find("{"), s.rfind("}") + 1
        if start >= 0 and end > start:
            cand = s[start:end]
            if '"' not in cand and "'" in cand:
                cand = cand.replace("'", '"')
            return json.loads(cand)
        raise ValueError(f"Invalid JSON payload: {s[:120]}...")

# ── name canonicalization for coverage checks ─────────────────────────────
def _canon(name: str) -> str:
    """
    Canonicalize an ingredient/pantry token to a base form:
    - import canonical_item_name lazily to avoid circular imports,
    - lowercase,
    - very simple singularization (tomatoes -> tomato).
    """
    # Lazy import to break the cycle with manager_tools
    try:
        from tools.manager_tools import canonical_item_name as _canon_name
    except Exception:
        # Safe fallback if manager_tools isn't available yet
        def _canon_name(x: str) -> str:
            return (x or "").strip().lower()

    base = _canon_name(name) if name else ""
    s = (base or "").strip().lower()
    if s.endswith("ies"):
        s = s[:-3] + "y"
    elif s.endswith("s") and len(s) > 3:
        s = s[:-1]
    return s

# ── LangChain tools ────────────────────────────────────────────────────────
@tool
def get_recipe(name: str) -> str:
    """Return one full recipe (ingredients & steps) or an error."""
    name = _clean_name(name)
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

def _canon(s: str) -> str:
    return canonical_key(s)

@tool
def find_recipes_by_items(payload: dict | str) -> str:
    """
    Suggest up to *k* recipes that best match the given *items* list.

    Accepts either a dict or a JSON string:
    {"items":["chicken"], "cuisine":"indian", "max_time": null, "diet": null, "k": 5}

    Ranking policy (strict):
    • First, show all recipes whose ingredient set is 100% covered by the pantry items (after canonicalization).
    • If no recipe is 100% covered, show partial matches ranked by: more items covered, then shorter total time, then name.
    """
    data    = _coerce_payload(payload)
    items   = data.get("items") or []
    cuisine = data.get("cuisine")
    max_time = data.get("max_time")
    diet    = data.get("diet")
    k       = int(data.get("k", 5) or 5)

    if isinstance(items, str):
        s = items.strip()
        try:
            data = json.loads(s) if (s.startswith("{") and s.endswith("}")) else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            items   = data.get("items", items if isinstance(items, list) else [])
            cuisine = data.get("cuisine", cuisine)
            max_time = data.get("max_time", max_time)
            diet    = data.get("diet", diet)
            k       = data.get("k", k)
        else:
            items = [w.strip() for w in re.split(r"[,;\n]", s) if w.strip()]

    items = [s.strip() for s in (items or []) if s and s.strip()]

    # ---- load & filter candidates
    recipes = _load()
    if diet:
        recipes = [r for r in recipes if diet_ok(r.get("diet"), diet)]
    if cuisine:
        recipes = [r for r in recipes if r.get("cuisine", "").lower() == cuisine.lower()]
    if max_time is not None:
        recipes = [
            r for r in recipes
            if r.get("prep_time_min", 0) + r.get("cook_time_min", 0) <= max_time
        ]

    # ---- fallback: no pantry items provided -> shortest total time
    if not items:
        recipes_sorted = sorted(
            recipes,
            key=lambda r: (r.get("prep_time_min", 0) + r.get("cook_time_min", 0), r.get("name", "")),
        )[:k]
        return (
            "\n".join(f"- {r['name'].title()} ({r['cuisine']})" for r in recipes_sorted)
            or "📭 No recipes match those filters."
        )

    # ---------- Canonicalize and rank ----------
    have_set = set(canonicalize_many(items))  # spaCy primary → inflect fallback
    ranked = []  # (is_full_cover: bool, covered_count: int, total_time: int, recipe: dict, coverage_ratio: float)

    for r in recipes:
        need_set = {
            canonical_key(i.get("item", ""))
            for i in (r.get("ingredients") or [])
            if (i.get("item") or "").strip()
        }
        need_set.discard("")
        total_need = len(need_set)
        if total_need == 0:
            continue

        covered_cnt = len(need_set & have_set)
        is_full = (covered_cnt == total_need)
        total_time = int(r.get("prep_time_min", 0)) + int(r.get("cook_time_min", 0))
        ratio = covered_cnt / total_need
        ranked.append((is_full, covered_cnt, total_time, r, ratio))

    if not ranked:
        return "📭 No recipes match those items."

    # Sort: 100% coverage first, then more items covered, then quicker, then name
    ranked.sort(key=lambda t: (not t[0], -t[1], t[2], (t[3].get("name") or "").lower()))

    # Optional bias by requested diet (only meaningful for "non-veg" preference)
    def _diet_rank(recipe_diet: str, user_want: Optional[str]) -> int:
        want = _normalise_diet(user_want)
        r    = _normalise_diet(recipe_diet)
        if want == "non-veg":
            order = {"non-veg": 0, "eggtarian": 1, "veg": 2}
            return order.get(r, 3)
        return 0

    full = [t for t in ranked if t[0]]
    partial = [t for t in ranked if not t[0]]

    if diet:
        full.sort(key=lambda t: (_diet_rank(t[3].get("diet", ""), diet), t[2], (t[3].get("name") or "").lower()))
        partial.sort(key=lambda t: (_diet_rank(t[3].get("diet", ""), diet), -t[1], t[2], (t[3].get("name") or "").lower()))

    bucket = full if full else partial
    top = bucket[:k]

    return "\n".join(
        f"- {t[3]['name'].title()} ({t[3]['cuisine']}) — {round(t[4] * 100):>3}% ingredients covered"
        for t in top
    )
