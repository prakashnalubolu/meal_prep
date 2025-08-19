"""
Lightweight manager utilities kept for shared memory and the string-only
missing_ingredients tool. No agent-to-agent wrappers are used anymore.
"""

from __future__ import annotations
import json, os, re
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.tools import tool
from langchain.memory import SimpleMemory

# Expose a small memory object so the UI can still show "Manager slots".
memory: SimpleMemory = SimpleMemory(memories={})

# --------------------------------------------------------------------- Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PANTRY_JSON_PATH = os.path.join(ROOT_DIR, "data", "pantry.json")

# ----------------------------------------------------------------- Helpers

_name_unit_re = re.compile(r"^\s*(.*?)\s*\(([^)]+)\)\s*$")

def _split_pantry_key(key: str) -> Tuple[str, str]:
    """'tomato (count)' -> ('tomato', 'count')  |  'rice (g)' -> ('rice','g')"""
    m = _name_unit_re.match(key)
    if not m:
        base = key.split("(")[0]
        return base.strip().lower(), "count"
    return m.group(1).strip().lower(), m.group(2).strip().lower()

def _normalize_unit(u: Optional[str]) -> str:
    if not u:
        return "count"
    u = u.strip().lower()
    if u in ("kg", "kilogram", "kilograms"):
        return "g"
    if u in ("g", "gram", "grams", "gms"):
        return "g"
    if u in ("l", "litre", "liter", "liters", "litres"):
        return "ml"
    if u in ("ml", "millilitre", "milliliter", "milliliters", "millilitres"):
        return "ml"
    if u in ("count", "pcs", "piece", "pieces"):
        return "count"
    return u  # leave as-is for any custom units

def _normalise(name: str) -> str:
    """Lower-case and strip very simple plurals (onions → onion)."""
    n = name.strip().lower()
    if n.endswith("ies"):
        n = n[:-3] + "y"
    elif n.endswith("s") and len(n) > 3:
        n = n[:-1]
    return n

# Generic descriptors we drop for base-name matching (kept intentionally short)
_DESCRIPTORS = {
    "white", "boneless", "skinless", "lean", "fresh", "frozen", "dried",
    "ground", "powdered", "powder", "whole", "sliced", "chopped", "fillet", "fillets",
    "medium", "large", "small","red", "green", "yellow", "black", "brown",
}

# A *tiny* alias map (not a big dictionary) to collapse very common variants
_ALIASES = {
    "chilli": "chili", "chilies": "chili", "chillies": "chili",
    "scallion": "spring onion", "scallions": "spring onion",
    "coriander leaves": "coriander leave", "cilantro": "coriander leave",
    "curry leave": "curry leaf",
    "curry leaves": "curry leaf",
}

_plural_re = re.compile(r"(?i)(ies|s)$")

def _depluralize(w: str) -> str:
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

# ---------------------- Recipe access (structured, no agent hop)
from tools.cuisine_tools import _load as _load_recipes

def _load_recipe_by_name(name: str) -> Optional[Dict[str, Any]]:
    name = _clean_name(name)
    name_l = name.strip().lower()
    for r in _load_recipes():
        if r["name"].strip().lower() == name_l:
            return r
    return None

# ----------------------------------------------------------------- Tool: gaps
def _clean_name(s: str) -> str:
    s = str(s or "").strip()
    # strip balanced outer quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    # strip any stray quotes/whitespace and collapse spaces
    s = s.strip('\'"\n\r\t ')
    s = re.sub(r"\s+", " ", s)
    return s


def _canonical_item_name(name: str) -> str:
    """Lowercase, drop generic descriptors, collapse trivial aliases, depluralize."""
    s = _clean_name(name).lower()
    # collapse multiword aliases first
    for k, v in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"\b{k}\b", v, s)
    # drop descriptors
    tokens = [t for t in re.split(r"\W+", s) if t]
    tokens = [t for t in tokens if t not in _DESCRIPTORS]
    # depluralize each token (lightweight)
    tokens = [_depluralize(t) for t in tokens]
    # heuristics: keep up to two words for things like "spring onion"
    if not tokens:
        return ""
    if len(tokens) >= 2 and "spring" in tokens and "onion" in tokens:
        return "spring onion"
    if len(tokens) >= 2 and tokens[-2] == "fish" and tokens[-1] == "fillet":
        return "fish"
    # fallback: last token as head noun
    return " ".join(tokens[-2:]) if len(tokens) > 1 else tokens[-1]

def canonical_item_name(name: str) -> str:
    return _canonical_item_name(name)

# ------------------------------- Pantry IO ----------------------------------
def _load_pantry() -> Dict[str, int]:
    try:
        with open(PANTRY_JSON_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            return {k: int(v) for (k, v) in data.items()}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def _find_matching_key(pantry: Dict[str, int], item: str, unit: str) -> Optional[str]:
    """Exact base-name + unit match after canonicalization."""
    base = _canonical_item_name(item)
    unit = _normalize_unit(unit)
    for k in pantry.keys():
        b, u = _split_pantry_key(k)
        if _canonical_item_name(b) == base and u == unit:
            return k
    return None

# ----------------------------- Tools ----------------------------------------
@tool
def missing_ingredients(dish: str) -> str:
    """
    Tell the user which ingredients for *dish* are not in their pantry.
    STRING-ONLY input. Returns a short natural-language sentence.

    Matching rules:
    • Name matching uses canonical base names (descriptor-stripped) + exact unit family (g/ml/count).
    • Units are normalized (kg→g, l→ml, default count).
    • This is a strict check (no creative swaps). Use `suggest_substitutions` to propose alternatives.
    """
    dish = _clean_name(dish)
    if not isinstance(dish, str) or not dish:
        return "Please provide a dish name."

    recipe = _load_recipe_by_name(dish)
    if not recipe:
        return f"⚠️ Recipe '{dish}' not found."

    pantry = _load_pantry()

    deficits: List[str] = []
    for ing in recipe.get("ingredients", []):
        raw_item = str(ing.get("item", "")).strip()
        if not raw_item:
            continue
        need_qty = int(ing.get("quantity", 0) or 0)
        unit = _normalize_unit(ing.get("unit") or "count")
        if need_qty <= 0:
            continue

        key = _find_matching_key(pantry, raw_item, unit)
        if key is None:
            # Entire amount missing
            deficits.append(f"{need_qty} {unit} {raw_item}")
            continue

        have = int(pantry.get(key, 0))
        if have < need_qty:
            deficits.append(f"{need_qty - have} {unit} {raw_item}")

    dish_title = dish.strip().title()
    if not deficits:
        return f"You already have every ingredient for {dish_title}!"
    if len(deficits) == 1:
        return f"You'll still need {deficits[0]} to cook {dish_title}."
    *rest, last = deficits
    return f"You'll still need {', '.join(rest)} and {last} to cook {dish_title}."

# ---------- Substitution suggester (schema-bound, deterministic heuristics) --
def _coerce_payload(payload: dict | str) -> dict:
    if isinstance(payload, dict):
        return payload
    s = str(payload or "").strip()
    try:
        return json.loads(s)
    except Exception:
        # salvage the first {...}
        a, b = s.find("{"), s.rfind("}") + 1
        if a >= 0 and b > a:
            return json.loads(s[a:b])
        raise ValueError("Invalid JSON payload for suggest_substitutions")

def _aggregate_pantry_by_base(pantry: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """
    Returns { base_item: {unit: qty, ...}, ... } using canonical base names.
    """
    out: Dict[str, Dict[str, int]] = {}
    for k, v in pantry.items():
        b, u = _split_pantry_key(k)
        base = _canonical_item_name(b)
        out.setdefault(base, {})
        out[base][u] = int(v)
    return out

def _prep_note_for(missing_raw: str, base: str) -> str:
    s = _clean_name(missing_raw).lower()
    if "fillet" in s and "fish" in base:
        return "cut into boneless fillets; remove skin if present"
    if "dried" in s and "chili" in base:
        return "dry-roast chilies 2–3 min to mimic dried heat"
    return ""

def _confidence_for(missing_raw: str, base: str) -> float:
    s = _clean_name(missing_raw).lower()
    if "fillet" in s and "fish" in base:
        return 0.84
    if "dried" in s and "chili" in base:
        return 0.75
    return 0.70  # default for close base-name matches

@tool
def suggest_substitutions(payload: dict | str) -> str:
    """
    Propose substitutions for remaining deficits using the user's pantry.

    Input (dict or JSON string):
    {
      "dish": "Kung Pao Chicken",
      "deficits": [{"item":"dried chili","need_qty":5,"unit":"count"}],
      "pantry": [{"item":"red chili","qty":12,"unit":"count"}, ...],
      "constraints": {"allow_prep": true, "max_subs_per_item": 2}
    }

    Output (JSON string):
    {"subs":[
      {"missing":"dried chili",
       "use":[{"item":"red chili","qty":5,"unit":"count"}],
       "prep":"dry-roast 2–3 min",
       "confidence":0.78,
       "reason":"Close variant; roasting approximates dried"}
    ]}
    """
    data = _coerce_payload(payload)
    deficits = data.get("deficits") or []
    pantry_list = data.get("pantry") or []
    allow_prep = bool((data.get("constraints") or {}).get("allow_prep", True))

    # Build a base-name index for the pantry
    # Prefer the snapshot passed in; fall back to file.
    if pantry_list:
        pantry: Dict[str, int] = {}
        for p in pantry_list:
            k = f"{_canonical_item_name(p.get('item',''))} ({_normalize_unit(p.get('unit'))})"
            pantry[k] = pantry.get(k, 0) + int(p.get("qty", 0))
    else:
        pantry = _load_pantry()

    pantry_by_base = _aggregate_pantry_by_base(pantry)

    results: List[Dict[str, Any]] = []

    for d in deficits:
        raw_item = str(d.get("item",""))
        unit = _normalize_unit(d.get("unit") or "count")
        need_qty = int(d.get("need_qty", 0) or 0)
        if not raw_item or need_qty <= 0:
            continue

        base = _canonical_item_name(raw_item)

        # 1) If pantry already has the base in the same unit family, suggest direct use
        unit_map = pantry_by_base.get(base, {})
        if unit in unit_map and unit_map[unit] >= need_qty:
            results.append({
                "missing": raw_item,
                "use": [{"item": base, "qty": need_qty, "unit": unit}],
                "prep": "",
                "confidence": 0.9,
                "reason": "Same ingredient available under a variant name."
            })
            continue

        # 2) Heuristic generic swaps (no giant dictionary)
        #    fish fillet -> fish; dried chili -> chili
        #    Only if we actually have the base in pantry.
        if base in pantry_by_base and allow_prep:
            prep = _prep_note_for(raw_item, base)
            conf = _confidence_for(raw_item, base)
            # choose the same unit if present; otherwise pick any available unit (agent can rely on alt-units)
            pick_unit = unit if unit in pantry_by_base[base] else (next(iter(pantry_by_base[base].keys())) if pantry_by_base[base] else unit)
            results.append({
                "missing": raw_item,
                "use": [{"item": base, "qty": need_qty, "unit": pick_unit}],
                "prep": prep,
                "confidence": conf,
                "reason": "Close culinary equivalent; simple prep bridges the gap."
            })
            continue

        # 3) Nothing reasonable
        #    (We intentionally do NOT fabricate substitutes.)
        #    Skip adding an entry.

    return json.dumps({"subs": results}, ensure_ascii=False)
