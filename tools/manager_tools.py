"""Wrapper tools so ManagerAgent can call PantryAgent & CuisineAgent and keep a short-term slot memory (current dish, last inventory)."""
import re
from langchain_core.tools import tool
from langchain.memory import SimpleMemory
from difflib import get_close_matches 
from tools.cuisine_tools import _load 
from difflib import SequenceMatcher

# ----------------------------------------------------------
memory = SimpleMemory(memories={})      
# ----------------------------------------------------------

def _push_recent(dish: str, limit: int = 10):
    lst = memory.memories.setdefault("recent_dishes", [])
    if dish not in lst:          
        lst.append(dish)
        if len(lst) > limit:
            del lst[0]
    memory.memories["current_dish"] = dish


def _capture_possible_dish(text: str):
    """Find one recipe name in free-text (user or agent) and store it."""
    clean = re.sub(r"[-*•]\s*", "", text.lower()).strip()
    m = re.search(r"(?:cook|make|prepare|try)\s+(.+?)(?:[,?.!]|$)", clean)
    guess = m.group(1).strip() if m else clean

    names = [r["name"].lower() for r in _load()]
    match = get_close_matches(guess, names, n=1, cutoff=0.95)
    if match:
        dish = match[0]
        memory.memories["current_dish"] = dish
        _push_recent(dish)

def _capture_dish_list(text: str):
    names = {r["name"].lower() for r in _load()}
    for chunk in re.split(r"[,\n]", text):
        cand = re.sub(r"^[\d\-\*•.]+\s*", "", chunk.lower()).split("(")[0].strip()
        if cand in names:
            _push_recent(cand)



def _parse_inventory(text: str) -> list[str]:
    """
    Extract item names from either
      • multi-line bullet list
      • one-line comma list returned by PantryAgent
    Returns lower-case names, e.g. ["apple", "orange", "banana"].
    """
    # if the response contains ':' treat everything **after** the first colon
    # as the comma-separated list
    if ":" in text:
        text = text.split(":", 1)[1]

    names = []
    # replace newlines with commas so we have a uniform delimiter
    for chunk in text.replace("\n", ",").split(","):
        m = re.match(r"\s*([a-zA-Z][a-zA-Z\s]*)\s*\(", chunk)
        if m:
            names.append(m.group(1).strip().lower())
    return names


# ── Lazy chat imports to avoid circular refs --------------------------------
def _pantry_chat(msg: str) -> str:
    from agents.pantry_agent import chat as pantry_chat
    return pantry_chat(msg)

def _cuisine_chat(msg: str) -> str:
    from agents.cuisine_agent import chat as cuisine_chat
    return cuisine_chat(msg)

# ── Tools -------------------------------------------------------------------

import time

@tool
def call_pantry(message: str) -> str:
    """Forward *message* to PantryAgent and refresh inventory cache."""
    _capture_possible_dish(message)       
    reply = _pantry_chat(message)

    if message.lower().startswith("list"):
        items = _parse_inventory(reply)
        memory.memories.update({
            "last_inventory_items": items,
            "inv_timestamp": time.time(),
        })
    elif message.lower().startswith(("add", "remove", "update")):
        inv_txt = _pantry_chat("list pantry")
        items   = _parse_inventory(inv_txt)
        memory.memories.update({
            "last_inventory_items": items,
            "inv_timestamp": time.time(),
        })
    return reply


_SIM_THRESHOLD = 0.85 
@tool
def call_cuisine(message: str) -> str:
    """Forward *message* to CuisineAgent and refresh inventory cache."""
    _capture_possible_dish(message)
    reply = _cuisine_chat(message)

    # --- similarity guard --------------------------------------------------
    m = re.search(r"\*\*(.+?)\*\*", reply)
    if m:
        recipe_name = m.group(1).strip().lower()
        # strip helper words from the original user message
        requested = re.sub(r"(?:recipe|how to (?:cook|make))", "",
                           message, flags=re.I).strip().lower()
        if SequenceMatcher(None, recipe_name, requested).ratio() < _SIM_THRESHOLD:
            # treat as 'not found' -> ask CuisineAgent for suggestions
            alt = _cuisine_chat(f'suggest_similar "{requested}" top_k=5')
            return ("Apologies — I don’t have an exact recipe for that dish.\n\n"
                    f"Here are some close alternatives you could try:\n{alt}")

        _push_recent(recipe_name)  # we accept it as a good match

    _capture_dish_list(reply)
    return reply

_word_re = re.compile(r"[a-zA-Z]+")
BULLET = re.compile(r"^\s*([-*•]|•|\d+\.)\s*")
def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") else word
 
def _extract_ing_names(recipe_txt: str) -> set[str]:
    """Return the set of (raw) ingredient names found in recipe bullets."""
    names: set[str] = set()
    for line in recipe_txt.splitlines():
        if not BULLET.match(line):
            continue                    # not a list item
        words = [w.lower() for w in _word_re.findall(line)]
        if not words:
            continue
        last = _singular(words[-1])
        names.add(last)
        if len(words) >= 2:
            names.add(_singular(" ".join(words[-2:])))
    return names

DESCRIPTORS = {"cooked", "fresh", "dried", "ground", "chopped", "sliced", "large","medium", 
               "small", "whole", "raw", "ripe", "frozen", "canned", "baked", "steamed", "boiled", "grilled", "roasted",
               "g","kg","ml","l"}

def _normalise(name: str) -> str:
    """strip leading descriptors, lower-case, singularise."""
    words = [w.lower() for w in name.split() if w.lower() not in DESCRIPTORS]
    base  = " ".join(words) if words else name.lower()
    # simple plural → singular
    if base.endswith("ies"):              # berries -> berry
        base = base[:-3] + "y"
    elif base.endswith(("es", "s")) and len(base) > 3:
        base = base[:-1]                  # onions -> onion, eggs -> egg
    return base


@tool
def missing_ingredients(dish: str) -> str:
    """
    Tell the user which ingredients for *dish* are not in their pantry.
    """

    # --- ensure we have an up-to-date pantry snapshot ----------------------
    inv_items = memory.memories.get("last_inventory_items")
    if inv_items is None:
        _ = call_pantry("list pantry")
        inv_items = memory.memories.get("last_inventory_items", [])

    pantry = {_normalise(it) for it in inv_items}

    # --- fetch recipe ------------------------------------------------------
    recipe_txt = call_cuisine(f"get_recipe {dish}")
    if "Ingredients" not in recipe_txt:          # fallback: bullets only
        recipe_txt = call_cuisine(
            f'give ingredients list only for "{dish}"'
        )

    if recipe_txt.strip().startswith("⚠️"):
        return recipe_txt                        # CuisineAgent already apologised

    need = _extract_ing_names(recipe_txt)
    if not need:                                # still couldn’t read list
        return (f"Sorry, I couldn’t read the ingredient list for "
                f"{dish.title()}. Could you try another recipe?")

    # --- compare -----------------------------------------------------------
    collapsed = {}
    for item in need:
        base = _normalise(item)
        collapsed[base] = min(collapsed.get(base, item), item, key=len)

    missing = sorted(
        {base: pretty for pretty in need
         if (base := _normalise(pretty)) not in pantry}.values()
    )

    # --- user-friendly response -------------------------------------------
    if not missing:
        return f"You already have every ingredient for {dish.title()}!"
    if len(missing) == 1:
        return (f"You'll still need {missing[0]} " f" to cook {dish.title()}.")
    *rest, last = missing
    return (f"You'll still need {', '.join(rest)} and {last}" f" to cook {dish.title()}.")
