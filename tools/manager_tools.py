"""Wrapper tools so ManagerAgent can call PantryAgent & CuisineAgent and keep a short-term slot memory (current dish, last inventory)."""

import re
from typing import List
from langchain_core.tools import tool
from langchain.memory import SimpleMemory
from difflib import get_close_matches 
from tools.cuisine_tools import _load 

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
    match = get_close_matches(guess, names, n=1, cutoff=0.6)
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

@tool
def call_cuisine(message: str) -> str:
    """Forward *message* to CuisineAgent and update dish memory."""
    _capture_possible_dish(message)        # user prompt
    reply = _cuisine_chat(message)
    _capture_dish_list(reply)              # bullet / numbered list
    # if reply contains exactly one **RecipeName**
    m = re.search(r"\*\*(.+?)\*\*", reply)
    if m:
        dish = m.group(1).strip().lower()
        memory.memories["current_dish"] = dish        # focus!
        _push_recent(dish)
    return reply

_word_re = re.compile(r"[a-zA-Z]+")

def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") else word

def _extract_ing_names(recipe_txt: str) -> set[str]:
    names: set[str] = set()
    for line in recipe_txt.splitlines():
        if line.lstrip().startswith("-"):
            words = [w.lower() for w in _word_re.findall(line)]
            if not words:
                continue
            last = _singular(words[-1])
            names.add(last)                    
            if len(words) >= 2:
                two = " ".join(words[-2:])
                names.add(_singular(two))       
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
    List which ingredients for *dish* are NOT in the cached pantry.
    Returns a friendly sentence; if nothing is missing, says so.
    """

    # 1) get cached inventory — refresh once if absent
    inv_items = memory.memories.get("last_inventory_items")
    if inv_items is None:
        _ = call_pantry("list pantry")            # populates memory
        inv_items = memory.memories.get("last_inventory_items", [])

    pantry = {_normalise(it) for it in inv_items}

    # fetch full recipe (fuzzy capture handled in call_cuisine)
    recipe_txt = call_cuisine(f"get_recipe {dish}")
    if "Ingredients" not in recipe_txt:                   # ⇦  new robust fallback
        # try forcing the cuisine agent to send the bullet version
        recipe_txt = call_cuisine(f'give ingredients list only for "{dish}"')

    # if the cuisine agent couldn’t find the recipe, bubble up the warning
    if recipe_txt.strip().startswith("⚠️"):
        return recipe_txt

    need     = _extract_ing_names(recipe_txt)
    # map each raw ingredient to its normalised key
    collapsed: dict[str, str] = {}
    for item in need:
        base = _normalise(item)
        collapsed[base] = min(collapsed.get(base, item), item, key=len)
    need = set(collapsed.values())
    missing = sorted({base: item for item in need if (base := _normalise(item)) not in pantry}.values())  

    if not missing:
        return f"You already have every ingredient for {dish.title()}!"
    if len(missing) == 1:
        return f"You'll still need {missing[0]} to cook {dish.title()}."

    *rest, last = missing
    return f"You'll still need {', '.join(rest)} and {last} to cook {dish.title()}."