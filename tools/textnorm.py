# tools/textnorm.py
from __future__ import annotations
import os, re
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Optional spaCy pipeline
# ─────────────────────────────────────────────────────────────────────────────
_NLP = None
_SPACY_ERR = None

def _lazy_load_spacy():
    global _NLP, _SPACY_ERR
    if _NLP is not None or _SPACY_ERR is not None:
        return
    try:
        import spacy  # type: ignore
        model = os.getenv("SPACY_MODEL", "en_core_web_sm")
        _NLP = spacy.load(model, disable=["ner", "textcat"])  # fast + we only need tagger/dep/lemmatizer
    except Exception as e:
        _SPACY_ERR = e
        _NLP = None

def has_spacy() -> bool:
    _lazy_load_spacy()
    return _NLP is not None

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight fallback singularizer (no corpora)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import inflect  # type: ignore
    _INFLECT = inflect.engine()
except Exception:
    _INFLECT = None

def _singular_fallback(word: str) -> str:
    w = word.strip().lower()
    if not w:
        return w
    # High-value morphology folds
    if w in ("leaves", "leave"):  # curry leaves → curry leaf
        return "leaf"
    # inflect handles many irregulars (children→child, cookies→cookie, etc.)
    if _INFLECT:
        s = _INFLECT.singular_noun(w)
        if isinstance(s, str) and s:
            return s
    # crude endings if inflect missing
    if re.search(r"([^aeiou]ies)$", w):
        return re.sub(r"ies$", "y", w)
    if re.search(r"(ches|shes|xes|zes|ses)$", w):
        return re.sub(r"es$", "", w)
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

# ─────────────────────────────────────────────────────────────────────────────
# Spelling/orthography folds (tiny, surgical; NOT a big alias map)
# ─────────────────────────────────────────────────────────────────────────────
# Normalize chili/chile/chilli/chillies → chili
STATE_ADJS = {"cooked","boiled","steamed","raw","dried","fresh","smoked","roasted","grilled","fried","baked"}
CHILI_ALIASES = {
    "chilli":"chili", "chillies":"chili", "chilly":"chili", "chilles":"chili",
    "chile":"chili", "chiles":"chili"
}
_CHILI_RE = re.compile(r"^chil(?:i|ie|ies|y|li|lies|lies?)$", re.I)

def _fold_token_spelling(tok: str) -> str:
    t = tok.strip().lower()
    if not t:
        return t
    if _CHILI_RE.match(t):
        return "chili"
    # Common ASCII unifications
    t = t.replace("’", "'")
    return t

# ─────────────────────────────────────────────────────────────────────────────
# Descriptor words to drop (don’t change identity)
# Keep identity adjectives like colors/cuisines separately (see KEEP_AMOD)
# ─────────────────────────────────────────────────────────────────────────────
# keep this line, just ensure it includes 'ground'
STATE_ADJS = {"cooked","boiled","steamed","raw","dried","fresh","smoked","roasted","grilled","fried","baked","ground"}

# remove all of those from DROP_DESCRIPTORS (so they aren’t discarded)
DROP_DESCRIPTORS = {
    "dry", "powdered", "grated", "minced", "crushed",
    "sliced", "chopped", "large", "small", "medium", "boneless", "skinless",
    "uncooked", "unsalted", "salted", "sweetened",
    "unsweetened", "canned", "frozen", "ripe", "peeled", "whole",
}


# Identity adjectives we DO keep when attached to the head (amod)
# e.g., thai basil, spring onion, green chili, white fish, red chili
KEEP_AMOD = {
    "thai", "indian", "chinese", "italian",  # cuisine/nationality (expand later if needed)
    "spring",
    "green", "red", "yellow", "white", "black", "brown", "purple",
}
# Identity adjectives that depend on the head noun
HEAD_SPECIFIC_KEEP = {
    "rice": {"cooked", "steamed", "boiled"},
    "noodle": {"cooked", "boiled"},
    "noodles": {"cooked", "boiled"},
    "chicken": {"ground", "minced"},
    "beef": {"ground", "minced"},
    "pork": {"ground", "minced"},
    "lamb": {"ground", "minced"},
    "mutton": {"ground", "minced"},
}
def _keep_amod_for(head_lemma: str, adj_lemma: str) -> bool:
    adj = _fold_token_spelling(adj_lemma)
    head = _fold_token_spelling(head_lemma)
    # global keepers (colors, cuisines, spring, etc.)
    if adj in KEEP_AMOD:
        return True
    # head-specific identity adjectives
    return adj in HEAD_SPECIFIC_KEEP.get(head, set())

_PARENS_RE = re.compile(r"\s*\([^)]*\)\s*")  # remove parenthetical notes

def _preclean(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("_", " ")
    s = _PARENS_RE.sub(" ", s)
    s = re.sub(r"[^\w\s'-]+", " ", s)  # keep letters, digits, _, hyphen, apostrophe
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ─────────────────────────────────────────────────────────────────────────────
# Public: canonical_key
# Canon form: "<(optional modifiers)> <head_lemma>"
# Examples:
#   "curry leaves" → "curry leaf"
#   "green chilli" → "green chili"
#   "thai basil (fresh)" → "thai basil"
#   "fish sauce" → "fish sauce"   (multiword identity preserved)
#   "basil leaf" → "basil leaf"
# ─────────────────────────────────────────────────────────────────────────────
def canonical_key(name: str) -> str:
    s = _preclean(name)
    if not s:
        return s

    _lazy_load_spacy()
    if _NLP is not None:
        doc = _NLP(s)
        # Heuristic: pick the rightmost NOUN/PROPN as head; else last token
        head = None
        for tok in reversed(doc):
            if tok.pos_ in ("NOUN", "PROPN"):
                head = tok
                break
        head = head or doc[-1]

        # Collect left modifiers that are identity-bearing
        kept_pairs = []  # (token_index, surface)
        for lt in head.lefts:
            lem = _fold_token_spelling(lt.lemma_)
            dep = lt.dep_
            # always keep compounds (e.g., 'fish' in 'fish sauce', 'spring' in 'spring onion')
            if dep == "compound":
                kept_pairs.append((lt.i, lem))
                continue
            
            # keep state adjectives like 'cooked', 'dried', 'ground' when used as adjectival mods
            if dep == "amod" and _keep_amod_for(head.lemma_, lt.lemma_):
                kept_pairs.append((lt.i, lem))
                continue


        kept_pairs.sort(key=lambda t: t[0])
        left_mods = [t[1] for t in kept_pairs]

        head_lemma = _fold_token_spelling(head.lemma_)
        head_lemma = _singular_fallback(head_lemma)

        parts = [p for p in left_mods if p and p not in DROP_DESCRIPTORS]
        parts.append(head_lemma)
        canon = " ".join(parts).strip()
        return canon or head_lemma


    # ── Fallback path (no spaCy) ────────────────────────────────────────────
    # Split, drop non-identity descriptors, keep up to a bigram (identity + head)
    toks = [t for t in re.split(r"\s+", s) if t]
    toks = [_fold_token_spelling(t) for t in toks if t not in DROP_DESCRIPTORS]
    if not toks:
        return ""
    if len(toks) == 1:
        return _singular_fallback(toks[0])
    # keep last two tokens as identity bigram, singularize the head
    prefix = toks[-2]
    head = _singular_fallback(toks[-1])
    return f"{prefix} {head}".strip()

# Convenience helpers
def canonicalize_many(names: List[str]) -> List[str]:
    out = []
    for n in names or []:
        c = canonical_key(n)
        if c:
            out.append(c)
    return out

def canonical_and_unit(item: str, unit: str) -> Tuple[str, str]:
    """Return (canonical_name, normalized_unit('g'|'ml'|'count'))."""
    u = (unit or "").strip().lower()
    if u in ("g", "gram", "grams", "gms", "kg", "kilogram", "kilograms"):
        nu = "g"
    elif u in ("ml", "milliliter", "milliliters", "millilitre", "millilitres", "l", "liter", "liters", "litre", "litres"):
        nu = "ml"
    else:
        nu = "count"
    return canonical_key(item), nu
