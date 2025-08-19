# Run with:  streamlit run app.py
from __future__ import annotations
import os, json, re, datetime
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

from agents.kitchen_agent import chat as kitchen_chat

# --- Planner tools & memories
from tools.meal_plan_tools import (
    memory as planner_memory,         # plan / shopping list / logs
    update_plan, cook_meal,
    get_shopping_list, save_plan,
)

# (Optional) manager slot-memory; useful to fully reset caches
try:
    from tools.manager_tools import memory as slot_memory
except Exception:
    slot_memory = None  # safe if missing

# --- DRY: reuse cuisine helpers from tools (read-only in the UI)
from tools.cuisine_tools import _load as cuisine_load
from tools.cuisine_tools import diet_ok as cuisine_diet_ok

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kitchen Chat: • Pantry • Recipes • Meal Plans", page_icon="🍳", layout="wide")
st.markdown("## 🍳 Kitchen Chat: • Pantry • Recipes • Meal Plans")

# Style just the reset button we'll wrap in #reset-fab
st.markdown("""
<style>
#reset-fab button{
  border-radius:999px; width:36px; height:36px; padding:0;
  background:#e8f1ff; color:#1b64f2; border:1px solid #c8ddff;
}
#reset-fab button:hover{ background:#d9e9ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages_kitchen", [])     # [{"role":"user|assistant","content": "..."}]
ss.setdefault("events", [])               # [{"label": str, "msg_idx": int}]
ss.setdefault("focus_msg_idx", None)
ss.setdefault("show_pantry_json", False)
ss.setdefault("show_cuisine_json", False)
ss.setdefault("start_date", datetime.date.today())
ss.setdefault("cuisine_autofocus", "")    # when you click a dish in the plan board

# ─────────────────────────────────────────────────────────────────────────────
# Paths & helpers
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PANTRY_PATH = os.path.join(DATA_DIR, "pantry.json")

KEY_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

def _load_json_ok(path: str) -> Tuple[bool, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return True, json.load(f)
    except FileNotFoundError:
        return False, f"Missing: {os.path.relpath(path, BASE_DIR)}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in {os.path.relpath(path, BASE_DIR)} · {e}"

ALT_UNITS_PATH = os.path.join(DATA_DIR, "alt_units.json")
def _load_alt_hints() -> dict:
    try:
        with open(ALT_UNITS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        data = {}
    rules = data.get("rules", []) or []
    labels = (data.get("labels", {}) or {})
    hints: Dict[str, Dict[str, Any]] = {}
    base_count_aliases = ["count", "pc", "pcs", "piece", "pieces"]
    for r in rules:
        item = str(r.get("item","")).strip().lower()
        fr   = str(r.get("from","")).strip().lower()
        to   = str(r.get("to","")).strip().lower()
        factor = r.get("factor", None)
        if not item or fr != "count" or factor is None:
            continue
        h = hints.setdefault(item, {"count_to_g": None, "count_to_ml": None, "count_aliases": list(base_count_aliases)})
        if to == "g":
            h["count_to_g"] = float(factor)
        elif to == "ml":
            h["count_to_ml"] = float(factor)
        lbl = (labels.get(item) or {}).get("count_label")
        if lbl:
            lbl = str(lbl).strip().lower()
            if lbl and lbl not in h["count_aliases"]:
                h["count_aliases"].append(lbl)
    return hints

ALT_HINTS = _load_alt_hints()

def _pretty_quantity(item: str, unit: str, qty: Any) -> str:
    try:
        q = float(qty)
    except Exception:
        return f"{qty} {unit}"
    name = (item or "").strip().lower()
    u    = (unit or "").strip().lower()
    h    = ALT_HINTS.get(name)
    def _fmt(x: float, suffix: str) -> str:
        return f"{int(x)} {suffix}" if abs(x - int(x)) < 1e-9 else f"{x:g} {suffix}"
    if not h:
        return _fmt(q, u)
    if u == "g" and h.get("count_to_g"):
        approx_cnt = q / float(h["count_to_g"])
        label = (ALT_HINTS.get(name, {}).get("count_aliases") or ["count"])[-1]
        return f"{_fmt(q, 'g')} (~{_fmt(round(approx_cnt), label)})"
    if u == "ml" and h.get("count_to_ml"):
        approx_cnt = q / float(h["count_to_ml"])
        label = (ALT_HINTS.get(name, {}).get("count_aliases") or ["count"])[-1]
        return f"{_fmt(q, 'ml')} (~{_fmt(round(approx_cnt), label)})"
    if h and u in (h.get("count_aliases") or []):
        if h.get("count_to_g"):
            approx_g = q * float(h["count_to_g"])
            return f"{_fmt(q, u)} (~{_fmt(approx_g, 'g')})"
        if h.get("count_to_ml"):
            approx_ml = q * float(h["count_to_ml"])
            return f"{_fmt(q, u)} (~{_fmt(approx_ml, 'ml')})"
    return _fmt(q, u)

def _parse_pantry_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, v in sorted((d or {}).items(), key=lambda kv: kv[0].lower()):
        m = KEY_RE.match(k)
        if m:
            item = m.group(1).strip().lower()
            unit = m.group(2).strip().lower()
        else:
            item, unit = k.strip().lower(), "count"
        try:
            qty = int(v)
        except Exception:
            try:
                qty = float(v)
            except Exception:
                qty = v
        rows.append({"item": item, "unit": unit, "quantity": qty})
    for i, r in enumerate(rows, start=1):
        r["S.No"] = i
    return [{"S.No": r["S.No"], "item": r["item"], "unit": r["unit"], "quantity": r["quantity"]} for r in rows]

def _fmt_recipe_md(r: dict) -> str:
    name = str(r.get("name","")).title()
    cuisine = (r.get("cuisine","") or "").title()
    prep = int(r.get("prep_time_min",0))
    cook = int(r.get("cook_time_min",0))
    ings = r.get("ingredients", [])
    steps = r.get("steps", [])
    ing_lines = [
        f"- {i.get('quantity')} {i.get('unit')} {i.get('item')}" if (i.get("unit") and i.get("unit") != "count")
        else f"- {i.get('quantity')} {i.get('item')}"
        for i in ings
    ]
    step_lines = [f"{i+1}. {s}" for i, s in enumerate(steps)]
    return "\n".join([
        f"**{name}** · {cuisine} — Prep {prep} min · Cook {cook} min",
        "",
        "Ingredients:",
        *ing_lines,
        "",
        "Steps:",
        *step_lines
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Tiny event labeller for the chat list
# ─────────────────────────────────────────────────────────────────────────────
_NUM  = r"(?P<num>\d+(?:\.\d+)?)"
_UNIT = r"(?P<unit>count|counts|pcs?|pieces?|gms?|grams?|kg|ml|l)\b"
_ITEM = r"(?P<item>[a-zA-Z][a-zA-Z \-']{0,40})"

USER_PATTERNS: List[Tuple[re.Pattern, callable]] = [
    (re.compile(r"(?i)\b(what('?s| is) in|list|show).*pantry\b"), lambda m: "List pantry"),
    (re.compile(r"(?i)\b(how (many|much)|do i have|have i got)\s+(?P<item>[a-zA-Z \-']+)\??"),
     lambda m: f"Qty: {m.group('item').strip().lower()}"),
    (re.compile(fr"(?i)\badd\b\s+{_NUM}\s*(?:{_UNIT})?\s+{_ITEM}"),
     lambda m: f"Add {m.group('num')}{' '+m.group('unit') if m.groupdict().get('unit') else ''} {m.group('item').strip().lower()}"),
    (re.compile(fr"(?i)\b(?:remove|delete|take\s+out)\b\s+{_NUM}\s*(?:{_UNIT})?\s+{_ITEM}"),
     lambda m: f"Remove {m.group('num')}{' '+m.group('unit') if m.groupdict().get('unit') else ''} {m.group('item').strip().lower()}"),
    (re.compile(fr"(?i)\b(?:set|update)\b\s+{_ITEM}\s+to\s+{_NUM}\s*(?:{_UNIT})?"),
     lambda m: f"Set {m.group('item').strip().lower()} → {m.group('num')}{' '+m.group('unit') if m.groupdict().get('unit') else ''}"),
    (re.compile(r"(?i)\b(get|show|give).*(recipe|steps?)\b.*?(for|of)?\s*([a-zA-Z \-']+)$"),
     lambda m: f"Recipe: {m.group(4).strip().lower()}"),
    (re.compile(r"(?i)\b(how to (make|cook)|recipe for)\s+([a-zA-Z \-']+)$"),
     lambda m: f"Recipe: {m.group(3).strip().lower()}"),
    (re.compile(r"(?i)\b(plan|meal plan)\b"),                lambda m: "Plan meals"),
    (re.compile(r"(?i)\b(shopping list|what(?:'|)s missing|gaps?)\b"), lambda m: "Shopping list / gaps"),
    (re.compile(r"(?i)\b(mark )?cooked\b"),                 lambda m: "Cooked a dish"),
    (re.compile(r"(?i)\bexport\b"),                         lambda m: "Export plan"),
    (re.compile(r"(?i)\bwhat can i cook\b"),                lambda m: "Cookable dishes"),
]

def label_user_turn(text: str) -> str:
    s = text.strip()
    for pat, labeller in USER_PATTERNS:
        m = pat.search(s)
        if m: return labeller(m)
    words = re.findall(r"[^\s]+", s)
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")

# ─────────────────────────────────────────────────────────────────────────────
# Layout (Left: Planner | Middle: Chat | Right: Pantry + Cuisine)
# ─────────────────────────────────────────────────────────────────────────────
left, middle, right = st.columns([0.35, 0.40, 0.25], gap="large")

# LEFT — Planner

with left:
    # --- Header row: title + reset button right next to it ----
    hdr_title, hdr_btn, _spacer = st.columns([0.25, 0.06, 0.69])
    with hdr_title:
        st.markdown("### 📅 Planner")
        # small mode badge under the Planner title
        _constraints = planner_memory.memories.get("constraints", {})
        _mode_badge = "Pantry-first (strict)" if (_constraints.get("mode") or "") == "pantry-first-strict" else "Freeform"
        st.caption(f"Mode: {_mode_badge}")


    reset_done = False
    with hdr_btn:
        st.markdown('<div id="reset-fab">', unsafe_allow_html=True)
        reset_clicked = st.button("↺", key="reset_plan", help="Clear chat & plan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if reset_clicked:
            ss["messages_kitchen"].clear()
            ss["events"].clear()
            ss["focus_msg_idx"] = None
            ss["cuisine_autofocus"] = ""
            try:
                planner_memory.memories.clear()
            except Exception:
                pass
            try:
                if slot_memory is not None:
                    slot_memory.memories.clear()
            except Exception:
                pass
            reset_done = True

    # show toast at full left-pane width (not in the tiny button column)
    if reset_done:
        st.success("Cleared chat and plan.")
    ss["start_date"] = st.date_input("Plan start date", ss["start_date"], key="plan_start")

    with st.expander("Generate meal plan", expanded=False):
        days = st.number_input("Days", min_value=1, max_value=14, value=3, step=1)
        meals_per_day = st.selectbox("Meals per day",
            ["3 (Breakfast,Lunch,Dinner)", "2 (Lunch,Dinner)", "1 (Dinner only)"], index=0)
        cuisine = st.text_input("Cuisine (optional)", "")
        diet = st.selectbox("Diet", ["any", "vegetarian", "eggtarian", "non-veg"], index=0)
        max_time = st.number_input("Max cook time (minutes, optional)", min_value=0, max_value=240, value=0, step=5)
        avoid_repeats = st.checkbox("Avoid repeats", value=False)

        if st.button("Generate plan", type="primary", use_container_width=True):
            meals_hint = ("3 meals/day" if meals_per_day.startswith("3")
                          else "2 meals/day" if meals_per_day.startswith("2")
                          else "1 meal/day")
            req = f"Please generate a {int(days)}-day plan with {meals_hint}. Prefer using items already in my pantry."
            if cuisine.strip(): req += f" Cuisine: {cuisine.strip()}."
            if diet != "any":   req += f" Diet: {diet}."
            if max_time > 0:    req += f" Max cook time: {int(max_time)} minutes."
            if avoid_repeats:   req += " No repeats."
            out = kitchen_chat(req)
            st.info(out)

    # ---------- Current Plan (single component: view + edit)
    plan: Dict[str, Dict[str, str]] = planner_memory.memories.get("plan", {}) or {}
    if plan:
        edit_mode = st.toggle("✏️ Edit mode", value=False, help="Turn on to type new dish names; Save to commit.")

        def _label_with_date(day_key: str) -> str:
            try:
                n = int(re.sub(r"\D", "", day_key) or "1")
            except Exception:
                n = 1
            d = ss["start_date"] + datetime.timedelta(days=n-1)
            return f"{day_key} ({d.strftime('%a %d %b')})"

        days_sorted = sorted(plan.keys(), key=lambda d: (int(re.sub(r"\D", "", d) or 0), d))
        st.caption("Tip: Click a dish to preview it on the right. In Edit mode, type to change names, then Save.")
        # Build a grid for all days × meals
        pending_updates: List[Dict[str, str]] = []
        for day in days_sorted:
            st.markdown(f"**{_label_with_date(day)}**")
            c1, c2, c3 = st.columns(3)
            for col, meal in zip((c1, c2, c3), ("Breakfast", "Lunch", "Dinner")):
                with col:
                    dish = plan.get(day, {}).get(meal, "")
                    if not edit_mode:
                        label = dish if dish else "(empty)"
                        if st.button(label, key=f"plan_btn_{day}_{meal}", use_container_width=True, disabled=not bool(dish)):
                            ss["cuisine_autofocus"] = dish
                    else:
                        new_val = st.text_input(
                            f"{meal}",
                            value=dish,
                            key=f"edit_{day}_{meal}",
                            placeholder="Dish name…",
                        )
                        if new_val.strip() and new_val.strip() != (dish or "").strip():
                            pending_updates.append({"day": day, "meal": meal, "recipe_name": new_val.strip(), "reason": "edited in UI"})

        if edit_mode and st.button("Save edits", type="primary", use_container_width=True):
            changes = []
            for upd in pending_updates:
                try:
                    # Tool is a LangChain Tool; pass the arg by name via invoke
                    msg = update_plan.invoke({"payload": upd})
                except Exception as e:
                    msg = f"Error: {e}"
                changes.append(str(msg))
                # reflect in memory immediately to keep board in sync
                plan.setdefault(upd["day"], {})[upd["meal"]] = upd["recipe_name"]
                planner_memory.memories["plan"] = plan
            if changes:
                st.success("Updated:\n" + "\n".join(changes))
            else:
                st.info("No changes to save.")

        st.divider()

        # Quick actions under the board
        cook_c1, cook_c2, cook_c3 = st.columns(3)
        with cook_c1:
            ck_day = st.text_input("Cook: Day (optional if you give a dish)", value="")
        with cook_c2:
            ck_meal = st.selectbox("Cook: Meal", ["Breakfast","Lunch","Dinner"], index=2, key="cook_meal_select")
        with cook_c3:
            ck_dish = st.text_input("Cook: Dish (optional if day+meal provided)", value="")
        if st.button("Mark cooked", use_container_width=True):
            payload = {"dish": ck_dish.strip()} if ck_dish.strip() else {"day": ck_day.strip(), "meal": ck_meal}
            try:
                msg = cook_meal.invoke({"payload": payload})
            except Exception as e:
                msg = f"Error: {e}"
            st.info(str(msg))

        colA, colB = st.columns(2)
        with colA:
            if st.button("🛒 Get shopping list", use_container_width=True):
                try:
                    sl = get_shopping_list.invoke({"_": None})
                except Exception as e:
                    sl = f"Error: {e}"
                st.text(str(sl))
        with colB:
            file_name = st.text_input("Export filename (optional, no extension)", value="")
            if st.button("💾 Save plan", use_container_width=True):
                payload = file_name.strip() or None
                try:
                    msg = save_plan.invoke({"payload": payload})
                except Exception as e:
                    msg = f"Error: {e}"
                st.success(str(msg))
    else:
        st.caption("No plan yet — generate one above.")

# MIDDLE — Chat
with middle:
    prompt = st.chat_input("Ask anything: pantry, recipes, meal plans, shopping list, mark cooked, export…")
    if prompt:
        ss["messages_kitchen"].append({"role": "user", "content": prompt})
        ss["events"].append({"label": label_user_turn(prompt), "msg_idx": len(ss["messages_kitchen"]) - 1})

        with st.spinner("Thinking…"):
            try:
                reply = kitchen_chat(prompt)
            except Exception as err:
                reply = f"🚨 Error: {err}"
        if isinstance(reply, dict) and "output" in reply:
            reply = reply["output"]
        ss["messages_kitchen"].append({"role": "assistant", "content": str(reply)})
        st.rerun()

    # render history — newest first
    msgs = ss["messages_kitchen"]
    for disp_i, msg in enumerate(reversed(msgs)):
        orig_i = len(msgs) - 1 - disp_i
        role = msg["role"]
        avatar = "🙂" if role == "user" else "🤖"
        highlight = (ss.get("focus_msg_idx") == orig_i)
        style = "background-color:#fff5cc;border-radius:8px;padding:6px;" if highlight else ""
        with st.chat_message(role, avatar=avatar):
            st.markdown(f"<div style='{style}'>{msg['content']}</div>", unsafe_allow_html=True)

# RIGHT — Pantry + Cuisine
with right:
    # Pantry
    st.markdown("### 📦 Pantry")
    if st.button("Show Pantry (table)", use_container_width=True):
        ss["show_pantry_json"] = not ss.get("show_pantry_json", False)

    if ss.get("show_pantry_json"):
        ok, payload = _load_json_ok(PANTRY_PATH)
        if ok and isinstance(payload, dict):
            rows = _parse_pantry_rows(payload)
            df = pd.DataFrame(rows)
            df["Quantity"] = df.apply(lambda r: _pretty_quantity(r["item"], r["unit"], r["quantity"]), axis=1)
            st.dataframe(df[["S.No", "item", "Quantity"]], use_container_width=True, hide_index=True)
        elif ok:
            st.info("Pantry JSON exists but isn’t an object; showing raw content:")
            st.json(payload)
        else:
            st.error(payload)

    # Cuisine search (autofocus if you clicked a dish)
    st.markdown("### 🍽️ Cuisine")
    recipes = cuisine_load()
    cuisines = sorted({(r.get("cuisine") or "").title() for r in recipes if r.get("cuisine")})
    diets = ["Any", "veg", "eggtarian", "non-veg"]

    if ss.get("cuisine_autofocus"):
        dish = ss["cuisine_autofocus"]
        st.info(f"Showing: {dish}")
        picked = next((r for r in recipes if (r.get("name") or "").lower() == dish.lower()), None)
        if picked:
            st.markdown(_fmt_recipe_md(picked))
        else:
            st.warning("Recipe not found in DB.")
        default_query = dish
    else:
        default_query = ""

    with st.form("cuisine_search"):
        c1, c2, c3 = st.columns([1.2, 1, 1])
        q_name = c1.text_input("Recipe search", value=default_query, placeholder="e.g. palak paneer / fried rice")
        sel_cuisine = c2.selectbox("Cuisine", options=["Any"] + cuisines, index=0)
        sel_diet = c3.selectbox("Diet", options=diets, index=0)
        submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        q = (q_name or "").strip().lower()
        want_cuisine = (sel_cuisine if sel_cuisine != "Any" else "").lower()
        want_diet = (sel_diet if sel_diet != "Any" else "")

        matches = []
        for r in recipes:
            name = (r.get("name") or "")
            if q and q not in name.lower():
                continue
            if want_cuisine and (r.get("cuisine","").lower() != want_cuisine):
                continue
            if not cuisine_diet_ok(r.get("diet"), want_diet):
                continue
            total = int(r.get("prep_time_min",0)) + int(r.get("cook_time_min",0))
            matches.append({
                "name": name.title(),
                "cuisine": (r.get("cuisine") or "").title(),
                "diet": (r.get("diet") or ""),
                "total_time_min": total,
                "_raw": r,
            })

        if not matches:
            st.info("No recipes match those filters.")
        else:
            st.dataframe(
                [{k:v for k,v in m.items() if k != "_raw"} for m in matches],
                use_container_width=True, hide_index=True
            )
            names = [m["name"] for m in matches]
            pick = st.selectbox("View recipe", options=names, index=0)
            picked = next(m["_raw"] for m in matches if m["name"] == pick)
            st.markdown(_fmt_recipe_md(picked))
