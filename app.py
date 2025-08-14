"""Run with:  streamlit run app.py

This version adds a fourth agent – MealPlanner – while keeping the original
three. The Planner tab is a chat window plus a sidebar dump of the current
planner_state and controls for planning/editing/cooking.
"""
from __future__ import annotations
import streamlit as st
from langchain.schema import HumanMessage

# ---------------------------------------------------------------------------
# 1 · Agents in the backend
# ---------------------------------------------------------------------------
from agents.pantry_agent import chat as pantry_chat
from agents.cuisine_agent import chat as cuisine_chat
from agents.manager_agent import chat as manager_chat, chat_memory
from agents.meal_planner_agent import chat as planner_chat

# Manager slot-memory (inventory cache, last dishes, etc.)
from tools.manager_tools import memory as slot_memory

# Planner in-memory state (plan + constraints)
from tools.meal_plan_tools import memory as planner_memory

# ---------------------------------------------------------------------------
# 2 · Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Kitchen Chat", page_icon="👩‍🍳", layout="centered")
st.title("👩‍🍳 Kitchen Chat")

# ---------------------------------------------------------------------------
# 3 · Agent selector & helper maps
# ---------------------------------------------------------------------------
PANTRY_LABEL   = "PantryAgent 🥫"
CUISINE_LABEL  = "CuisineAgent 🍽️"
MANAGER_LABEL  = "ManagerAgent 🧑‍🍳"
PLANNER_LABEL  = "MealPlanner 📅"

choice = st.radio("Talk to:", (PANTRY_LABEL, CUISINE_LABEL, MANAGER_LABEL, PLANNER_LABEL), horizontal=True)

AGENT_MAP = {
    PANTRY_LABEL:  (pantry_chat,  "Type a pantry request – e.g. 'Add 2 onions'…"),
    CUISINE_LABEL: (cuisine_chat, "Ask a recipe question – e.g. 'Pad Thai recipe'…"),
    MANAGER_LABEL: (manager_chat, "Ask anything about meal prep – e.g. 'What Thai dishes use chicken?'…"),
    PLANNER_LABEL: (planner_chat, "Plan meals – e.g. 'Vegetarian meals for 3 days, 3/day'…"),
}

agent_func, placeholder = AGENT_MAP[choice]

def _normalize_planner_prompt(user_text: str) -> str:
    """
    Guardrails for MealPlanner:
    - default to 1 day if days not mentioned
    - default to 3 meals/day if meals/day not mentioned
    - do NOT add 'no repeats' unless the user asks
    """
    txt = user_text.strip().lower()
    import re

    # detect explicit days
    days = None
    m = re.search(r"(\d+)\s*day", txt)
    if m:
        days = int(m.group(1))

    # detect meals per day
    meals_per_day = None
    m2 = re.search(r"(\d+)\s*meal[s]?\s*/?\s*day", txt)
    if m2:
        meals_per_day = int(m2.group(1))

    hints = []
    if days is None:
        hints.append(" [assume 1 day]")
    if meals_per_day is None:
        hints.append(" [assume 3 meals/day]")

    return user_text + ("".join(hints) if hints else "")

# slug for session_state key (one chat history per agent)
SLUG_MAP = {PANTRY_LABEL: "pantry", CUISINE_LABEL: "cuisine", MANAGER_LABEL: "manager", PLANNER_LABEL: "planner"}
hist_key = f"messages_{SLUG_MAP[choice]}"
if hist_key not in st.session_state:
    st.session_state[hist_key] = []

# ---------------------------------------------------------------------------
# 4 · Render chat history (center column)
# ---------------------------------------------------------------------------
for msg in st.session_state[hist_key]:
    role, avatar = ("user", "🙂") if msg["role"] == "user" else ("assistant", "🤖")
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 5 · Chat input & agent invocation
# ---------------------------------------------------------------------------
prompt = st.chat_input(placeholder)
if prompt:
    user_msg = _normalize_planner_prompt(prompt) if choice == PLANNER_LABEL else prompt
    st.session_state[hist_key].append({"role": "user", "content": user_msg})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_msg)

    try:
        if choice == PLANNER_LABEL:
            st.session_state["planner_busy"] = True
            with st.spinner("Planning… please don’t switch tabs until it finishes."):
                reply = agent_func(user_msg)
            st.session_state["planner_busy"] = False
        else:
            reply = agent_func(user_msg)
    except Exception as err:
        st.session_state["planner_busy"] = False
        reply = f"🚨 Error: {err}"

    st.session_state[hist_key].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)

# ---------------------------------------------------------------------------
# 6 · Sidebar – reset & memory views
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Utilities")
    if st.button("🔄 Reset chat & memories"):
        # clear per-agent chat logs
        for key in list(st.session_state.keys()):
            if key.startswith("messages_"):
                del st.session_state[key]
        # wipe Manager/Planner memories
        slot_memory.memories.clear()
        planner_memory.memories.clear()
        st.rerun()

    # ------------------------------- Manager diagnostics --------------------
    if choice == MANAGER_LABEL:
        st.markdown("### 🧑‍🍳 Manager slots")
        st.json(slot_memory.memories)

        st.markdown("### 🕑 Conversation buffer (last messages)")
        hist = chat_memory.load_memory_variables({}).get("chat_history", [])
        for m in hist[-10:]:
            who  = "user" if isinstance(m, HumanMessage) else "assistant"
            text = m.content.replace("\n", " ")[:120] + ("…" if len(m.content) > 120 else "")
            st.markdown(f"*{who}:* {text}")

    # --------------------------------- Planner diagnostics ------------------
    if choice == PLANNER_LABEL:
        st.markdown("### 📅 Current plan (planner_state)")
        st.json(planner_memory.memories)
        if st.session_state.get("planner_busy"):
            st.warning("Planner is running… please don’t switch tabs until it finishes.")

        # ---- Mode toggle
        st.markdown("### ⚙️ Planning mode")
        current_mode = planner_memory.memories.get("mode", "pantry-first")
        new_mode = st.radio(
            "Select how to plan new meals:",
            options=["pantry-first", "user-choice"],
            index=0 if current_mode == "pantry-first" else 1,
            horizontal=True,
            help="Pantry-first uses what you already have. User-choice prioritizes your choices and can generate a shopping list.",
        )
        if new_mode != current_mode:
            planner_memory.memories["mode"] = new_mode
            st.success(f"Mode set to **{new_mode}**")
        st.caption(f"Current mode: **{planner_memory.memories.get('mode','pantry-first')}**")

        # ---- Planning calc log (diagnostics)
        with st.expander("🧮 Planning calc log", expanded=False):
            calc = planner_memory.memories.get("calc_log", [])
            if not calc:
                st.caption("No calculation entries yet.")
            else:
                for i, row in enumerate(calc, 1):
                    st.markdown(f"**#{i}** — {row.get('slot','(slot)')} → *{row.get('dish','')}*")
                    if row.get("reason"):
                        st.caption(f"Reason: {row['reason']}")
                    if row.get("virtual_deducted"):
                        st.write("Simulated consume: " + ", ".join(row["virtual_deducted"]))
                    if row.get("still_missing"):
                        st.write("Missing for this dish: " + ", ".join(row["still_missing"]))
                    st.divider()

        # ---- Cook history
        with st.expander("✅ Cook history", expanded=False):
            cook_log = planner_memory.memories.get("planner_log", [])
            if not cook_log:
                st.write("No dishes marked as cooked yet.")
            else:
                for evt in cook_log:
                    st.write(f"- {evt.get('event','')} · {evt.get('dish','')}")
                    if evt.get("deducted"):
                        st.caption("Consumed: " + ", ".join(evt["deducted"]))
                    if evt.get("missing"):
                        st.caption("Still needed: " + ", ".join(evt["missing"]))

        # ---- Shopping list view (quantity-aware)
        with st.expander("🛒 Shopping list (latest)", expanded=False):
            latest = planner_memory.memories.get("shopping_list")

            def _render_shopping(sl):
                # sl can be a dict {"g":[...], "ml":[...], "count":[...]} OR a string
                if isinstance(sl, dict):
                    any_lines = False
                    for unit in ("g", "ml", "count"):
                        lines = sl.get(unit, [])
                        if lines:
                            any_lines = True
                            st.markdown(f"**{unit}**")
                            for line in lines:
                                st.write("• " + line)
                    # render any other unexpected bucket
                    for unit, lines in sl.items():
                        if unit in ("g", "ml", "count"):
                            continue
                        if lines:
                            any_lines = True
                            st.markdown(f"**{unit}**")
                            for line in lines:
                                st.write("• " + line)
                    if not any_lines:
                        st.caption("No items needed.")
                else:
                    st.markdown(sl if sl else "_No items needed._")

            if latest:
                _render_shopping(latest)
            else:
                st.caption("No shopping list cached yet.")

            if st.button("Recompute shopping list", key="recompute_shopping_list"):
                with st.spinner("Recomputing…"):
                    out = planner_chat("get shopping list")
                st.markdown(out)
                latest2 = planner_memory.memories.get("shopping_list")
                if latest2:
                    st.divider()
                    st.caption("Updated list:")
                    _render_shopping(latest2)

        # ---- Slot actions (unified: cook / suggest / apply change)
        st.subheader("Slot actions")

        _plan = planner_memory.memories.get("plan", {})
        # Prefer real days from the plan; otherwise a friendly default
        all_days = sorted(_plan.keys(), key=lambda d: (len(d), d)) or ["Day1", "Day2", "Day3"]
        all_meals = ["Breakfast", "Lunch", "Dinner"]

        with st.form("slot_actions"):
            col1, col2 = st.columns(2)
            with col1:
                day_sel  = st.selectbox("Day", all_days, key="slot_day")
            with col2:
                meal_sel = st.selectbox("Meal", all_meals, key="slot_meal")

            new_dish = st.text_input("New dish (optional for Apply change):", key="slot_new_dish")

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                do_cook = st.form_submit_button("✅ Mark cooked")
            with c2:
                do_suggest = st.form_submit_button("🔎 Suggest dishes")
            with c3:
                do_apply = st.form_submit_button("✅ Apply change")

        if do_cook:
            out = planner_chat(f"mark {meal_sel} on {day_sel} as cooked")
            st.success(out)

        if do_suggest:
            ask = f"suggest 5 options for {meal_sel} on {day_sel} based on current mode"
            tips = planner_chat(ask)
            st.info("Suggestions:\n" + tips)

        if do_apply:
            if new_dish.strip():
                out = planner_chat(f'update {meal_sel} on {day_sel} to "{new_dish.strip()}"')
                st.success(out)
            else:
                st.error("Please enter a dish name first for Apply change.")

        # One-click gaps (text reply) – handy from the sidebar
        if st.button("🧾 Show gaps (text)", key="show_gaps_text"):
            with st.spinner("Computing shopping list…"):
                out = planner_chat("show_gaps")
            st.markdown(out)
