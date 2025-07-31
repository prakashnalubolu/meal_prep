# app.py – Streamlit UI for Pantry / Cuisine / Manager / Meal-Planner agents
import calendar
import datetime as dt
import streamlit as st
from langchain.schema import HumanMessage

# --- Agents ---------------------------------------------------------------
from agents.pantry_agent import chat as pantry_chat
from agents.cuisine_agent import chat as cuisine_chat
from agents.manager_agent import chat as manager_chat, chat_memory
from agents.meal_planner_agent import chat as planner_chat

# --- Slot memory used by Manager (inventory, current_dish, etc.) ----------
from tools.manager_tools import memory as slot_memory

# --- Meal-plan persistence helpers (pure-Python; no LLM) ------------------
from tools.meal_plan_tools import load_plan_dict, add_meal, delete_meal

# -------------------------------------------------------------------------
# 1. Page config
# -------------------------------------------------------------------------
st.set_page_config(page_title="Kitchen Chat", page_icon="👩‍🍳", layout="centered")
st.title("👩‍🍳 Kitchen Chat")

# -------------------------------------------------------------------------
# 2. Agent selector
# -------------------------------------------------------------------------
PANTRY_LABEL   = "PantryAgent 🥫"
CUISINE_LABEL  = "CuisineAgent 🍽️"
MANAGER_LABEL  = "ManagerAgent 🧑‍🍳"
PLANNER_LABEL  = "MealPlanner 📅"  # use this one string everywhere

choice = st.radio(
    "Talk to:",
    (PANTRY_LABEL, CUISINE_LABEL, MANAGER_LABEL, PLANNER_LABEL),
    horizontal=True,
)

AGENT_MAP = {
    PANTRY_LABEL:  (pantry_chat,  "Type a pantry request (e.g., 'Add 2 onions')…"),
    CUISINE_LABEL: (cuisine_chat, "Ask a recipe question (e.g., 'Pad Thai recipe')…"),
    MANAGER_LABEL: (manager_chat, "Ask anything about meal prep (e.g., 'What Thai dishes use chicken?')…"),
    PLANNER_LABEL: (planner_chat, "Plan meals (e.g., 'Plan meals 2025-07-10 for 5 days veg')…"),
}

agent_func, placeholder = AGENT_MAP[choice]

# explicit slugs for session_state keys
SLUG_MAP = {
    PANTRY_LABEL:  "pantry",
    CUISINE_LABEL: "cuisine",
    MANAGER_LABEL: "manager",
    PLANNER_LABEL: "planner",
}
hist_key = f"messages_{SLUG_MAP[choice]}"

if hist_key not in st.session_state:
    st.session_state[hist_key] = []

# -------------------------------------------------------------------------
# 3. Render chat history (main pane)
# -------------------------------------------------------------------------
for msg in st.session_state[hist_key]:
    role, avatar = ("user", "🙂") if msg["role"] == "user" else ("assistant", "🤖")
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# -------------------------------------------------------------------------
# 4. Chat input + agent call
# -------------------------------------------------------------------------
# We *keep* chat for MealPlanner so you can test planner prompts.
prompt = st.chat_input(placeholder)
if prompt:
    st.session_state[hist_key].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    try:
        reply = agent_func(prompt)
    except Exception as e:
        reply = f"🚨 Error: {e}"

    st.session_state[hist_key].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)

# -------------------------------------------------------------------------
# 5. Sidebar: reset + per-agent memory views
# -------------------------------------------------------------------------
with st.sidebar:
    if st.button("🔄 Reset chat & slots"):
        # clear Streamlit chat histories
        for k in list(st.session_state.keys()):
            if k.startswith("messages_"):
                del st.session_state[k]
        # clear Manager slot memory
        slot_memory.memories.clear()
        # forget day selection
        st.session_state.pop("selected_day", None)
        st.rerun()

    # ----- Manager memory --------------------------------------------------
    if choice == MANAGER_LABEL:
        st.markdown("### Manager slots")
        st.json(slot_memory.memories)

        st.markdown("### Conversation buffer (last messages)")
        hist = chat_memory.load_memory_variables({})["chat_history"]
        for m in hist[-10:]:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            st.markdown(f"*{role}: {m.content[:80]}…*")

    # ----- MealPlanner calendar --------------------------------------------
    # ----- MealPlanner calendar --------------------------------------------
    # ----- MealPlanner calendar --------------------------------------------
    if choice == PLANNER_LABEL:
        st.markdown("### Meal‑plan calendar")

        # 0 · month in session_state
        if "cal_date" not in st.session_state:
            st.session_state["cal_date"] = dt.date.today().replace(day=1)
        cal_date: dt.date = st.session_state["cal_date"]

        # 1 · header with arrows
        col_prev, col_title, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀", key="cal-prev"):
                st.session_state["cal_date"] = (cal_date - dt.timedelta(days=1)).replace(day=1)
                st.rerun()
        col_title.markdown(f"**{cal_date.strftime('%B %Y')}**")
        with col_next:
            if st.button("▶", key="cal-next"):
                y, m = cal_date.year, cal_date.month
                st.session_state["cal_date"] = dt.date(y + m // 12, m % 12 + 1, 1)
                st.rerun()

        # weekday header  ------------------------------------------------------
        weekday_cols = st.columns(7)
        for wd, col in enumerate(weekday_cols):
            col.markdown(["**Mon**","**Tue**","**Wed**","**Thu**","**Fri**","**Sat**","**Sun**"][wd],unsafe_allow_html=True)
        
        #colour palette  (light → dark orange)
        ORANGES = ["#FFE8CC","#FFD6A6","#FFC280", "#FFAE59",  "#FF9A33",  "#FF851A", "#FFE8CC"]
        # ---- global button colour (one shot) ------------------------------------
        st.markdown("""<style> /* every Streamlit button inside the calendar grid */
                    div[data-testid="column"] button {
                    background-color: #FFB84D !important;   /* orange */
                    color: #000 !important;}</style>""",unsafe_allow_html=True,)

        
        # month grid  ----------------------------------------------------------    
        plan_dict = load_plan_dict()
        y, m = cal_date.year, cal_date.month
        for wk_idx, week in enumerate(calendar.monthcalendar(y, m)):
            cols = st.columns(7)
            for wd, (d, col) in enumerate(zip(range(7), cols)):   # wd = 0..6
                day_num = week[wd]
                if day_num == 0:
                    col.write(" ")
                    continue

                date_str = f"{y}-{m:02d}-{day_num:02d}"
                label    = f"{day_num}•" if date_str in plan_dict else str(day_num)
                key      = f"day-{y}-{m}-{day_num}"

                if col.button(label, key=key):
                    st.session_state["selected_day"] = date_str
                    st.rerun()

        # selected‑day details
        sel_day = st.session_state.get("selected_day")
        if sel_day:
            st.write(f"#### {sel_day}")
            meals = plan_dict.get(sel_day, [])
            if not meals:
                st.caption("No meals planned yet.")
            else:
                for rec in meals:
                    col1, col2 = st.columns([5,1])
                    col1.markdown(f"- **{rec['meal'].title()}**: {rec['dish'].title()}")
                    if col2.button("❌", key=f"del-{sel_day}-{rec['meal']}"):
                        delete_meal(sel_day, rec["meal"])  # updates JSON
                        st.experimental_rerun()

            st.divider()
            st.markdown("*Add / update meal*")
            new_meal = st.selectbox("Meal", ["breakfast", "lunch", "dinner"], key="planner-meal")
            dish = st.text_input("Dish", key="planner-dish")
            if st.button("➕ Add / replace", key="planner-add"):
                if dish.strip():
                    add_meal(sel_day, new_meal, dish.strip())
                    st.experimental_rerun()

            st.divider()
            if st.button("🛒 Shopping list for week", key="planner-shop"):
                # build 7-day window containing sel_day
                sel_date = dt.date.fromisoformat(sel_day)
                wk_start = sel_date - dt.timedelta(days=sel_date.weekday())  # Monday
                wk_end   = wk_start + dt.timedelta(days=6)
                shop = planner_chat(f"shopping list for {wk_start.isoformat()} to {wk_end.isoformat()}")
                st.markdown(shop)
