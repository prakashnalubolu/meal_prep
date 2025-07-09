# app.py  –  Streamlit UI for Pantry / Cuisine / Manager agents
import streamlit as st

from agents.pantry_agent  import chat as pantry_chat
from agents.cuisine_agent import chat as cuisine_chat
from agents.manager_agent import chat as manager_chat, chat_memory
from tools.manager_tools   import memory  
from langchain.schema import HumanMessage
    

# ── 1  Page setup ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Kitchen Chat", page_icon="👩‍🍳", layout="centered")
st.title("👩‍🍳 Kitchen Chat")

# ── 2  Agent selector ───────────────────────────────────────────────────────
choice = st.radio(
    "Which assistant would you like to talk to?",
    ("PantryAgent 🥫", "CuisineAgent 🍽️", "ManagerAgent 🧑‍🍳"),
    horizontal=True,
)

AGENT_MAP = {
    "PantryAgent 🥫": (
        pantry_chat,
        "Type a pantry request (e.g., 'Add 2 onions')…"
    ),
    "CuisineAgent 🍽️": (
        cuisine_chat,
        "Ask a recipe question (e.g., 'Pad Thai recipe')…"
    ),
    "ManagerAgent 🧑‍🍳": (
        manager_chat,
        "Ask anything about meal prep (e.g., 'What Thai dishes can I cook with chicken?')…"
    ),
}

agent_func, placeholder = AGENT_MAP[choice]

# ── 3  Per-agent chat history ──────────────────────────────────────────────
hist_key = f"messages_{choice.split()[0].lower()}"   # messages_pantry / _cuisine / _manager
if hist_key not in st.session_state:
    st.session_state[hist_key] = []

# ── 4  Render chat history ─────────────────────────────────────────────────
for msg in st.session_state[hist_key]:
    role, avatar = ("user", "🙂") if msg["role"] == "user" else ("assistant", "🤖")
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ── 5  Input box and agent invocation ──────────────────────────────────────
prompt = st.chat_input(placeholder)
if prompt:
    # show user message
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

# ── 6  Sidebar: reset + memory view (only for Manager) ─────────────────────
with st.sidebar:
    if st.button("🔄 Refresh chat"):
        # clear Streamlit histories
        for k in list(st.session_state.keys()):
            if k.startswith("messages_"):
                del st.session_state[k]
        # clear Manager’s short-term slots
        memory.memories.clear()
        st.rerun() 

    if choice == "ManagerAgent 🧑‍🍳":
        st.markdown("### Manager slots")
        st.json(memory.memories)
        st.markdown("### Conversation buffer (last messages)")
        hist = chat_memory.load_memory_variables({})["chat_history"]

        # A very compact, one-line-per-turn view
        for m in hist[-10:]:               
            role   = "user"      if isinstance(m, HumanMessage) else "assistant"
            avatar = "🧑‍💬"      if role == "user" else "🤖"
            st.markdown(f"*{role} {m.content[:80]}…*")   