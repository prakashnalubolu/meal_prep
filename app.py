import streamlit as st
from agents.pantry_agent  import chat as pantry_chat
from agents.cuisine_agent import chat as cuisine_chat

st.set_page_config(page_title="Kitchen Chat", page_icon="👩‍🍳", layout="centered")
st.title("👩‍🍳 Kitchen Chat")

agent_choice = st.radio(
    "Which assistant would you like to talk to?",
    ("PantryAgent 🥫", "CuisineAgent 🍽️"),
    horizontal=True,
)
agent = pantry_chat if agent_choice.startswith("Pantry") else cuisine_chat
placeholder = ("Type a pantry request…" if agent is pantry_chat
               else "Ask a recipe question…")

# keep separate histories per agent
key = "messages_pantry" if agent is pantry_chat else "messages_cuisine"
if key not in st.session_state:
    st.session_state[key] = []

for m in st.session_state[key]:
    role   = "user" if m["role"] == "user" else "assistant"
    avatar = "🙂" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"])

prompt = st.chat_input(placeholder)
if prompt:
    st.session_state[key].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    try:
        response = agent(prompt)
    except Exception as e:
        response = f"🚨 Error: {e}"

    st.session_state[key].append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
