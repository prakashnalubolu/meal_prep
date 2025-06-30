# agents/cuisine_agent.py
from __future__ import annotations
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.structured_chat.base import StructuredChatAgent

from tools.cuisine_tools import (
    get_recipe,
    list_recipes,
    add_recipe,
    delete_recipe,
)

load_dotenv()

# ── 1. LLM (Gemini) ────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   
    temperature=0.0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── 2. Tools list ──────────────────────────────────────────────────────────
TOOLS = [get_recipe, list_recipes, add_recipe, delete_recipe]
TOOL_NAMES = ", ".join(t.name for t in TOOLS)
# ── 3. Build the prompt that embeds every tool’s JSON schema ──────────────
#     (no manual prompt writing!)
base_prompt = StructuredChatAgent.create_prompt(TOOLS)

prompt = base_prompt.partial(                           
    tools      = "\n".join(f"- {t.name}" for t in TOOLS),
    tool_names = TOOL_NAMES,
)      

# ── 4. Create the structured-chat agent (Gemini function-calling) ─────────
structured_agent = create_structured_chat_agent(
    llm    = llm,
    tools  = TOOLS,
    prompt = prompt,
)

# ── 5. Wrap in an executor ────────────────────────────────────────────────
cuisine_agent = AgentExecutor(
    agent                 = structured_agent,
    tools                 = TOOLS,
    max_iterations        = 5,
    handle_parsing_errors = True,
    verbose               = True,  
)

# ── 6. Convenience wrapper for Streamlit or tests ─────────────────────────
def chat(message: str) -> str:
    """Send text through CuisineAgent and return its reply."""
    return cuisine_agent.invoke({"input": message})["output"]
