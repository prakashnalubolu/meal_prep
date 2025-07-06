from __future__ import annotations
import os
from dotenv import load_dotenv
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from tools.manager_tools import call_pantry, call_cuisine, missing_ingredients, memory

load_dotenv()

# ── 1  LLM ──────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── 2  Tools exposed to Manager ────────────────────────────────────────────
TOOLS      = [call_pantry, call_cuisine, missing_ingredients]
TOOL_NAMES = ", ".join(t.name for t in TOOLS)


MAX_INV_AGE_SEC = 60        # refresh if older than 1 minute (tweak)

def ensure_fresh_inventory():
    """Refresh inventory cache if missing or too old."""
    ts = memory.memories.get("inv_timestamp")
    if ts is None or (time.time() - ts) > MAX_INV_AGE_SEC:
        call_pantry("list pantry")      # updates memory in place

# ── 3  Prompt with routing rules ───────────────────────────────────────────
template = """
You are **MealPrepManager**, the orchestrator for a kitchen assistant.

You control two specialist agents exposed as tools:
{tools}

Routing rules:
1.  **Pantry-only query**  
    If the user’s request is purely about pantry inventory  
    (add / remove / update / list) →  
    **Action → call_pantry** with their exact message.

2.  **Cuisine-only query**  
    If the request is purely about recipes, cuisines, cooking instructions, etc. →  
    **Action → call_cuisine** with their exact message.

3.  **“What can I cook with what's in my pantry?” (cross-domain)**  
    3-a. **Action → call_pantry("list pantry")** → Observation (inventory lines)  
    3-b. Extract ingredient names (everything before the first “(” on each line).  
    3-c. **Action → call_cuisine** with a message like:  
        “Given these items <list>, please run `find_recipes_by_items`  
        with `cuisine=<if specified>` and `k=5`.”

4.  **Ingredient-gap query** — user asks   
    “what else do I need”, “what ingredients am I missing”,  
    “check pantry and update”, etc.  
    **Action → missing_ingredients("<dish name>")**  
    Append its natural-language sentence to your Final Answer.  
    *Never* ask the user to list their items again.

5. When you call missing_ingredients, pass *only* the dish name string, e.g.
   Action: missing_ingredients
   Action Input: "egg fried rice"
   Never wrap it like a python call.
6. After completing all necessary calls, end with exactly one Final Answer: <your concise reply>

Thought: …
Action: <{tool_names}>
Action Input: …
Observation: …
… (repeat) …
Thought: I now know the final answer
Final Answer: …

Begin!

Question: {input}
{agent_scratchpad}
"""

PROMPT = PromptTemplate(
    template       = template,
    input_variables= ["input", "agent_scratchpad", "tools", "tool_names"],
)

# ── 4  Build ReAct manager agent ───────────────────────────────────────────
react_agent = create_react_agent(
    llm    = llm,
    tools  = TOOLS,
    prompt = PROMPT.partial(
        tools      = "\n".join(f"- {t.name}" for t in TOOLS),
        tool_names = TOOL_NAMES,
    ),
)

manager_agent = AgentExecutor(
    agent                 = react_agent,
    tools                 = TOOLS,
    max_iterations        = 6,
    memory = memory,
    handle_parsing_errors = True,
    verbose               = True,
)

# ── 5  Convenience wrapper --------------------------------------------------
def chat(message: str) -> str:
    """Send a user message through the manager and return its reply."""
    ensure_fresh_inventory()              # ← refresh cache if stale
    return manager_agent.invoke({"input": message})["output"]