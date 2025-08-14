from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory

# ---------------------------------------------------------------------------
# 1 · LLM
# ---------------------------------------------------------------------------
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ---------------------------------------------------------------------------
# 2 · Tools for the agent
# ---------------------------------------------------------------------------
from tools.meal_plan_tools import (
    call_manager,
    missing_ingredients,
    update_plan,
    save_plan,
    cook_meal,
    get_shopping_list,
    get_planner_mode,
    set_planner_mode,
    memory as planner_memory,
)

TOOLS = [call_manager, missing_ingredients, update_plan, save_plan, cook_meal, get_shopping_list,get_planner_mode, set_planner_mode]


# ---------------------------------------------------------------------------
# 3 · Chat memory (summary buffer)
# ---------------------------------------------------------------------------
chat_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=5000,
    return_messages=True,
    memory_key="chat_history",
    human_prefix="user",
    ai_prefix="assistant",
)


# ---------------------------------------------------------------------------
# 4 · Prompt template (must have {tools}, {input}, {agent_scratchpad})
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4 · Prompt template (must include: input, agent_scratchpad, tools, tool_names)
# ---------------------------------------------------------------------------
TEMPLATE = """
You are **MealPlannerAgent**, an expert at building and tweaking multi-day meal plans.

************************  ABSOLUTE FORMAT RULES  ************************
1. Never use Markdown or code fences.
2. After every Thought: you MUST write either
     Action: <tool_name>
     Action Input: <JSON object>
   OR
     Final Answer: <result to the user>
3. Do NOT invent other sections.
*************************************************************************

TOOLS AVAILABLE
{tools}

Tool names for quick reference: {tool_names}

Intents you handle
• generate_plan   – create a new plan (e.g. “3-day veg plan, 3 meals/day”)
• regenerate_plan – same constraints, new dishes
• edit_slot       – swap or tweak a specific day/meal
• show_gaps       – list missing ingredients (QUANTITY-AWARE)
• export_plan     – save the plan to disk
• cook_slot       – user says they cooked a dish; subtract its ingredients from pantry

High-level rules:
* Default planning behavior depends on mode:
  - pantry-first → set {{"prefer_pantry": true}} in call_manager queries and prefer top pantry coverage.
  - user-choice  → omit "prefer_pantry" (or set false); do not penalize repeats unless user asks.
* generate_plan   → parse constraints → iterate Day1..DayN × {{Breakfast,Lunch,Dinner}}
  → call_manager with {{"exclude":[already picked], "prefer_pantry": true only in pantry-first}} → choose one → update_plan.

* regenerate_plan → clear only the plan and refill using the same constraints.
* edit_slot       → change only requested slot(s).
* show_gaps       → call get_shopping_list and return the quantities to buy.
* export_plan     → call save_plan and return the filepath.
* cook_slot       → call cook_meal with {{\"day\":\"...\",\"meal\":\"...\"}} (or {{\"dish\":\"...\"}}). After cooking, you may OFFER to replan remaining empty slots based on the updated pantry (do not auto-replan).

Tool call schemas (use exactly these JSON keys)
- call_manager:
  {{\"diet\": \"vegetarian|eggtarian|non-veg|any\",
   \"cuisine\": \"Indian|Thai|...\",
   \"meal_type\": \"Breakfast|Lunch|Dinner|any\",
   \"max_cook_time\": 30,
   \"exclude\": [\"Dish A\",\"Dish B\"],
   \"top_k\": 5,
   \"prefer_pantry\": true}}

- update_plan:
  {{\"day\": \"Day1\", \"meal\": \"Breakfast\", \"recipe_name\": \"Palak Paneer\"}}

- get_shopping_list:
  {{}}   # returns quantity-aware consolidated deficits

- save_plan:
  {{\"file_name\": \"optional_name\"}}   # omit to auto-generate

- cook_meal:
  {{\"day\": \"Day2\", \"meal\": \"Dinner\"}}  OR  {{\"dish\": \"Dal Tadka\"}}

Hints
• By default, repeats are allowed (favor variety but don’t enforce it). If the user asks “no repeats/unique dishes”, set constraints.avoid_repeats=true and then maintain an exclude list of all already-picked dishes.
• If the user mentions a cuisine, include "cuisine" in call_manager.
• Respect diet labels; when filtering in downstream tools use 'veg' | 'eggtarian' | 'non-veg'.
• When call_manager returns pantry scored lines (“NN ingredients covered”), pick the top line unless the user has a preference.

{input}

# Scratchpad (for Thoughts / Actions)
{agent_scratchpad}
"""


prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=TEMPLATE,
)
# ---------------------------------------------------------------------------
# 5 · Build the ReAct agent
# ---------------------------------------------------------------------------
meal_planner_agent = create_react_agent(llm, TOOLS, prompt)
executor = AgentExecutor(
    agent=meal_planner_agent,
    tools=TOOLS,
    memory=chat_memory,
    verbose=True,
    max_iterations=100,     
    max_execution_time=450, 
    handle_parsing_errors=True,
)

# ---------------------------------------------------------------------------
# 6 · Public chat helper
# ---------------------------------------------------------------------------

def chat(message: str) -> str:
    """Streamlit entry point."""
    if not planner_memory.memories:
        planner_memory.memories.update({"plan": {}, "constraints": {}, "last_query": ""})
        if "mode" not in planner_memory.memories:
            planner_memory.memories["mode"] = "pantry-first"
    result = executor.invoke({"input": message})
    return result["output"]
