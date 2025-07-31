# agents/meal_planner_agent.py
from __future__ import annotations
import os, datetime as dt, re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ── meal‑plan tools -------------------------------------------------------
from tools.meal_plan_tools import (
    plan_meals,
    shopping_list_for_plan,
    add_meal,
    delete_meal,
    list_meal_plan,
)

load_dotenv()

TOOLS      = [plan_meals, shopping_list_for_plan, add_meal, delete_meal, list_meal_plan]
TOOL_NAMES = ", ".join(t.name for t in TOOLS)

# ── base LLM --------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── prompt ----------------------------------------------------------------
TEMPLATE = """
You are MealPlanner, an assistant that organizes meal plans by calling these tools:
{tools}

When the user asks for a plan, extract the constraints: start date (natural language like today, tomorrow, next Monday; default to today), number of days (default 7), meals per day (if unspecified assume breakfast, lunch, dinner), dietary preferences (veg / non-veg / mixed), specific inclusions (e.g., include rice at lunch and dinner), and nutritional goals (e.g., ensure a protein source in every meal). Normalize any date to ISO format (YYYY-MM-DD) before using it.

Begin a new plan by calling the plan_meals tool with keys: start_date, days, meals_per_day, and optionally diet. After a plan exists, you may call list_meal_plan (Action Input: {{}}) to view it, add_meal or delete_meal to adjust individual slots, or shopping_list_for_plan to get shortages.

Follow the ReAct pattern exactly:

Thought: your reasoning.  
Action: the tool name (one of [{tool_names}]).  
Action Input: a single JSON object, never wrapped in quotes. Example for plan_meals:
{{ "start_date": "tomorrow", "days": 5, "meals_per_day": 3, "diet": "mixed" }}  
Observation: the tool's reply.  

You may loop Thought/Action/Observation multiple times.  
Finish with:  
Thought: I now know the final answer.  
Final Answer: a concise user-facing summary.

Do not invent extra keys outside the JSON block and do not wrap the JSON in backticks.

────────────────────────────── BEGIN
Question: {input}
{agent_scratchpad}
"""

PROMPT = PromptTemplate(
    template=TEMPLATE,
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join(f"- {t.name}" for t in TOOLS),
        "tool_names": TOOL_NAMES,
    },
)


# ── build ReAct agent -----------------------------------------------------
react_agent = create_react_agent(
    llm   = llm,
    tools = TOOLS,
    prompt= PROMPT,
)

meal_planner = AgentExecutor(
    agent = react_agent,
    tools = TOOLS,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
)

# convenience wrapper ------------------------------------------------------
def chat(message: str) -> str:
    return meal_planner.invoke({"input": message})["output"]
