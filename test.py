from langchain_core.prompts import PromptTemplate
from agents.meal_planner_agent import TEMPLATE  # or paste the string here

PromptTemplate(
    input_variables=["input","agent_scratchpad","tools","tool_names"],
    template=TEMPLATE,
).format(input="hi", agent_scratchpad="", tools="[...]", tool_names="a,b")
