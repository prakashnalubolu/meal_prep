# 🍳 Kitchbot 
## An Agentic Chatbot built to help you manage your pantry, Solve cuisine related queries, helps in meal prep.

### A fast, pragmatic meal-planning assistant built with Streamlit + LangChain (single ReAct agent).
### It manages your pantry, finds recipes you can cook right now, plans multi-day menus, builds a quantity-aware shopping list, and marks meals cooked (with real deductions).

# Table of contents:
## Demo
## What can Kitchbot do?
## Why Kitchbot?
## Architecture
## Key features
## Best-fit use cases
## Getting started
## Repository layout
## Tooling (APIs the agent calls)
## Planner logic
## Memory

# Demo:
## Video URL: #TBU
## Screenshots:
Read the chats from Bottom to Up, Because everytime you send a message its pushed down so agent's response is on top. I wanted my UI to be little different :P
<img width="1892" height="1103" alt="image" src="https://github.com/user-attachments/assets/4a0694bd-b23f-4c2b-b908-fe0d86c8b373" />
<img width="1823" height="937" alt="image" src="https://github.com/user-attachments/assets/c4598c7a-5ebd-4721-bcb6-8e1b11d547fb" />
<img width="1865" height="1012" alt="image" src="https://github.com/user-attachments/assets/6e2b66a8-987f-485e-a026-403f5ff9c18f" />
You can also choose to generate meal plan with just a click:
<img width="1389" height="925" alt="image" src="https://github.com/user-attachments/assets/bc6f88bd-7902-4b18-b959-fd287dc2a5ea" />  
And also edit the plan  
<img width="718" height="828" alt="image" src="https://github.com/user-attachments/assets/4b2cb72b-6a77-4984-bcef-45d377b84a19" />  
And much more......

# What can Kitchbot do?
## Pantry:
Add / remove / update pantry items. Items are auto-merged, singularized, and lower-cased.

## Cuisine & recipe search
Browse the recipes in the DB, or fetch complete steps in preparing a dish. (Currently I have Chatgpt generated Recipes)

## Questions related to both pantry and cuisine
Finds dishes 100% covered by your current pantry (with simple smart swaps optionally suggested). Later helps in pantry aware meal planning.    
Lets you know whether you can cook a dish with the ingredients available in your pantry.

## Meal planner
Plans multi-day meals into Day × {Breakfast, Lunch, Dinner}. 
### Two modes:
  Pantry-first (strict): only places dishes fully covered; simulates deductions between slots.  
  Freeform: plans first; shopping list covers gaps.

## Gaps & shopping list
Plan-wide, quantity-aware missing items with unified units.

## Cook & deduct
Mark a dish cooked to deduct ingredients from the pantry.

## Export
Save the current plan to disk.

Everything is accessible via a single chat box and a clean left/right layout: Planner (left), Chat (center), Pantry/Cuisine (right).

# Why Kitchbot?
Every week I was doing the same dance:
Open the fridge, guess what’s still usable.
Google for recipes, only to realize I’m missing one random ingredient.
Buy duplicates of things I actually had.
Start a plan in a note, then change half of it while cooking.

I didn’t want “another recipe app.” I wanted a kitchen sidekick that could answer two simple questions:
What can I cook right now with what I already have?
Can you plan a few days so I stop thinking about food every six hours?
That led to three guiding ideas:
Pantry-first, not aspiration-first. Most planners assume you’ll shop; I wanted to cook down the pantry first and shop only for gaps.
Actionable, not inspirational. If a dish can’t be cooked today, it shouldn’t be on today’s plan.
One assistant, not a committee. I started with a multi-agent setup, but tool-hopping was slow and flaky. A single agent with clear tools was faster, easier to reason about, and stayed in sync with the UI.

So Kitchbot is built to be practical:
Normalize units automatically (count/g/ml), merge stock, and keep names clean.
Plan Day × {Breakfast, Lunch, Dinner} using what you have (strict mode), or plan freely and generate a gap-only shopping list (freeform).
Deduct ingredients when you mark a dish “cooked,” so the next plan is more accurate without you doing any bookkeeping.
The outcome is a simple app that actually reduces decision fatigue and food waste.

# Architecture
## Big picture
### UI: Streamlit (app.py) with three panels
Planner (left): Title + reset, start date, generator, editable board  
Chat (center): natural language interface to everything  
Pantry & Cuisine (right): table view, search, and recipe preview  

### Agent: a single ReAct agent
Model: Gemini 2.0 Flash (via langchain_google_genai)  
One prompt routes all tasks (pantry, cuisine, planning, shopping list, cook/deduct)  
Tools are directly imported (fail fast) and asserted at startup so planning never “silently disappears”  

### Tools: plain Python functions in tools/ with tight schemas
pantry_tools.py – list/add/remove/update with unit normalization  
cuisine_tools.py – list recipes, get full steps, “what can I cook with my pantry?”  
manager_tools.py – missing ingredients + smart substitutions  
meal_plan_tools.py – constraints, auto_plan, update slots, shopping list, cook/deduct, export  

### State & memory:
Conversation memory: ConversationSummaryBufferMemory to keep the chat short but contextual  
Planner memory: a simple in-process dict (plan, constraints, logs, shopping list cache) that both the tools and UI read/write—so the planner board always reflects what the agent actually did  

### Data:
data/pantry.json – your live pantry store  
data/alt_units.json – count↔g/ml hints and human labels (e.g., garlic “cloves”, spinach “bunches”)   
data/recipes.json – compact recipe dataset used for search and steps  

## Why single agent (after trying multi-agent)
Less latency: fewer tool hops and no cross-agent orchestration.  
Predictability: one prompt controls routing; fewer “who calls what next?” surprises.  
Observability: direct imports + startup asserts tell you immediately if a tool is missing (e.g., auto_plan).  
Consistency: a single shared planner memory prevents state drift between agents.  
Simplicity: easier to test, easier to debug, easier to extend.  

## Planner algorithm (why it feels practical)
### Pantry-first (strict) mode:
Build candidate dishes filtered by cuisine/diet/time.  
Take a shadow snapshot of the pantry (canonical units).  
Place each dish that’s 100% coverable at least once (unseen-first), avoiding consecutive repeats if requested.  
If none unseen fit, place any dish that is coverable right now.  
Simulate deductions between slots (so lunch “uses up” what dinner can’t rely on).  
Stop the moment a slot can’t be strictly filled—but keep everything already written.  

### Freeform mode:
Place eligible dishes without coverage checks; the shopping list then fills the gaps.  
Design choices at a glance  
ReAct agent with strict tool schemas → deterministic, auditable traces  
Direct tool imports + startup asserts → fail fast when something’s off  
Unit canonicalization + alt unit hints → less friction entering pantry data  
No-consecutive-repeats by default when variety is requested; global-unique is a planned enhancement  

Reset button wipes plan + chat and restores default constraints (Pantry-first strict)  

# Key features
✅ Pantry-first (strict) planning that respects real stock  
✅ Shadow pantry simulation: deductions between slots while planning  
✅ Unseen-first bias (try each fully-coverable dish at least once)  
✅ No-consecutive repeats when allow_repeats = False  
✅ Unit normalization + alt unit hints (count ↔ g/ml)  
✅ Compact, readable summaries in chat and synced Planner board  
✅ Reset button nukes plan/chat cleanly and restores default constraints  

# Best-fit use cases
You actually stock a pantry and want “cook now” options without shopping.  
You want a 2–7 day plan that uses what you already have.  
You batch cook and deduct inventory as you go.  
You like a chat interface but need a board you can edit and export.  

# Getting started  
Requirements  
Python 3.10+ recommended  
A Google Gemini API key in .env  

Install  
python -m venv .venv  
. .venv/Scripts/activate   # Windows  
# source .venv/bin/activate  # macOS/Linux  
pip install -r requirements.txt  
Create .env:  
GEMINI_API_KEY=your_key_here  
Run  
streamlit run app.py  
The app will open at http://localhost:8501    

# Repository layout
├─ app.py                       # Streamlit UI  
├─ agents/  
│  └─ kitchen_agent.py          # Single ReAct agent & prompt  
├─ tools/  
│  ├─ pantry_tools.py           # Pantry CRUD & listing  
│  ├─ cuisine_tools.py          # Recipe DB; search & steps  
│  ├─ manager_tools.py          # Missing-ingredients & substitutions  
│  └─ meal_plan_tools.py        # Planner: constraints, auto_plan, shopping list, cook, export  
├─ data/  
│  ├─ pantry.json               # Your pantry store (created/updated by app)  
│  ├─ alt_units.json            # Count↔g/ml hints & labels  
│  └─ recipes.json / *.py       # Recipe dataset used by cuisine_tools  
└─ docs/                        # (Add screenshots & demo assets)  

# Tooling (APIs the agent calls)  
All tools are imported directly and asserted at startup.  

## Pantry (tools/pantry_tools.py)  
list_pantry() -> str  
add_to_pantry({ "item": str, "quantity": int, "unit": "count|g|ml" })  
remove_from_pantry({ ... })  
update_pantry({ ... })  

## Cuisine (tools/cuisine_tools.py)  
find_recipes_by_items({ "items": [str], "cuisine": str|null, "max_time": int|null, "diet": str|null, "k": int })  
list_recipes({ "cuisine": str|null, "max_time": int|null })  
get_recipe("<Dish Name>") # string only  

## Manager (tools/manager_tools.py)  
missing_ingredients("<Dish Name>") # string only  
suggest_substitutions({ "dish": ..., "deficits": [...], "pantry": [...], "constraints": {...} })  

## Planner (tools/meal_plan_tools.py)  
set_constraints({ "mode": "pantry-first-strict"|"freeform", "allow_repeats": bool, "cuisine": str|null, "diet": "veg"|"eggtarian"|"non-veg"|null, "max_time": int|null, "sub_policy": "100%-coverage" })  
get_constraints()  
auto_plan({ "days": int, "meals": int|["Breakfast","Lunch","Dinner"], "continue": bool }) -> str  
update_plan({ "day": "Day1", "meal": "Breakfast|Lunch|Dinner", "recipe_name": "<Dish>", "reason": "<why>" })  
get_shopping_list() -> str  
cook_meal({ "day": "...", "meal": "..." } | { "dish": "..." })  
save_plan({ "file_name": "optional" }) -> path  

# Planner logic  
## Pantry-first(strict)  
Build a candidate set filtered by cuisine/diet/time.  
Take a shadow snapshot of the pantry (canonicalized units).  
Compute an initial “once-coverable” list: dishes you can fully make from the starting pantry.  
Fill Day×Meals with two passes:  
Unseen-first: place each once-coverable dish at least once (respects “no consecutive repeats”). Deduct from shadow.  
Any coverable now: if nothing unseen fits, place any dish fully coverable at this moment (still checks the no-consecutive rule). Deduct from shadow.  
Stop the moment a slot can’t be filled (strict mode), but keep everything already written to the plan.  

## Freeform  
Ignore coverage; place eligible dishes (still respects no-consecutive rule).  
Shopping list covers gaps after.  
Repeat policy  
allow_repeats = False means no consecutive repeats, not global uniqueness.  
(Global-unique-until-exhausted is a planned enhancement—see Roadmap.)  

# Memory  
Conversation memory: ConversationSummaryBufferMemory (short, model-summarized context for the agent).  
Planner memory: a plain dict inside tools/meal_plan_tools.memory that stores:  
plan (Day→Meal→Dish),  
calc_log (why a dish was chosen),  
constraints,  
shopping list cache.  

The Streamlit UI reads from the same memory the tools write to, so the Planner board stays in sync with agent actions.  
