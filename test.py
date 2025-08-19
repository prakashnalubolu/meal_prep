from tools.meal_plan_tools import set_constraints, auto_plan, memory

print(set_constraints.invoke({"payload": {"mode": "pantry-first-strict"}}))
print(auto_plan.invoke({"payload": {"days": 3, "meals": ["Lunch","Dinner"]}}))

print("\nPlan snapshot:", memory.memories.get("plan"))
print("\nCalc log (last 3):", memory.memories.get("calc_log", [])[-3:])
