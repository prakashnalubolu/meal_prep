from tools.cuisine_tools import list_recipes

# total list
print(list_recipes.invoke({}))                        # empty dict → no filters

# filter by cuisine
print(list_recipes.invoke({"cuisine": "italian"}))
