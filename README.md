# Meal Prep AI

Meal Prep AI is an intelligent meal planning and recipe management system powered by LLMs (Gemini) and LangChain. It helps users:

- Browse, add, and delete recipes from a diverse, multi-cuisine recipe database (JSON-based)
- Query recipes by cuisine, ingredient, or name
- Manage pantry items and check ingredient availability
- Integrate with Streamlit for a user-friendly interface

## Features
- Supports Italian, Chinese, Indian, Japanese, Thai, and Mexican cuisines
- Uses Google Gemini LLM for natural language recipe queries
- Modular agent-based architecture (CuisineAgent, PantryAgent)
- Easily extensible with new tools and cuisines

## Project Structure
```
meal_prep/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── agents/                 # LLM agent definitions
│   ├── cuisine_agent.py
│   └── pantry_agent.py
├── tools/                  # Tool functions for agents
│   ├── cuisine_tools.py
│   └── pantry_tools.py
├── data/                   # Recipe and pantry data (JSON)
│   ├── Recipe.json
│   └── pantry.json
└── .gitignore              # Files/folders to exclude from git
```

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google Gemini API key to a `.env` file as `GEMINI_API_KEY=your_key_here`
4. Run the app: `python app.py`

## License
MIT License
