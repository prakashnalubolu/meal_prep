"""tools/pantry_tools.py  - supports partial removals
============================================================
`remove_from_pantry` now mirrors the other tools: it accepts
`item`, `quantity`, and `unit`. If `quantity` is omitted (or `None`) the
whole stock for that item+unit is deleted; otherwise only the requested
amount is subtracted.
"""
import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "pantry.json"))
os.makedirs(DATA_DIR, exist_ok=True)

#############################
# JSON storage helper       #
#############################

class _PantryDB:
    """Persists pantry stock in *data/pantry.json*.
    This class provides methods to add, update, remove, and list items in the pantry, storing them as a dictionary in JSON format.
    """

    def __init__(self, path: str = DATA_PATH):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.items: Dict[str, int] = json.load(f)
            except json.JSONDecodeError:
                self.items = {}
        else:
            self.items = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2)

    # util ------------------------------------------------------------------
    @staticmethod
    def _key(item: str, unit: str) -> str:
        return f"{item.lower()} ({unit})"

    # CRUD ------------------------------------------------------------------
    def add(self, item: str, qty: int, unit: str) -> str:
        key = self._key(item, unit)
        self.items[key] = self.items.get(key, 0) + qty
        self._save()
        return f"✅ Added {qty} {unit} of {item}. Now you have {self.items[key]} {unit}."

    def update(self, item: str, qty: int, unit: str) -> str:
        key = self._key(item, unit)
        self.items[key] = qty
        self._save()
        return f"🔄 Set {item} to {qty} {unit}."

    def remove(self, item: str, qty: Optional[int], unit: str) -> str:
        key = self._key(item, unit)
        if key not in self.items:
            return f"⚠️ {item} ({unit}) not found."

        if qty is None or qty >= self.items[key]:
            del self.items[key]
            self._save()
            return f"🗑️ Removed all {item} ({unit})."
        if qty <= 0:
            return "⚠️ Quantity must be > 0."
        self.items[key] -= qty
        self._save()
        return f"🗑️ Removed {qty} {unit} of {item}. Remaining: {self.items[key]} {unit}."

    def list(self) -> str:
        if not self.items:
            return "📭 Pantry is empty."
        return "\n".join(f"{k}: {v}" for k, v in self.items.items())


_db = _PantryDB()

#############################
# LangChain tool wrappers   #
#############################

import json
from langchain_core.tools import tool

# ---- helpers -----------------------------------------------------------
def _parse_payload(payload: str) -> dict:
    """
    LangChain passes the agent's Action Input as a raw JSON string.
    This helper converts it to a Python dict and raises a clear error
    if the payload is not valid JSON.
    """
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Payload must decode to a JSON object.")
        return data
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON payload: {err}") from err


@tool
def add_to_pantry(tool_input: str) -> str:
    """Add *quantity* of *item* with the given *unit* (`count`, `g`, or `ml`)."""
    data = _parse_payload(tool_input)
    return _db.add(
        item=data["item"],
        qty=data["quantity"],
        unit=data.get("unit", "count"),
    )


@tool
def update_pantry(tool_input: str) -> str:
    """Set the stock level for *item* and *unit* exactly to *quantity*."""
    data = _parse_payload(tool_input)
    return _db.update(
        item=data["item"],
        qty=data["quantity"],
        unit=data.get("unit", "count"),
    )


@tool
def remove_from_pantry(tool_input: str) -> str:
    """
    Remove *quantity* of *item* (default all) for the specified *unit*.

    • If *quantity* is omitted, the entire entry is deleted.
    • Otherwise only that amount is deducted (or the entry removed if the
      remainder reaches zero).
    """
    data = _parse_payload(tool_input)
    return _db.remove(
        item=data["item"],
        qty=data.get("quantity"),      
        unit=data.get("unit", "count"),
    )


@tool
def list_pantry() -> str:
    """Return a human-readable listing of the pantry."""
    return _db.list()
