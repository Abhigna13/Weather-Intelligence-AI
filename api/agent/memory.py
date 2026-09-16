import json
from pathlib import Path


MEMORY_FILE = Path(__file__).resolve().parent / "agent_memory.json"


def load_memory():
    """Load previous agent interactions."""
    if not MEMORY_FILE.exists():
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_memory(user_query, response):
    """Save the latest agent interaction."""
    memory = load_memory()

    memory.append({
        "user_query": user_query,
        "response": response
    })

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def get_recent_memory(limit=5):
    """Return the most recent interactions."""
    memory = load_memory()
    return memory[-limit:]