"""Configuration for LLM modules."""
import os
from pathlib import Path

# Load .env from backend root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Gemini 2.5 Flash - uses GOOGLE_API_KEY or GEMINI_API_KEY
GEMINI_MODEL = "gemini-2.5-flash"
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Module temperatures per PRD
MODULE_1B_TEMPERATURE = 0.2
MODULE_2_TEMPERATURE = 0.1
ORCHESTRATION_TEMPERATURE = 0.2

# Module 2 batching: pharmacies processed per LLM call (avoids context overflow)
# MODULE_2_BATCH_SIZE env overrides; default 20
def _int_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return max(1, int(val))
    except ValueError:
        return default


MODULE_2_BATCH_SIZE = _int_env("MODULE_2_BATCH_SIZE", 20)
