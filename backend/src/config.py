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

# Anthropic Claude - uses ANTHROPIC_API_KEY
LLM_MODEL = "claude-3-5-sonnet-20241022"
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Crusoe (NVFP4/Qwen3) - Module 1B uses this
CRUSOE_BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://hackeurope.crusoecloud.com/v1/")
CRUSOE_MODEL = os.environ.get("CRUSOE_MODEL", "NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4")
CRUSOE_API_KEY = os.environ.get("CRUSOE_API_KEY")

# Module temperatures per PRD
MODULE_1B_TEMPERATURE = 0.2
MODULE_2_TEMPERATURE = 0.1
ORCHESTRATION_TEMPERATURE = 0.2

# Module 2: max pharmacies to analyze, batch size per LLM call
def _int_env(key: str, default: int, allow_zero: bool = False) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        v = int(val)
        return max(0, v) if ("MAX" in key or allow_zero) else max(1, v)
    except ValueError:
        return default


MODULE_2_MAX_PHARMACIES = _int_env("MODULE_2_MAX_PHARMACIES", 20)  # 0 = no limit; 20 avoids rate-limit stalls
MODULE_2_BATCH_SIZE = _int_env("MODULE_2_BATCH_SIZE", 5)  # 5 avoids output truncation
MODULE_2_REQUEST_TIMEOUT = _int_env("MODULE_2_REQUEST_TIMEOUT", 120)  # seconds per LLM call
MODULE_2_BATCH_DELAY = _int_env("MODULE_2_BATCH_DELAY", 5, allow_zero=True)  # seconds between batches
MODULE_2_SLOW_RETRY_MS = _int_env("MODULE_2_SLOW_RETRY_MS", 10000, allow_zero=True)  # retry batch if slower than this (0=off)

# Orchestration uses same batch size as Module 2
ORCHESTRATION_BATCH_SIZE = _int_env("ORCHESTRATION_BATCH_SIZE", 5)
ORCHESTRATION_SLOW_RETRY_MS = _int_env("ORCHESTRATION_SLOW_RETRY_MS", 10000, allow_zero=True)  # retry batch if slower (0=off)
ORCHESTRATION_REQUEST_TIMEOUT = _int_env("ORCHESTRATION_REQUEST_TIMEOUT", 120)  # seconds per LLM call

# Alerts / Supabase Realtime
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
ALERTS_API_KEY = os.environ.get("ALERTS_API_KEY", "")
