"""Push Module 1A risk assessment results to Supabase.

Table is truncated at the start of each run, then repopulated with new results.
"""

import sys


def _get_supabase_client():
    """Get Supabase client. Returns None if not configured."""
    try:
        from src.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            return None
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except (ImportError, AttributeError):
        return None


def truncate_module_1a_results(supabase_client=None) -> bool:
    """
    Truncate module_1a_results table (clear all rows).
    Call at the start of each Module 1A run.

    Returns True on success, False if Supabase is not configured or truncate fails.
    """
    client = supabase_client or _get_supabase_client()
    if client is None:
        return False
    try:
        client.rpc("truncate_module_1a_results").execute()
        return True
    except Exception as e:
        print(f"[module_1a] Supabase truncate failed: {e}", file=sys.stderr)
        return False


def push_module_1a_results(
    risk_assessments: list[dict],
    supabase_client=None,
) -> list[dict] | None:
    """
    Insert Module 1A risk_assessments into Supabase module_1a_results table.

    risk_assessments: list of dicts with country, risk_level, spread_likelihood,
        reasoning, recommended_disease_focus, twelve_week_forecast.

    Returns the list of inserted rows on success, None if Supabase is not
    configured or insert fails.
    """
    client = supabase_client or _get_supabase_client()
    if client is None:
        return None

    if not risk_assessments:
        return []

    rows = []
    for a in risk_assessments:
        rows.append({
            "country": a.get("country", ""),
            "risk_level": a.get("risk_level", "LOW"),
            "spread_likelihood": float(a.get("spread_likelihood", 0)),
            "reasoning": a.get("reasoning", ""),
            "recommended_disease_focus": a.get("recommended_disease_focus", []),
            "twelve_week_forecast": a.get("twelve_week_forecast", {}),
        })

    try:
        resp = client.table("module_1a_results").insert(rows).execute()
        return resp.data or []
    except Exception as e:
        print(f"[module_1a] Supabase insert failed: {e}", file=sys.stderr)
        return None
