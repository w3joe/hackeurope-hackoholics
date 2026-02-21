"""Push orchestration results to Supabase upon pipeline completion."""


def push_orchestration_to_supabase(
    result: dict,
    supabase_client=None,
) -> dict | None:
    """
    Insert orchestration output into Supabase orchestration_results table.

    result: dict with replenishment_directives, grand_total_cost_usd, overall_system_summary
    Returns the inserted row on success, None if Supabase is not configured or insert fails.
    """
    if supabase_client is None:
        try:
            from src.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
            from supabase import create_client
            if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
                return None
            supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        except (ImportError, AttributeError):
            return None

    directives = result.get("replenishment_directives", [])
    grand_total = result.get("grand_total_cost_usd", 0.0)
    summary = result.get("overall_system_summary", "")

    row = {
        "replenishment_directives": directives,
        "grand_total_cost_usd": float(grand_total),
        "overall_system_summary": summary,
    }

    try:
        resp = supabase_client.table("orchestration_results").insert(row).execute()
        data = resp.data
        return data[0] if data else None
    except Exception as e:
        import sys
        print(f"[orchestration] Supabase insert failed: {e}", file=sys.stderr)
        return None
