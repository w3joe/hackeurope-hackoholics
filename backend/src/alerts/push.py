"""Push alerts to Supabase for Realtime delivery to frontend."""

from src.alerts.transformer import risk_assessments_to_alerts


def push_alerts_to_supabase(
    risk_assessments: list[dict],
    supabase_client=None,
) -> list[dict] | None:
    """
    Transform Module 1B risk_assessments to Alerts and insert into Supabase.

    Returns the list of inserted alerts on success, None if Supabase is not configured
    or insert fails.
    """
    if supabase_client is None:
        try:
            from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
            from supabase import create_client
            if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
                return None
            supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        except (ImportError, AttributeError):
            return None

    alerts = risk_assessments_to_alerts(risk_assessments)
    if not alerts:
        return []

    rows = [
        {
            "id": a["id"],
            "affectedStoreIds": a["affectedStoreIds"],
            "timestamp": a["timestamp"],
            "description": a["description"],
            "severity": a["severity"],
        }
        for a in alerts
    ]

    try:
        supabase_client.table("alerts").insert(rows).execute()
        return alerts
    except Exception as e:
        import sys
        print(f"[alerts] Supabase insert failed: {e}", file=sys.stderr)
        return None
