#!/usr/bin/env python3
"""Smoke test for Module 1C — Risk Assessments to Alerts (Supabase push)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a
from module_1b import run_module_1b
from module_1c import run_module_1c


def verify_alerts_in_supabase(alert_ids: list[str]) -> bool:
    """Verify that the pushed alert IDs exist in Supabase."""
    try:
        from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            return False
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        for aid in alert_ids[:3]:  # Check first 3
            r = client.table("alerts").select("id").eq("id", aid).execute()
            if not r.data or len(r.data) == 0:
                return False
        return True
    except Exception:
        return False


if __name__ == "__main__":
    verify = "--verify" in sys.argv
    # Run 1A -> 1B to get risk_assessments, then 1C to push alerts
    module_1a_output = run_module_1a()
    module_1b_output = run_module_1b(module_1a_output)
    risk_assessments = module_1b_output.get("risk_assessments", [])

    if module_1b_output.get("error"):
        print("Module 1B error:", module_1b_output.get("error"))
        sys.exit(1)

    result = run_module_1c(risk_assessments)
    print(json.dumps(result, indent=2))

    if result.get("alerts_pushed", 0) > 0:
        pushed = result["alerts_pushed"]
        alert_ids = [a["id"] for a in result.get("alerts", [])]
        print(f"\n✓ {pushed} alert(s) pushed to Supabase.")
        if verify and alert_ids:
            if verify_alerts_in_supabase(alert_ids):
                print("✓ Verification: alerts found in Supabase table.")
            else:
                print("⚠ Verification: could not confirm alerts in table.")
    elif result.get("warning"):
        print(f"\n⚠ {result.get('warning')}")
    elif result.get("error"):
        print(f"\n✗ Error: {result.get('error')}")
        sys.exit(1)
