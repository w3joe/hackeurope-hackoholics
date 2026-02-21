"""Module 1C — Transform risk assessments to Alert format and push to Supabase Realtime."""

from src.alerts.push import push_alerts_to_supabase
from src.utils.logging import Timer, log_llm_call


def run_module_1c(risk_assessments: list[dict]) -> dict:
    """
    Run Module 1C: Convert Module 1B risk_assessments to Alerts and push to Supabase.

    Input: risk_assessments from Module 1B (may include twelve_week_forecast for map display)
    Output: { "alerts_pushed": int, "alerts": [...], "risk_assessments": [...] } or { "error": str }
    """
    module_name = "module_1c"

    if not risk_assessments:
        log_llm_call(module=module_name, latency_ms=0)
        return {"module": module_name, "alerts_pushed": 0, "alerts": [], "risk_assessments": []}

    try:
        with Timer() as timer:
            result = push_alerts_to_supabase(risk_assessments)
        log_llm_call(module=module_name, latency_ms=timer.elapsed_ms)

        if result is None:
            return {
                "module": module_name,
                "alerts_pushed": 0,
                "alerts": [],
                "risk_assessments": risk_assessments,
                "warning": "Supabase push failed (not configured or insert error)",
            }

        return {
            "module": module_name,
            "alerts_pushed": len(result),
            "alerts": result,
            "risk_assessments": risk_assessments,
        }
    except Exception as e:
        log_llm_call(module=module_name, latency_ms=0)
        return {"module": module_name, "error": str(e), "alerts_pushed": 0, "alerts": [], "risk_assessments": []}
