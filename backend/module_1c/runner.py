"""Module 1C — Transform risk assessments to Alert format and push to Supabase Realtime."""

import random

from src.alerts.push import push_alerts_to_supabase
from src.utils.logging import Timer, log_llm_call


def _sample_countries_by_severity(risk_assessments: list[dict], n: int = 5) -> list[dict]:
    """
    Sample up to n risk assessments with diverse severities.
    Picks one random country per severity level (LOW, MEDIUM, HIGH, CRITICAL),
    then fills remaining slots with random picks until n total.
    """
    if not risk_assessments or n <= 0:
        return []

    severity_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    by_severity: dict[str, list[dict]] = {}
    for r in risk_assessments:
        level = (r.get("risk_level") or "LOW").upper()
        by_severity.setdefault(level, []).append(r)

    picked: list[dict] = []
    # One random from each severity present
    for level in severity_order:
        pool = by_severity.get(level, [])
        if pool:
            picked.append(random.choice(pool))

    # Fill to n with random from remaining
    remaining = [r for r in risk_assessments if r not in picked]
    while len(picked) < n and remaining:
        r = random.choice(remaining)
        picked.append(r)
        remaining.remove(r)

    return picked[:n]


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

    # Sample 5 random countries with different severities to generate alerts for
    sampled = _sample_countries_by_severity(risk_assessments, n=5)

    try:
        with Timer() as timer:
            result = push_alerts_to_supabase(sampled)
        log_llm_call(module=module_name, latency_ms=timer.elapsed_ms)

        if result is None:
            return {
                "module": module_name,
                "alerts_pushed": 0,
                "alerts": [],
                "risk_assessments": sampled,
                "warning": "Supabase push failed (not configured or insert error)",
            }

        return {
            "module": module_name,
            "alerts_pushed": len(result),
            "alerts": result,
            "risk_assessments": sampled,
        }
    except Exception as e:
        log_llm_call(module=module_name, latency_ms=0)
        return {"module": module_name, "error": str(e), "alerts_pushed": 0, "alerts": [], "risk_assessments": []}
