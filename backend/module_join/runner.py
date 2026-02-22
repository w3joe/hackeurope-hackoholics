"""Join Module — Pre-join risk assessments, gap reports, and routing plan into per-pharmacy structures."""

from typing import Any


def run_join_module(
    risk_assessments: list[dict],
    gap_reports: list[dict],
    routing_plan: list[dict],
) -> dict:
    """
    Join outputs from Module 1B, Module 2, and Module 3 into per-pharmacy enriched structures.

    Each joined pharmacy has:
    - pharmacy_id, pharmacy_name, location, country (from Module 2)
    - country_risk: risk_level, spread_likelihood, recommended_disease_focus (from Module 1B)
    - critical_gaps (from Module 2)
    - available_distributors, delivery_urgency_score, estimated_delivery_time_hours (from Module 3)
    """
    country_to_risk: dict[str, dict] = {}
    for r in risk_assessments or []:
        country = (r.get("country") or "").strip()
        if country:
            country_to_risk[country] = {
                "risk_level": r.get("risk_level", "LOW"),
                "spread_likelihood": r.get("spread_likelihood", 0),
                "recommended_disease_focus": r.get("recommended_disease_focus", []),
                "reasoning": r.get("reasoning", "")[:300],
            }

    routing_by_pharmacy: dict[str, dict] = {r["pharmacy_id"]: r for r in routing_plan or []}

    joined: list[dict[str, Any]] = []
    for report in gap_reports or []:
        pharmacy_id = report.get("pharmacy_id", "")
        country = report.get("country", "").strip()

        country_risk = country_to_risk.get(country)
        if not country_risk:
            country_risk = {
                "risk_level": "LOW",
                "spread_likelihood": 0,
                "recommended_disease_focus": [],
                "reasoning": "",
            }

        routing = routing_by_pharmacy.get(pharmacy_id, {})
        available_distributors = routing.get("available_distributors", [])
        if not available_distributors:
            available_distributors = routing.get("available_suppliers", [])  # legacy fallback

        joined.append({
            "pharmacy_id": pharmacy_id,
            "pharmacy_name": report.get("pharmacy_name", ""),
            "location": report.get("location", ""),
            "country": country,
            "country_risk": country_risk,
            "critical_gaps": report.get("critical_gaps", []),
            "total_estimated_restock_cost_usd": report.get("total_estimated_restock_cost_usd", 0),
            "overall_readiness_score": report.get("overall_readiness_score", 0),
            "summary": report.get("summary", ""),
            "available_distributors": available_distributors,
            "delivery_urgency_score": routing.get("delivery_urgency_score", 0.5),
            "estimated_delivery_time_hours": routing.get("estimated_delivery_time_hours", 24),
            "capacity_remaining_units": routing.get("capacity_remaining_units", 500),
        })

    return {"joined_pharmacies": joined}
