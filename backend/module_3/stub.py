"""Stub for Module 3 — Logistics Routing Engine.

Returns mock routing plan for development and testing.
The real formula-based routing engine will replace this at integration.
"""

import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def run_module_3(gap_reports: list[dict] | None = None) -> dict:
    """Return mock Module 3 output (routing plan per pharmacy).

    If gap_reports is provided, generates routing plan for pharmacies in gap_reports.
    Otherwise uses default pharmacy IDs from inventory.
    """
    mock_path = BACKEND_ROOT / "data" / "module_3_mock.json"
    if mock_path.exists():
        return json.loads(mock_path.read_text())

    # Build routing plan from gap_reports or use default pharmacy IDs
    pharmacy_ids = (
        [g["pharmacy_id"] for g in gap_reports]
        if gap_reports
        else ["PH001", "PH002", "PH003", "PH004", "PH005"]
    )
    locations = {
        "PH001": "123 Main St, Central City",
        "PH002": "45 River Rd, Riverside",
        "PH003": "78 Mountain View, Highland",
        "PH004": "12 Harbour St, Coastal Bay",
        "PH005": "99 Northern Ave, North District",
    }

    routing_plan = []
    for i, pid in enumerate(pharmacy_ids):
        urgency = 0.9 - (i * 0.15)  # Decreasing urgency by rank
        if urgency < 0.3:
            urgency = 0.3
        routing_plan.append(
            {
                "pharmacy_id": pid,
                "location": locations.get(pid, f"Location for {pid}"),
                "delivery_urgency_score": round(urgency, 2),
                "estimated_delivery_time_hours": 4 + i * 2,
                "capacity_remaining_units": 500 - i * 80,
                "available_suppliers": [
                    {
                        "supplier_id": "SUP1",
                        "supplier_name": "MedSupply Corp",
                        "distance_km": 12.5 + i * 5,
                        "can_fulfil_by_urgency": i < 3,
                    },
                    {
                        "supplier_id": "SUP2",
                        "supplier_name": "PharmaDirect",
                        "distance_km": 8.0 + i * 3,
                        "can_fulfil_by_urgency": True,
                    },
                    {
                        "supplier_id": "SUP3",
                        "supplier_name": "ViralPharm",
                        "distance_km": 25.0 + i * 2,
                        "can_fulfil_by_urgency": i < 2,
                    },
                ],
            }
        )

    return {"routing_plan": routing_plan}
