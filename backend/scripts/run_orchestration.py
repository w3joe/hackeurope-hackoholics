#!/usr/bin/env python3
"""Smoke test for Final Orchestration Agent."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a
from module_1b import run_module_1b
from module_1c import run_module_1c
from module_2 import run_module_2
from module_3 import run_module_3
from orchestration import run_orchestration

if __name__ == "__main__":
    module_1a_output = run_module_1a()
    risk_result = run_module_1b(module_1a_output)
    risk_assessments = risk_result.get("risk_assessments", [])
    if risk_result.get("error"):
        from datetime import datetime
        iso_week = datetime.now().strftime("%G-W%V")
        fc = [12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0]
        risk_assessments = [
            {"country": "Germany", "risk_level": "HIGH", "spread_likelihood": 0.72, "reasoning": "Elevated risk.", "recommended_disease_focus": ["respiratory infections"], "twelve_week_forecast": {"weekly_cases_per_100k": fc, "forecast_start_week": iso_week}},
            {"country": "Austria", "risk_level": "MEDIUM", "spread_likelihood": 0.45, "reasoning": "Moderate risk.", "recommended_disease_focus": ["dengue fever"], "twelve_week_forecast": {"weekly_cases_per_100k": fc, "forecast_start_week": iso_week}},
        ]
    # Module 1C: push alerts to Supabase Realtime
    run_module_1c(risk_assessments)
    gap_result = run_module_2(risk_assessments=risk_assessments)
    gap_reports = gap_result.get("gap_reports", [])
    if gap_result.get("error"):
        sys.exit("Module 2 failed; cannot run orchestration")
    routing = run_module_3(gap_reports=gap_reports)
    routing_plan = routing.get("routing_plan", [])
    result = run_orchestration(
        risk_assessments=risk_assessments,
        gap_reports=gap_reports,
        routing_plan=routing_plan,
    )
    print(json.dumps(result, indent=2))
