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
        risk_assessments = [
            {"country": "Germany", "risk_level": "HIGH", "recommended_disease_focus": ["respiratory infections"]},
            {"country": "Austria", "risk_level": "MEDIUM", "recommended_disease_focus": ["dengue fever"]},
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
