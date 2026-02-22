#!/usr/bin/env python3
"""Smoke test for Join Module — pre-joins risk, gaps, routing per pharmacy."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a
from module_1b import run_module_1b
from module_2 import run_module_2
from module_3 import run_module_3
from module_join import run_join_module

if __name__ == "__main__":
    module_1a_output = run_module_1a()
    risk_result = run_module_1b(module_1a_output)
    risk_assessments = risk_result.get("risk_assessments", [])
    gap_result = run_module_2(risk_assessments=risk_assessments)
    gap_reports = gap_result.get("gap_reports", [])
    routing = run_module_3(gap_reports=gap_reports)
    routing_plan = routing.get("routing_plan", [])
    result = run_join_module(
        risk_assessments=risk_assessments,
        gap_reports=gap_reports,
        routing_plan=routing_plan,
    )
    print(json.dumps(result, indent=2))
