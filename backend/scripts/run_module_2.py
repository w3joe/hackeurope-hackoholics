#!/usr/bin/env python3
"""Smoke test for Module 2 — Pharmacy Inventory Gap Analyzer."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a
from module_1b import run_module_1b
from module_2 import run_module_2

if __name__ == "__main__":
    regions = run_module_1a()
    risk_assessments_result = run_module_1b(regions)
    risk_assessments = risk_assessments_result.get("risk_assessments", [])
    if risk_assessments_result.get("error"):
        print("Module 1B failed, using mock risk_assessments:", file=sys.stderr)
        risk_assessments = [
            {"region_id": "R1", "risk_level": "HIGH", "recommended_disease_focus": ["respiratory infections"]},
            {"region_id": "R2", "risk_level": "MEDIUM", "recommended_disease_focus": ["dengue fever"]},
            {"region_id": "R3", "risk_level": "LOW", "recommended_disease_focus": []},
        ]
    result = run_module_2(risk_assessments=risk_assessments)
    print(json.dumps(result, indent=2))
