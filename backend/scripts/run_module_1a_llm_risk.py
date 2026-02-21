#!/usr/bin/env python3
"""Run Module 1A LLM Risk Analyzer — generates llm_risk_output.txt for every pathogen×country."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a.llm_risk_analyzer import run_llm_risk_analyzer

if __name__ == "__main__":
    result = run_llm_risk_analyzer()
    output_path = result[0] if isinstance(result, tuple) else result
    print(f"Done. Output: {output_path}")
