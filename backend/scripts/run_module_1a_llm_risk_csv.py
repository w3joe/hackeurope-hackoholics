#!/usr/bin/env python3
"""Run Module 1A LLM Risk CSV Exporter — generates llm_risk_training.csv and .jsonl for every pathogen×country."""

import sys
from pathlib import Path

# backend/scripts -> backend -> project root
_backend = Path(__file__).resolve().parent.parent
_project_root = _backend.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_backend))

from llm_finetune.llm_risk_csv import run_llm_risk_csv

if __name__ == "__main__":
    csv_path, jsonl_path = run_llm_risk_csv()
    print(f"Done. CSV: {csv_path}")
    print(f"      JSONL: {jsonl_path}")
