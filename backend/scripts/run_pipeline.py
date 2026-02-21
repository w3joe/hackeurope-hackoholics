#!/usr/bin/env python3
"""Run the full LangGraph pipeline end-to-end."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import run_pipeline

if __name__ == "__main__":
    result = run_pipeline()
    # Output final result; intermediate state also available
    final = result.get("final_output", {})
    print(json.dumps(final, indent=2))
