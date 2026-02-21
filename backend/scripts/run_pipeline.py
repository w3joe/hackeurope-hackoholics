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
    # Log Module 1C status (alerts push)
    m1c = result.get("module_1c_output", {})
    if m1c.get("alerts_pushed", 0) > 0:
        print(f"\n[module_1c] {m1c['alerts_pushed']} alert(s) pushed to Supabase.", file=sys.stderr)
    elif m1c.get("warning"):
        print(f"\n[module_1c] {m1c['warning']}", file=sys.stderr)
