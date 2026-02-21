#!/usr/bin/env python3
"""Smoke test for Module 3 (Logistics Routing stub)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_3 import run_module_3

if __name__ == "__main__":
    result = run_module_3()
    print(json.dumps(result, indent=2))
