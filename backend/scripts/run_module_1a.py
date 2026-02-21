#!/usr/bin/env python3
"""Smoke test for Module 1A (TDA stub)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a

if __name__ == "__main__":
    result = run_module_1a()
    print(json.dumps(result, indent=2))
