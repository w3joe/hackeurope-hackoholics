#!/usr/bin/env python3
"""Smoke test for Module 1B — Country Disease Spread Risk Narrator."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_1a import run_module_1a
from module_1b import run_module_1b

if __name__ == "__main__":
    # Use Module 1A output as input
    module_1a_output = run_module_1a()
    result = run_module_1b(module_1a_output)
    print(json.dumps(result, indent=2))
