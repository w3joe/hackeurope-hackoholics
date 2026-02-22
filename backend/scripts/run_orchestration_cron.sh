#!/bin/bash
# Cron wrapper for run_orchestration.py
# Run daily via: 0 0 * * * /path/to/run_orchestration_cron.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Use project venv if it exists, otherwise system python3
if [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    source "$BACKEND_DIR/.venv/bin/activate"
elif [ -f "$BACKEND_DIR/../.venv/bin/activate" ]; then
    source "$BACKEND_DIR/../.venv/bin/activate"
fi

# Run orchestrator and log
LOG_DIR="${BACKEND_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/orchestration_$(date +%Y%m%d).log"
exec python3 scripts/run_orchestration.py >> "$LOG_FILE" 2>&1
