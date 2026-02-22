# Cron Setup for Orchestrator

Run the orchestrator (`scripts/run_orchestration.py`) once every 24 hours via cron.

## 1. Make the wrapper script executable

```bash
chmod +x backend/scripts/run_orchestration_cron.sh
```

## 2. Create a virtual environment (recommended)

From the project root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Add the cron job

Edit your crontab:

```bash
crontab -e
```

Add one of these lines (use the **full absolute path** to the script):

**Run daily at midnight (00:00):**
```cron
0 0 * * * /Users/w3joe/Documents/projects/hackeurope-hackoholics/backend/scripts/run_orchestration_cron.sh
```

**Run daily at 6:00 AM:**
```cron
0 6 * * * /Users/w3joe/Documents/projects/hackeurope-hackoholics/backend/scripts/run_orchestration_cron.sh
```

**Run daily at 3:00 AM:**
```cron
0 3 * * * /Users/w3joe/Documents/projects/hackeurope-hackoholics/backend/scripts/run_orchestration_cron.sh
```

## 4. Logs

Output is written to `backend/logs/orchestration_YYYYMMDD.log`. Ensure the directory exists; the wrapper creates it automatically.

```bash
tail -f backend/logs/orchestration_$(date +%Y%m%d).log
```

## Troubleshooting

- **Python not found**: Cron uses a minimal `PATH`. The wrapper adds common paths; if needed, use the full path to your Python in the script, e.g. `/Users/w3joe/Documents/projects/hackeurope-hackoholics/backend/.venv/bin/python` instead of `python3`.
- **Env vars missing**: The script loads `backend/.env` via `python-dotenv`. Ensure `.env` exists and has `ANTHROPIC_API_KEY`, `SUPABASE_URL`, etc.
- **Test manually** before relying on cron:
  ```bash
  cd backend && python3 scripts/run_orchestration.py
  ```
