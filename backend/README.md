# Preventive Risk-Based Stock Replenishment System

LLM-powered pipeline for vaccine/pharmacy inventory replenishment based on epidemiological risk, inventory gaps, and logistics.

## Structure

```
backend/
├── prompts/                    # Jinja2 prompt templates
│   ├── module_1b_*.jinja2
│   ├── module_2_*.jinja2
│   ├── alerts_reformat_*.jinja2
│   └── orchestration_*.jinja2
├── src/
│   ├── config.py               # Gemini model config
│   ├── schemas/                # Pydantic models
│   ├── alerts/                 # Module 1C: transformer, LLM reformatter, Supabase push
│   └── utils/
├── module_1a/                  # TDA stub (non-LLM)
├── module_1b/                  # Regional Disease Spread Risk Narrator (LLM)
├── module_1c/                  # Risk Assessments → Alerts (LLM + Supabase Realtime)
├── module_2/                   # Pharmacy Inventory Gap Analyzer (LLM)
│   ├── vaccine_stock_dataset.csv
│   └── loader.py               # CSV to inventory transform
├── module_3/                   # Logistics Routing (closest distributors from Supabase)
├── module_join/                 # Join Module (pre-joins risk, gaps, routing per pharmacy)
├── orchestration/              # Final Orchestration Agent (LLM)
├── pipeline/                   # LangGraph pipeline
└── scripts/                    # Smoke test scripts
```

## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
cd backend
uv venv
uv pip install -r requirements.txt
```

2. Set your Gemini API key:

```bash
export GOOGLE_API_KEY=your-gemini-api-key
# or
export GEMINI_API_KEY=your-gemini-api-key
```

## Run

From the `backend/` directory:

```bash
# Smoke tests (individual modules)
uv run python scripts/run_module_1a.py
uv run python scripts/run_module_1b.py
uv run python scripts/run_module_1c.py     # pushes alerts to Supabase
uv run python scripts/run_module_1c.py --verify  # push + verify in table
uv run python scripts/run_module_2.py
uv run python scripts/run_module_3.py
uv run python scripts/run_module_join.py
uv run python scripts/run_orchestration.py

# Full LangGraph pipeline
uv run python scripts/run_pipeline.py

# Alerts API (Supabase Realtime push)
uv run python scripts/run_alerts_api.py
# or: uv run uvicorn api.main:app --reload --host 0.0.0.0
```

## LLM

Uses **Gemini 2.5 Flash** via `langchain-google-genai`. Configured in `src/config.py`.

### Module 2 (Gap Analyzer)

The vaccine dataset CSV (`module_2/vaccine_stock_dataset.csv`) uses schema: `Snapshot_Date, Country, City, Address, Postal_Code, Store_ID, Target_Disease, Vaccine_Brand, Manufacturer, Stock_Quantity, Min_Stock_Level, Expiry_Date, Storage_Type`.

The dataset has 100 stores. Module 2 limits to 50 pharmacies by default (`MODULE_2_MAX_PHARMACIES=50`, use 0 for all), processes in batches of 5 (`MODULE_2_BATCH_SIZE=5`), sequential with per-batch retry. Orchestration uses the same batch size and concurrency.

## Alerts (Supabase Realtime)

When Module 1B completes, risk assessments are transformed into Alerts and pushed to Supabase. The frontend subscribes to the `alerts` table for real-time notifications.

1. Run `supabase_alerts.sql` in the Supabase SQL Editor.
2. Set env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `ALERTS_API_KEY`.
3. Pipeline automatically pushes alerts when it runs. For external triggers, POST to `POST /internal/alerts` with `X-API-Key` and body `{"risk_assessments": [...]}`.
4. Frontend: subscribe to `postgres_changes` on `alerts` (INSERT). Row columns match Alert interface: `id`, `affectedStoreIds`, `timestamp`, `description`, `severity`.

### Module 3 (Logistics Routing)

Module 3 retrieves the closest distributors from Supabase per pharmacy. Run `supabase_distributors.sql` in the Supabase SQL Editor to create and seed the `distributors` table. If Supabase is not configured, falls back to hardcoded distributors. The orchestration uses `assigned_distributor_id` and `assigned_distributor_name` from Module 3's routing plan (not supplier ids from Module 2).

## Module 1A Results (Supabase)

Module 1A risk assessments are stored in Supabase. On each run, the table is truncated and repopulated with the latest results.

1. Run `supabase_module_1a_results.sql` in the Supabase SQL Editor to create the `module_1a_results` table.
2. With `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` set, `run_module_1a` automatically truncates and pushes results.
3. Table columns: `id`, `country`, `risk_level`, `spread_likelihood`, `reasoning`, `recommended_disease_focus` (JSONB), `twelve_week_forecast` (JSONB), `created_at`.

## Orchestration Results (Supabase)

When orchestration completes successfully, the result is pushed to Supabase for persistence and real-time access.

1. Run `supabase_orchestration_results.sql` in the Supabase SQL Editor to create the `orchestration_results` table.
2. With `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` set, the pipeline and `run_orchestration.py` automatically push results on success.
3. Table columns: `id`, `replenishment_directives` (JSONB), `grand_total_cost_usd`, `overall_system_summary`, `created_at`.
