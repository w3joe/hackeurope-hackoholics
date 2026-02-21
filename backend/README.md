# Preventive Risk-Based Stock Replenishment System

LLM-powered pipeline for vaccine/pharmacy inventory replenishment based on epidemiological risk, inventory gaps, and logistics.

## Structure

```
backend/
├── prompts/                    # Jinja2 prompt templates
│   ├── module_1b_*.jinja2
│   ├── module_2_*.jinja2
│   └── orchestration_*.jinja2
├── src/
│   ├── config.py               # Gemini model config
│   ├── schemas/                # Pydantic models
│   └── utils/
├── module_1a/                  # TDA stub (non-LLM)
├── module_1b/                  # Regional Disease Spread Risk Narrator (LLM)
├── module_2/                   # Pharmacy Inventory Gap Analyzer (LLM)
│   ├── vaccine_stock_dataset.csv
│   └── loader.py               # CSV to inventory transform
├── module_3/                   # Logistics Routing stub (non-LLM)
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
uv run python scripts/run_module_2.py
uv run python scripts/run_module_3.py
uv run python scripts/run_orchestration.py

# Full LangGraph pipeline
uv run python scripts/run_pipeline.py
```

## LLM

Uses **Gemini 2.5 Flash** via `langchain-google-genai`. Configured in `src/config.py`.

### Module 2 (Gap Analyzer)

The vaccine dataset CSV (`module_2/vaccine_stock_dataset.csv`) uses schema: `Snapshot_Date, Country, City, Address, Postal_Code, Store_ID, Target_Disease, Vaccine_Brand, Manufacturer, Stock_Quantity, Min_Stock_Level, Expiry_Date, Storage_Type`.

The dataset has 100 stores. Module 2 processes pharmacies in **batches** to avoid context overflow and timeouts. Set `MODULE_2_BATCH_SIZE` (default 20) to control pharmacies per LLM call—e.g. all 100 stores in 5 batches.
