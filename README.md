# Vitarisk: AI & Topological Math for Medical Restock


**Preventive Risk-Based Stock Replenishment System**

An LLM-powered pipeline that predicts epidemiological risk, detects vaccine inventory gaps, and orchestrates replenishment—before shortages hit.

---

## What It Does

Imagine a system that watches disease surveillance data across Europe, spots rising flu or RSV signals, checks pharmacy inventory, and suggests *which vaccines to order, from whom, and when*. That’s HackOholics: **risk‑driven inventory replenishment** for vaccines and pharmacy stock.

- **Trend Analysis** turns ECDC data into structured risk blocks (TDA, Holt‑Winters, SARI/ILI validation)
- **LLM fine‑tuning** trains Qwen2.5‑3B on epidemiological forecasts (GPU‑accelerated QLoRA)
- **Backend** runs a LangGraph pipeline: risk assessment → gaps → routing → orchestration
- **Frontend** shows a live map, alerts, and “take action” flows for pharmacy managers

---

## Backend Architecture Overview

<img width="727" height="400" alt="image" src="https://github.com/user-attachments/assets/0814bc3a-6be7-44a6-832c-d8af46f8dc69" />

---

### Data Ingestion
- **ECDC Data** — European Centre for Disease Prevention and Control epidemiological data forms the foundation of the pipeline

### Stage 1: Epidemiological Analysis
- **1A: Topological Data Analysis + Holt-Winters Time Series** — Detects structural patterns in disease spread using TDA, combined with Holt-Winters exponential smoothing for seasonal demand forecasting
- **1B: Disease Predictive Analysis** — A fine-tuned Qwen2.5-Instruct 3B model (via QLoRA) generates disease outbreak predictions based on the processed time series signals
- **1C: Alert Generation** — Crusoe-hosted Qwen3-235B synthesises predictions into actionable supply alerts for downstream agents

### Stage 2: Inventory Gap Analysis
- **Inventory Gaps (Claude Haiku 4.6)** — Analyses current pharmacy stock levels against predicted demand to identify critical shortfalls

### Stage 3: Distributor Retrieval
- **Retriever of Closest Distributor Algo** — Haversine-based geospatial algorithm ranks and retrieves the nearest distributors for each pharmacy based on GPS coordinates

### Orchestration Layer
- **Union Join** — Merges alert signals from the predictive analysis and inventory gap streams
- **Orchestrator Agent** — Coordinates outputs from all upstream stages to generate a unified distribution recommendation

### Output
- **Recommended Distribution of Vaccines** — Pharmacy-level vaccine allocation plan optimised by proximity, predicted demand, and current inventory gaps

### Run

```bash
cd backend
uv venv && uv pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
uv run python scripts/run_pipeline.py
uv run uvicorn api.main:app --reload --host 0.0.0.0
```

---

## Frontend

**Tech:** Next.js 16, React 19, Tailwind 4, Drizzle ORM, Leaflet, shadcn.

**Features:**

- **Dashboard** — Map of pharmacies, severity markers, real‑time alerts
- **Status** — 12‑week prediction map and validation charts (country/pathogen)
- **Orders** — Confirmed orders, grouped by manufacturer, PDF export
- **Realtime** — Supabase `postgres_changes` on `alerts`

```bash
cd frontend
pnpm install && pnpm dev
```

---

## LLM Fine-Tuning (GPU-Focused)

**Goal:** Fine‑tune Qwen2.5‑3B‑Instruct to predict disease risk and 12‑week forecasts from ECDC surveillance data.

### GPU Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| **Recommended** | 12 GB+ | RTX 3090, 4090, A100 |
| **QLoRA 4-bit** | ~12–16 GB | Batch 4 |
| **Higher batch** | ~18–20 GB | Batch 8 |

**Training time:** ~2–4 hours on RTX 4090.

### Method

- **Base:** Qwen2.5‑3B‑Instruct
- **Method:** QLoRA (4‑bit), LoRA r=16, α=32
- **Data:** ECDC (Influenza, RSV, SARS‑CoV‑2), 30+ EU countries
- **Input:** Recent observations, Holt‑Winters forecast, TDA status (anomaly, trend, z‑scores), seasonality, PCA metrics

### Workflow

```bash
cd llm_finetune
pip install -r requirements.txt

# 1. Prepare instruction data
python prepare_training_data.py

# 2. Fine-tune (GPU)
python train_qwen.py --data_dir training_data --output_dir ./qwen-epi-forecast \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4

# 3. Infer / evaluate
python inference.py --model_path ./qwen-epi-forecast
python evaluate.py --model_path ./qwen-epi-forecast --test_file training_data/test.jsonl

# 4. Merge LoRA into single model (for backend integration)
python merge_adapter.py --adapter_path ./qwen-test --output_path ./qwen-epi-forecast-merged
```

### Use Fine-Tuned Model in Module 1A

Set `USE_LOCAL_EPI_MODEL=true` in `.env` to replace the heuristic risk extractor with the merged Qwen model:

```bash
# In backend/.env
USE_LOCAL_EPI_MODEL=true
```

Requires `pip install torch transformers` in the backend environment.

For constrained environments, see `llm_finetune/INSTALL_STEP_BY_STEP.md` (minimal install, CPU fallback, Colab).

---

## Trend Analysis

**Purpose:** Turn ECDC surveillance data into structured text blocks (A, B, C) for LLM consumption—no graphs, just analysis logic.

### Pipeline (LLM_Output.py)

1. **Load ECDC data** — Respiratory viruses from GitHub (`Respiratory_viruses_weekly_data`)
2. **Sentinel positivity** — Pivot by pathogen
3. **TDA matrix** — Non‑sentinel detections/tests → positivity; join sentinel
4. **PCA** — 2 PCs, centroid distance, z‑scores
5. **TDA sliding window** — Ripser persistence entropy H0/H1; anomaly flags (2σ/3σ)
6. **SARI/ILI check** — 3‑week smoothed trend (rising/falling)
7. **Holt‑Winters** — Fit, residuals, 12‑week forecast, risk flags
8. **Assemble output** — Blocks A, B, C

### Output Blocks

| Block | Content |
|-------|---------|
| **A – Historical anomaly record** | HW residuals, TDA anomaly windows, SARI/ILI validation, seasonal pattern |
| **B – Current state** | TDA entropy tail, PCA position (last 6 weeks), latest HW residual |
| **C – Future outlook** | HW 12‑week forecast, TDA trend, convergence flags |

**Output:** `llm_risk_output.txt` — text ready for LLM prompts.

---

## Quick Start

1. **Backend:** `cd backend && uv pip install -r requirements.txt && export ANTHROPIC_API_KEY=…`
2. **Frontend:** `cd frontend && pnpm install`
3. **Supabase:** Run `supabase_*.sql` scripts from `backend/` where applicable
4. **Pipeline:** `uv run python scripts/run_pipeline.py` (backend)
5. **Frontend:** `pnpm dev` → [http://localhost:3000](http://localhost:3000)

---

## Environment (Backend)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude for LLM modules (Module 2, Orchestration) |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Supabase connection |
| `ALERTS_API_KEY` | For `/internal/alerts` |
| `CRUSOE_API_KEY` | Optional override for Module 1B |

---

## Data Sources

- **ECDC** — European Centre for Disease Prevention and Control (Influenza, RSV, SARS‑CoV‑2)
- **Vaccine stock** — `module_2/vaccine_stock_dataset.csv`
- **Distributors** — Supabase `distributors` table

---

## References

- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [ECDC Respiratory Viruses Data](https://github.com/EU-ECDC/Respiratory_viruses_weekly_data)
- [Ripser (TDA)](https://github.com/scikit-tda/ripser.py)

---

*Built for HackEurope. Risk‑driven replenishment—stay ahead of the curve.*
