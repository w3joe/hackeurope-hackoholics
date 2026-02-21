"""Stub for Module 1A — TDA + Isolation Forest pipeline.

Runs LLM risk analyzer for every pathogen×country, writes llm_risk_output.txt,
then returns mock country-level anomaly scores for development and testing.
"""

import json
from pathlib import Path

# Resolve paths relative to backend root
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def run_module_1a() -> dict:
    """Run LLM risk analyzer, then return mock Module 1A output (countries with anomaly signals)."""
    from .llm_risk_analyzer import run_llm_risk_analyzer
    run_llm_risk_analyzer()

    mock_path = BACKEND_ROOT / "data" / "module_1a_mock.json"
    if mock_path.exists():
        return json.loads(mock_path.read_text())

    # Default inline mock — countries aligned with vaccine_stock_dataset.csv
    return {
        "countries": [
            {
                "country": "Germany",
                "anomaly_score": 0.72,
                "persistence_shift": 0.15,
                "flagged": True,
                "historical_outbreaks": ["seasonal flu 2023", "respiratory surge 2024"],
                "population_density": "high",
                "season": "winter",
            },
            {
                "country": "Austria",
                "anomaly_score": 0.45,
                "persistence_shift": 0.08,
                "flagged": True,
                "historical_outbreaks": ["dengue outbreak 2022"],
                "population_density": "medium",
                "season": "summer",
            },
            {
                "country": "Poland",
                "anomaly_score": 0.55,
                "persistence_shift": 0.10,
                "flagged": True,
                "historical_outbreaks": ["respiratory surge 2024"],
                "population_density": "medium",
                "season": "winter",
            },
            {
                "country": "Czech Republic",
                "anomaly_score": 0.32,
                "persistence_shift": 0.05,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "medium",
                "season": "winter",
            },
            {
                "country": "Hungary",
                "anomaly_score": 0.28,
                "persistence_shift": 0.04,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "medium",
                "season": "summer",
            },
            {
                "country": "Slovakia",
                "anomaly_score": 0.22,
                "persistence_shift": 0.03,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "low",
                "season": "summer",
            },
            {
                "country": "Italy",
                "anomaly_score": 0.38,
                "persistence_shift": 0.06,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "medium",
                "season": "summer",
            },
            {
                "country": "Romania",
                "anomaly_score": 0.48,
                "persistence_shift": 0.09,
                "flagged": True,
                "historical_outbreaks": ["influenza 2023"],
                "population_density": "medium",
                "season": "winter",
            },
            {
                "country": "Croatia",
                "anomaly_score": 0.25,
                "persistence_shift": 0.02,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "low",
                "season": "summer",
            },
            {
                "country": "Slovenia",
                "anomaly_score": 0.20,
                "persistence_shift": 0.02,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "low",
                "season": "summer",
            },
            {
                "country": "Bulgaria",
                "anomaly_score": 0.35,
                "persistence_shift": 0.05,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "low",
                "season": "winter",
            },
            {
                "country": "Serbia",
                "anomaly_score": 0.42,
                "persistence_shift": 0.07,
                "flagged": False,
                "historical_outbreaks": [],
                "population_density": "medium",
                "season": "summer",
            },
        ]
    }
