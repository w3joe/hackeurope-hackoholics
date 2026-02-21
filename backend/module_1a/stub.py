"""Stub for Module 1A — TDA + Isolation Forest pipeline.

Runs LLM risk analyzer for every pathogen×country, writes llm_risk_output.txt,
then returns risk_assessments in RiskAssessment schema for Module 1B.
"""

import json
from pathlib import Path

# Resolve paths relative to backend root
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _mock_risk_assessments() -> dict:
    """Fallback mock in RiskAssessment format when analyzer unavailable."""
    from datetime import datetime
    iso_week = datetime.now().strftime("%G-W%V")
    base_forecast = [12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0]
    return {
        "risk_assessments": [
            {"country": "Germany", "risk_level": "HIGH", "spread_likelihood": 0.72, "reasoning": "Historical seasonal flu and respiratory surge. Holt-Winters forecast indicates elevated risk.", "recommended_disease_focus": ["Influenza", "COVID-19"], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Austria", "risk_level": "MEDIUM", "spread_likelihood": 0.45, "reasoning": "Dengue outbreak in 2022. Moderate forecast risk.", "recommended_disease_focus": ["dengue fever"], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Poland", "risk_level": "MEDIUM", "spread_likelihood": 0.55, "reasoning": "Respiratory surge 2024. Forecast shows moderate spread.", "recommended_disease_focus": ["respiratory infections"], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Czech Republic", "risk_level": "LOW", "spread_likelihood": 0.32, "reasoning": "Below threshold. Normal seasonal pattern.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Hungary", "risk_level": "LOW", "spread_likelihood": 0.28, "reasoning": "Below threshold. Stable trend.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Slovakia", "risk_level": "LOW", "spread_likelihood": 0.22, "reasoning": "Low forecast. No anomalies.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Italy", "risk_level": "LOW", "spread_likelihood": 0.38, "reasoning": "Within normal range.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Romania", "risk_level": "MEDIUM", "spread_likelihood": 0.48, "reasoning": "Influenza 2023. Some elevated risk weeks.", "recommended_disease_focus": ["Influenza"], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Croatia", "risk_level": "LOW", "spread_likelihood": 0.25, "reasoning": "Stable. No significant signals.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Slovenia", "risk_level": "LOW", "spread_likelihood": 0.20, "reasoning": "Low activity.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Bulgaria", "risk_level": "LOW", "spread_likelihood": 0.35, "reasoning": "Normal range.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
            {"country": "Serbia", "risk_level": "LOW", "spread_likelihood": 0.42, "reasoning": "Slightly elevated but below threshold.", "recommended_disease_focus": [], "twelve_week_forecast": {"weekly_cases_per_100k": base_forecast, "forecast_start_week": iso_week}},
        ]
    }


def run_module_1a() -> dict:
    """Run LLM risk analyzer, then return risk_assessments in RiskAssessment schema."""
    from .llm_risk_analyzer import run_llm_risk_analyzer, extract_risk_assessments
    try:
        result = run_llm_risk_analyzer()
        if isinstance(result, tuple):
            _, risk_assessments = result
        else:
            risk_assessments = extract_risk_assessments()
        if risk_assessments:
            return {"risk_assessments": risk_assessments}
    except Exception:
        pass
    mock_path = BACKEND_ROOT / "data" / "module_1a_mock.json"
    if mock_path.exists():
        data = json.loads(mock_path.read_text())
        if "risk_assessments" in data:
            return data
        if "countries" in data:
            return _mock_risk_assessments()
    return _mock_risk_assessments()
