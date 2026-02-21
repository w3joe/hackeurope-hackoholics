"""Transform Module 1B risk assessments to Alert format for frontend.

Uses Crusoe LLM to reformat; falls back to deterministic mapping if LLM fails.
"""

import csv
import time
import uuid
from pathlib import Path

from typing import Literal

Severity = Literal["low", "watch", "urgent"]

VACCINE_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "module_2" / "vaccine_stock_dataset.csv"

_COUNTRY_TO_STORES: dict[str, list[str]] | None = None


def _load_country_to_stores() -> dict[str, list[str]]:
    """Build country -> list of Store_ID from vaccine CSV."""
    global _COUNTRY_TO_STORES
    if _COUNTRY_TO_STORES is not None:
        return _COUNTRY_TO_STORES

    if not VACCINE_CSV_PATH.exists():
        _COUNTRY_TO_STORES = {}
        return _COUNTRY_TO_STORES

    by_country: dict[str, set[str]] = {}
    with VACCINE_CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            store_id = row.get("Store_ID", "").strip()
            country = row.get("Country", "").strip()
            if store_id and country:
                by_country.setdefault(country, set()).add(store_id)

    _COUNTRY_TO_STORES = {c: sorted(ids) for c, ids in by_country.items()}
    return _COUNTRY_TO_STORES


def _risk_level_to_severity(risk_level: str) -> Severity:
    """Map Module 1B risk_level to Alert severity."""
    mapping = {
        "LOW": "low",
        "MEDIUM": "watch",
        "HIGH": "urgent",
        "CRITICAL": "urgent",
    }
    return mapping.get(risk_level.upper(), "low")


def _risk_assessments_to_alerts_deterministic(
    risk_assessments: list[dict], country_to_stores: dict[str, list[str]]
) -> list[dict]:
    """Deterministic transform (fallback when LLM fails)."""
    alerts: list[dict] = []
    ts_ms = int(time.time() * 1000)

    for assessment in risk_assessments or []:
        country = assessment.get("country", "")
        risk_level = assessment.get("risk_level", "LOW")
        reasoning = assessment.get("reasoning", "")

        store_ids = country_to_stores.get(country, [])
        if not store_ids:
            store_ids = [f"country:{country}"]

        alert = {
            "id": str(uuid.uuid4()),
            "affectedStoreIds": store_ids,
            "timestamp": ts_ms,
            "description": reasoning[:500] if reasoning else f"Risk assessment for {country}",
            "severity": _risk_level_to_severity(risk_level),
        }
        alerts.append(alert)

    return alerts


MIN_ALERTS = 1
MAX_ALERTS = 5


def _apply_alert_limits(alerts: list[dict]) -> list[dict]:
    """Enforce min 1 and max 5 alerts. When trimming, prioritize by severity (urgent > watch > low)."""
    if not alerts:
        return alerts
    severity_order = {"urgent": 0, "watch": 1, "low": 2}
    sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.get("severity", "low"), 2))
    return sorted_alerts[:MAX_ALERTS]


def risk_assessments_to_alerts(risk_assessments: list[dict]) -> list[dict]:
    """
    Transform Module 1B risk_assessments to Alert objects.

    Uses Crusoe LLM to reformat descriptions and severity; falls back to
    deterministic mapping if LLM fails. Each Alert has: id, affectedStoreIds,
    timestamp, description, severity.

    Produces between 1 and 5 alerts (when risk_assessments are present).
    """
    country_to_stores = _load_country_to_stores()

    from src.alerts.llm_reformatter import reformat_to_alerts_with_llm

    llm_result = reformat_to_alerts_with_llm(risk_assessments, country_to_stores)
    if llm_result is not None:
        return _apply_alert_limits(llm_result)

    alerts = _risk_assessments_to_alerts_deterministic(
        risk_assessments, country_to_stores
    )
    return _apply_alert_limits(alerts)
