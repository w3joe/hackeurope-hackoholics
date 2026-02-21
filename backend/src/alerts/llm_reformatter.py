"""LLM (Crusoe) reformatter: convert Module 1B risk assessments to Alert format."""

import json
import time
import uuid
import warnings
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.schemas.alert import AlertsReformatOutput

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = BACKEND_ROOT / "prompts"

warnings.filterwarnings("ignore", message=".*field_name='parsed'.*", category=UserWarning)


def _get_llm():
    from src.config import CRUSOE_API_KEY, CRUSOE_BASE_URL, CRUSOE_MODEL

    if not CRUSOE_API_KEY:
        raise ValueError("CRUSOE_API_KEY must be set for alerts reformatter")
    return ChatOpenAI(
        model=CRUSOE_MODEL,
        temperature=0.1,  # Lower than Module 1B for consistent formatting
        api_key=CRUSOE_API_KEY,
        base_url=CRUSOE_BASE_URL,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )


def _load_prompts():
    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
    system_tpl = env.get_template("alerts_reformat_system.jinja2")
    user_tpl = env.get_template("alerts_reformat_user.jinja2")
    return system_tpl, user_tpl


def reformat_to_alerts_with_llm(
    risk_assessments: list[dict],
    country_to_stores: dict[str, list[str]],
) -> list[dict] | None:
    """
    Use Crusoe LLM to reformat risk assessments to Alert items (description, severity).
    Returns None if LLM fails; caller should fall back to deterministic transformer.
    """
    if not risk_assessments:
        return []

    try:
        system_tpl, user_tpl = _load_prompts()
        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(
            risk_assessments_json=json.dumps(risk_assessments, indent=2)
        )

        llm = _get_llm()
        structured_llm = llm.with_structured_output(
            AlertsReformatOutput, method="json_schema"
        )

        from src.utils.logging import Timer, log_llm_call

        with Timer() as timer:
            response = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

        log_llm_call(module="alerts_reformat", latency_ms=timer.elapsed_ms)
    except Exception:
        return None

    if not response.items or len(response.items) != len(risk_assessments):
        return None

    ts_ms = int(time.time() * 1000)
    alerts: list[dict] = []

    for i, assessment in enumerate(risk_assessments):
        country = assessment.get("country", "")
        item = response.items[i]
        store_ids = country_to_stores.get(country, [])
        if not store_ids:
            store_ids = [f"country:{country}"]

        alerts.append({
            "id": str(uuid.uuid4()),
            "affectedStoreIds": store_ids,
            "timestamp": ts_ms,
            "description": item.description[:500],
            "severity": item.severity,
        })

    return alerts
