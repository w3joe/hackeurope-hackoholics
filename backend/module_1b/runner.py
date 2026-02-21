"""Module 1B — Country Disease Spread Risk Narrator."""

import json
import warnings
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Suppress LangChain/Pydantic structured-output serialization warning
# (parsed field expects None but receives Module1BOutput; output is correct)
warnings.filterwarnings("ignore", message=".*field_name='parsed'.*", category=UserWarning)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Resolve paths
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BACKEND_ROOT / "prompts"


def _get_llm():
    from src.config import (
        CRUSOE_API_KEY,
        CRUSOE_BASE_URL,
        CRUSOE_MODEL,
        MODULE_1B_TEMPERATURE,
    )

    if not CRUSOE_API_KEY:
        raise ValueError("CRUSOE_API_KEY must be set for Module 1B")
    return ChatOpenAI(
        model=CRUSOE_MODEL,
        temperature=MODULE_1B_TEMPERATURE,
        api_key=CRUSOE_API_KEY,
        base_url=CRUSOE_BASE_URL,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )


def _load_prompts():
    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
    system_tpl = env.get_template("module_1b_system.jinja2")
    user_tpl = env.get_template("module_1b_user.jinja2")
    return system_tpl, user_tpl


def run_module_1b(input_data: dict) -> dict:
    """
    Run Module 1B: Country Disease Spread Risk Narrator.

    Input: { "countries": [...] }
    Output: { "risk_assessments": [...] } or error object
    """
    from src.schemas.module_1b import Module1BOutput
    from src.utils.logging import Timer, log_llm_call

    module_name = "module_1b"
    input_snapshot = dict(input_data)

    try:
        countries = input_data.get("countries", [])
        if not countries:
            return {
                "module": module_name,
                "error": "No countries in input",
                "input_snapshot": input_snapshot,
            }

        system_tpl, user_tpl = _load_prompts()
        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(countries_json=json.dumps(countries, indent=2))

        llm = _get_llm()
        structured_llm = llm.with_structured_output(Module1BOutput, method="json_schema")

        for attempt in range(2):
            retry = attempt > 0
            validation_errors = []

            with Timer() as timer:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )

            # Validate: ensure all countries present
            countries_in = {c["country"] for c in countries}
            countries_out = {a.country for a in response.risk_assessments}
            missing = countries_in - countries_out
            if missing:
                validation_errors.append(f"Missing countries in output: {missing}")
                if retry:
                    user_prompt += f"\n\n[RETRY] Validation error: {validation_errors[-1]}. Include assessments for ALL input countries."
                    continue
                else:
                    log_llm_call(
                        module=module_name,
                        latency_ms=timer.elapsed_ms,
                        retry_triggered=False,
                        validation_errors=validation_errors,
                    )
                    raise ValueError(validation_errors[0])

            log_llm_call(
                module=module_name,
                latency_ms=timer.elapsed_ms,
                retry_triggered=retry,
            )
            return response.model_dump()

    except Exception as e:
        return {
            "module": module_name,
            "error": str(e),
            "input_snapshot": input_snapshot,
        }
