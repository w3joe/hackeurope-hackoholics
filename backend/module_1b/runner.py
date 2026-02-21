"""Module 1B — Regional Disease Spread Risk Narrator."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Resolve paths
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BACKEND_ROOT / "prompts"


def _get_llm():
    from src.config import API_KEY, GEMINI_MODEL, MODULE_1B_TEMPERATURE

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=MODULE_1B_TEMPERATURE,
        api_key=API_KEY,
    )


def _load_prompts():
    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
    system_tpl = env.get_template("module_1b_system.jinja2")
    user_tpl = env.get_template("module_1b_user.jinja2")
    return system_tpl, user_tpl


def run_module_1b(input_data: dict) -> dict:
    """
    Run Module 1B: Regional Disease Spread Risk Narrator.

    Input: { "regions": [...] }
    Output: { "risk_assessments": [...] } or error object
    """
    from src.schemas.module_1b import Module1BOutput
    from src.utils.logging import Timer, log_llm_call

    module_name = "module_1b"
    input_snapshot = dict(input_data)

    try:
        regions = input_data.get("regions", [])
        if not regions:
            return {
                "module": module_name,
                "error": "No regions in input",
                "input_snapshot": input_snapshot,
            }

        system_tpl, user_tpl = _load_prompts()
        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(regions_json=json.dumps(regions, indent=2))

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

            # Validate: ensure all regions present
            region_ids_in = {r["region_id"] for r in regions}
            region_ids_out = {a.region_id for a in response.risk_assessments}
            missing = region_ids_in - region_ids_out
            if missing:
                validation_errors.append(f"Missing regions in output: {missing}")
                if retry:
                    user_prompt += f"\n\n[RETRY] Validation error: {validation_errors[-1]}. Include assessments for ALL input regions."
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
