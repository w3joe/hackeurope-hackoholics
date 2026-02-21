"""Final Orchestration Agent — reconciles all upstream outputs."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BACKEND_ROOT / "prompts"


def _get_llm():
    from src.config import API_KEY, GEMINI_MODEL, ORCHESTRATION_TEMPERATURE

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=ORCHESTRATION_TEMPERATURE,
        api_key=API_KEY,
    )


def _validate_cost_arithmetic(output: dict) -> list[str]:
    """Verify quantity * unit_cost = total_cost, and sums match."""
    errors = []
    grand_total = 0.0
    for directive in output.get("replenishment_directives", []):
        order_total = 0.0
        for drug in directive.get("drugs_to_deliver", []):
            qty = drug.get("quantity", 0)
            unit = drug.get("unit_cost_usd", 0)
            total = drug.get("total_cost_usd", 0)
            expected = round(qty * unit, 2)
            if abs(total - expected) > 0.01:
                errors.append(
                    f"{drug.get('drug_name')}: {qty}*{unit}={expected} != {total}"
                )
            order_total += total
        reported_order = directive.get("total_order_cost_usd", 0)
        if abs(order_total - reported_order) > 0.01:
            errors.append(
                f"Pharmacy {directive.get('pharmacy_id')}: sum(drugs)={order_total} != total_order_cost_usd={reported_order}"
            )
        grand_total += order_total
    reported_grand = output.get("grand_total_cost_usd", 0)
    if abs(grand_total - reported_grand) > 0.01:
        errors.append(f"grand_total_cost_usd: sum={grand_total} != reported={reported_grand}")
    return errors


def run_orchestration(
    risk_assessments: list[dict],
    gap_reports: list[dict],
    routing_plan: list[dict],
) -> dict:
    """
    Run Final Orchestration Agent.

    Inputs from Module 1B, Module 2, Module 3.
    Output: replenishment_directives + grand_total_cost_usd + overall_system_summary
    """
    from src.schemas.orchestration import OrchestrationOutput
    from src.utils.logging import Timer, log_llm_call

    module_name = "orchestration"
    input_snapshot = {
        "risk_assessments": risk_assessments,
        "gap_reports": gap_reports,
        "routing_plan": routing_plan,
    }

    try:
        env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
        system_tpl = env.get_template("orchestration_system.jinja2")
        user_tpl = env.get_template("orchestration_user.jinja2")
        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(
            risk_assessments_json=json.dumps(risk_assessments, indent=2),
            gap_reports_json=json.dumps(gap_reports, indent=2),
            routing_plan_json=json.dumps(routing_plan, indent=2),
        )

        llm = _get_llm()
        structured_llm = llm.with_structured_output(
            OrchestrationOutput, method="json_schema"
        )

        for attempt in range(2):
            retry = attempt > 0

            with Timer() as timer:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )

            output = response.model_dump()
            validation_errors = _validate_cost_arithmetic(output)

            if validation_errors and not retry:
                user_prompt += f"\n\n[RETRY] Cost validation errors: {'; '.join(validation_errors)}. Ensure quantity*unit_cost_usd=total_cost_usd per drug, and sums match total_order_cost_usd and grand_total_cost_usd."
                continue

            # Post-correct cost arithmetic
            for d in output["replenishment_directives"]:
                order_total = 0.0
                for dr in d.get("drugs_to_deliver", []):
                    qty = dr.get("quantity", 0)
                    unit = dr.get("unit_cost_usd", 0)
                    dr["total_cost_usd"] = round(qty * unit, 2)
                    order_total += dr["total_cost_usd"]
                d["total_order_cost_usd"] = round(order_total, 2)
            output["grand_total_cost_usd"] = round(
                sum(d["total_order_cost_usd"] for d in output["replenishment_directives"]),
                2,
            )

            log_llm_call(
                module=module_name,
                latency_ms=timer.elapsed_ms,
                retry_triggered=retry,
                validation_errors=validation_errors if validation_errors else None,
            )
            return output

    except Exception as e:
        return {
            "module": module_name,
            "error": str(e),
            "input_snapshot": input_snapshot,
        }
