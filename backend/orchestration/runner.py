"""Final Orchestration Agent — reconciles all upstream outputs."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def _push_to_supabase(result: dict) -> None:
    """Push orchestration result to Supabase on success. No-op if not configured."""
    try:
        from src.orchestration_push import push_orchestration_to_supabase
        push_orchestration_to_supabase(result)
    except Exception:
        pass
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


def _run_orchestration_batch(
    batch_gap_reports: list[dict],
    batch_routing_plan: list[dict],
    risk_assessments: list[dict],
    system_prompt: str,
    user_tpl,
    llm,
    module_name: str,
) -> dict:
    """Run orchestration for a single batch. Returns dict with replenishment_directives, grand_total_cost_usd, overall_system_summary. Retries on parse/invoke errors."""
    from src.utils.logging import Timer, log_llm_call

    user_prompt = user_tpl.render(
        risk_assessments_json=json.dumps(risk_assessments, indent=2),
        gap_reports_json=json.dumps(batch_gap_reports, indent=2),
        routing_plan_json=json.dumps(batch_routing_plan, indent=2),
    )

    last_error = None
    for attempt in range(3):
        retry = attempt > 0

        try:
            with Timer() as timer:
                response = llm.invoke(
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
            last_error = e
            if attempt < 2 and ("parse" in str(e).lower() or "completion" in str(e).lower() or "output" in str(e).lower()):
                continue
            raise

    raise last_error


def run_orchestration(
    risk_assessments: list[dict],
    gap_reports: list[dict],
    routing_plan: list[dict],
    batch_size: int | None = None,
) -> dict:
    """
    Run Final Orchestration Agent.

    Inputs from Module 1B, Module 2, Module 3.
    Processes in batches (same size as Module 2) with concurrency.
    Output: replenishment_directives + grand_total_cost_usd + overall_system_summary
    """
    from src.config import ORCHESTRATION_BATCH_SIZE
    from src.schemas.orchestration import OrchestrationOutput

    module_name = "orchestration"
    input_snapshot = {
        "risk_assessments": risk_assessments,
        "gap_reports": gap_reports,
        "routing_plan": routing_plan,
    }

    try:
        effective_batch_size = batch_size if batch_size is not None else ORCHESTRATION_BATCH_SIZE
        effective_batch_size = max(1, effective_batch_size)

        routing_by_pharmacy = {r["pharmacy_id"]: r for r in routing_plan}

        if not gap_reports:
            out = {
                "replenishment_directives": [],
                "grand_total_cost_usd": 0.0,
                "overall_system_summary": "No gap reports to process.",
            }
            _push_to_supabase(out)
            return out

        env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
        system_tpl = env.get_template("orchestration_system.jinja2")
        user_tpl = env.get_template("orchestration_user.jinja2")
        system_prompt = system_tpl.render()

        llm = _get_llm()
        structured_llm = llm.with_structured_output(
            OrchestrationOutput, method="json_schema"
        )

        # Split into batches
        batches: list[tuple[list[dict], list[dict]]] = []
        for i in range(0, len(gap_reports), effective_batch_size):
            batch_reports = gap_reports[i : i + effective_batch_size]
            batch_pharmacy_ids = [r["pharmacy_id"] for r in batch_reports]
            batch_routing = [
                routing_by_pharmacy[pid]
                for pid in batch_pharmacy_ids
                if pid in routing_by_pharmacy
            ]
            batches.append((batch_reports, batch_routing))

        all_directives: list[dict] = []
        summaries: list[str] = []

        country_to_severity = {
            r["country"]: r.get("risk_level", "LOW")
            for r in risk_assessments
            if r.get("country")
        }
        for batch_reports, batch_routing in batches:
            batch_out = _run_orchestration_batch(
                batch_gap_reports=batch_reports,
                batch_routing_plan=batch_routing,
                risk_assessments=risk_assessments,
                system_prompt=system_prompt,
                user_tpl=user_tpl,
                llm=structured_llm,
                module_name=module_name,
            )
            for d in batch_out["replenishment_directives"]:
                if "severity" not in d:
                    pharmacy_country = next(
                        (r.get("country") for r in batch_reports if r.get("pharmacy_id") == d.get("pharmacy_id")),
                        None,
                    )
                    d["severity"] = country_to_severity.get(pharmacy_country, "LOW")
            all_directives.extend(batch_out["replenishment_directives"])
            summaries.append(batch_out.get("overall_system_summary", ""))

        # Reassign priority_rank globally (by total_order_cost desc = highest cost first)
        all_directives.sort(key=lambda d: d.get("total_order_cost_usd", 0), reverse=True)
        for rank, d in enumerate(all_directives, start=1):
            d["priority_rank"] = rank

        grand_total = round(sum(d["total_order_cost_usd"] for d in all_directives), 2)
        overall_summary = " ".join(s.strip() for s in summaries if s.strip()) or "Processed replenishment directives."

        out = {
            "replenishment_directives": all_directives,
            "grand_total_cost_usd": grand_total,
            "overall_system_summary": overall_summary,
        }
        _push_to_supabase(out)
        return out

    except Exception as e:
        return {
            "module": module_name,
            "error": str(e),
            "input_snapshot": input_snapshot,
        }
