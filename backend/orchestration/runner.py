"""Final Orchestration Agent — reconciles all upstream outputs."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

SUPABASE_PUSH_TIMEOUT_SEC = 30


def _push_to_supabase(result: dict) -> None:
    """Push orchestration result to Supabase on success. No-op if not configured or timeout."""
    try:
        from src.orchestration_push import push_orchestration_to_supabase

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(push_orchestration_to_supabase, result)
            try:
                future.result(timeout=SUPABASE_PUSH_TIMEOUT_SEC)
            except FuturesTimeoutError:
                import sys
                print("[orchestration] Supabase push timed out after 30s", file=sys.stderr)
    except Exception:
        pass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BACKEND_ROOT / "prompts"


def _get_llm():
    from src.config import API_KEY, GEMINI_MODEL, ORCHESTRATION_TEMPERATURE, ORCHESTRATION_REQUEST_TIMEOUT

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=ORCHESTRATION_TEMPERATURE,
        api_key=API_KEY,
        timeout=ORCHESTRATION_REQUEST_TIMEOUT,
    )


def _load_drug_quantity_by_pharmacy() -> dict[str, dict[str, int]]:
    """Load inventory and return pharmacy_id -> { drug_name -> quantity }."""
    try:
        from module_2.loader import load_vaccine_inventory
        inv = load_vaccine_inventory(max_pharmacies=0)
        result: dict[str, dict[str, int]] = {}
        for p in inv.get("pharmacies", []):
            pid = p.get("pharmacy_id", "")
            by_drug: dict[str, int] = {}
            for s in p.get("stock", []):
                name = s.get("drug_name", "")
                qty = s.get("quantity", 0)
                by_drug[name] = by_drug.get(name, 0) + qty  # sum if same drug appears twice
            result[pid] = by_drug
        return result
    except Exception:
        return {}


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
    batch_joined: list[dict],
    system_prompt: str,
    user_tpl,
    llm,
    module_name: str,
) -> dict:
    """Run orchestration for a single batch of joined pharmacies. Returns dict with replenishment_directives, grand_total_cost_usd, overall_system_summary."""
    from src.utils.logging import Timer, log_llm_call

    user_prompt = user_tpl.render(joined_pharmacies_json=json.dumps(batch_joined, indent=2))

    from src.config import ORCHESTRATION_SLOW_RETRY_MS

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

            # Retry if batch took > 10s (configurable) and we have retries left
            if ORCHESTRATION_SLOW_RETRY_MS > 0 and timer.elapsed_ms > ORCHESTRATION_SLOW_RETRY_MS and attempt < 2:
                time.sleep(2)
                continue

            output = response.model_dump()
            validation_errors = _validate_cost_arithmetic(output)

            # Always fix arithmetic post-hoc (ensures correctness without retry)
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
            # Don't log validation errors when we've corrected them (output is now correct)
            log_llm_call(
                module=module_name,
                latency_ms=timer.elapsed_ms,
                retry_triggered=retry,
                validation_errors=None,
            )
            return output

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            retriable = (
                attempt < 2
                and (
                    "parse" in err_str
                    or "completion" in err_str
                    or "output" in err_str
                    or "timeout" in err_str
                    or "timed out" in err_str
                )
            )
            if retriable:
                time.sleep(2)
                continue
            raise

    raise last_error


def run_orchestration(
    joined_pharmacies: list[dict],
    batch_size: int | None = None,
) -> dict:
    """
    Run Final Orchestration Agent.

    Input: joined_pharmacies from Join Module (pre-joined risk, gaps, routing per pharmacy).
    Processes in batches (same size as Module 2).
    Output: replenishment_directives + grand_total_cost_usd + overall_system_summary
    """
    from src.config import ORCHESTRATION_BATCH_SIZE
    from src.schemas.orchestration import OrchestrationOutput

    module_name = "orchestration"
    input_snapshot = {"joined_pharmacies": joined_pharmacies}

    try:
        effective_batch_size = batch_size if batch_size is not None else ORCHESTRATION_BATCH_SIZE
        effective_batch_size = max(1, effective_batch_size)

        if not joined_pharmacies:
            out = {
                "replenishment_directives": [],
                "grand_total_cost_usd": 0.0,
                "overall_system_summary": "No pharmacies to process.",
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

        batches: list[list[dict]] = []
        for i in range(0, len(joined_pharmacies), effective_batch_size):
            batches.append(joined_pharmacies[i : i + effective_batch_size])

        all_directives: list[dict] = []
        summaries: list[str] = []
        pharmacy_to_severity = {p["pharmacy_id"]: p.get("country_risk", {}).get("risk_level", "LOW") for p in joined_pharmacies}

        for batch_joined in batches:
            batch_out = _run_orchestration_batch(
                batch_joined=batch_joined,
                system_prompt=system_prompt,
                user_tpl=user_tpl,
                llm=structured_llm,
                module_name=module_name,
            )
            for d in batch_out["replenishment_directives"]:
                if "severity" not in d:
                    d["severity"] = pharmacy_to_severity.get(d.get("pharmacy_id"), "LOW")
            all_directives.extend(batch_out["replenishment_directives"])
            summaries.append(batch_out.get("overall_system_summary", ""))

        # Reassign priority_rank globally (by total_order_cost desc = highest cost first)
        all_directives.sort(key=lambda d: d.get("total_order_cost_usd", 0), reverse=True)
        for rank, d in enumerate(all_directives, start=1):
            d["priority_rank"] = rank

        # Add current_stock_quantity to each drug in drugs_to_deliver
        drug_qty_by_pharmacy = _load_drug_quantity_by_pharmacy()
        for d in all_directives:
            pid = d.get("pharmacy_id", "")
            drug_qtys = drug_qty_by_pharmacy.get(pid, {})
            for dr in d.get("drugs_to_deliver", []):
                dr["current_stock_quantity"] = drug_qtys.get(dr.get("drug_name", ""), 0)

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
