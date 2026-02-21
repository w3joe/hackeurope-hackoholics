"""LangGraph pipeline for the full replenishment system."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict, total=False):
    """State for the replenishment pipeline."""

    countries: list[dict]
    risk_assessments: list[dict]
    gap_reports: list[dict]
    routing_plan: list[dict]
    final_output: dict
    module_1a_output: dict
    module_1b_output: dict
    module_1c_output: dict
    module_2_output: dict
    module_3_output: dict
    errors: list[dict]


def _extract_or_default(output: dict, key: str, default: list):
    """Extract key from output; if output is error object, return default."""
    if not output:
        return default
    if output.get("error"):
        return default
    return output.get(key, default)


def module_1a_node(state: PipelineState) -> PipelineState:
    """Run Module 1A stub."""
    from module_1a import run_module_1a

    out = run_module_1a()
    countries = out.get("countries", [])
    return {"countries": countries, "module_1a_output": out}


def module_1b_node(state: PipelineState) -> PipelineState:
    """Run Module 1B — Country Disease Spread Risk Narrator."""
    from module_1b import run_module_1b

    countries = state.get("countries", [])
    input_data = {"countries": countries}
    out = run_module_1b(input_data)
    risk_assessments = _extract_or_default(out, "risk_assessments", [])
    return {"risk_assessments": risk_assessments, "module_1b_output": out}


def module_1c_node(state: PipelineState) -> PipelineState:
    """Run Module 1C — Transform to Alerts and push to Supabase Realtime."""
    from module_1c import run_module_1c

    risk_assessments = state.get("risk_assessments", [])
    out = run_module_1c(risk_assessments)
    return {"module_1c_output": out}


def module_2_node(state: PipelineState) -> PipelineState:
    """Run Module 2 — Pharmacy Inventory Gap Analyzer."""
    from module_2 import run_module_2

    risk_assessments = state.get("risk_assessments", [])
    out = run_module_2(risk_assessments=risk_assessments)
    gap_reports = _extract_or_default(out, "gap_reports", [])
    return {"gap_reports": gap_reports, "module_2_output": out}


def module_3_node(state: PipelineState) -> PipelineState:
    """Run Module 3 stub — Logistics Routing Engine."""
    from module_3 import run_module_3

    gap_reports = state.get("gap_reports", [])
    out = run_module_3(gap_reports=gap_reports)
    routing_plan = out.get("routing_plan", [])
    return {"routing_plan": routing_plan, "module_3_output": out}


def orchestration_node(state: PipelineState) -> PipelineState:
    """Run Final Orchestration Agent."""
    from orchestration import run_orchestration

    risk_assessments = state.get("risk_assessments", [])
    gap_reports = state.get("gap_reports", [])
    routing_plan = state.get("routing_plan", [])
    out = run_orchestration(
        risk_assessments=risk_assessments,
        gap_reports=gap_reports,
        routing_plan=routing_plan,
    )
    final_output = out if "error" not in out else {"error": out.get("error"), "input_snapshot": out.get("input_snapshot")}
    return {"final_output": final_output}


def create_graph() -> StateGraph:
    """Build the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("module_1a", module_1a_node)
    graph.add_node("module_1b", module_1b_node)
    graph.add_node("module_1c", module_1c_node)
    graph.add_node("module_2", module_2_node)
    graph.add_node("module_3", module_3_node)
    graph.add_node("orchestration", orchestration_node)

    graph.add_edge(START, "module_1a")
    graph.add_edge("module_1a", "module_1b")
    graph.add_edge("module_1b", "module_1c")
    graph.add_edge("module_1c", "module_2")
    graph.add_edge("module_2", "module_3")
    graph.add_edge("module_3", "orchestration")
    graph.add_edge("orchestration", END)

    return graph


def run_pipeline() -> dict:
    """Compile and run the full pipeline."""
    g = create_graph()
    app = g.compile()
    result = app.invoke({})
    return result
