"""Schemas for Final Orchestration Agent."""

from typing import Literal

from pydantic import BaseModel, Field

FeasibilityFlag = Literal["FEASIBLE", "CONSTRAINED", "INFEASIBLE"]


class DrugToDeliver(BaseModel):
    """Drug order line item."""

    drug_name: str
    category: str
    quantity: int
    unit_cost_usd: float
    total_cost_usd: float
    rationale: str


class ReplenishmentDirective(BaseModel):
    """Per-pharmacy replenishment directive."""

    pharmacy_id: str
    pharmacy_name: str
    location: str
    priority_rank: int
    assigned_supplier_id: str
    assigned_supplier_name: str
    drugs_to_deliver: list[DrugToDeliver]
    total_order_cost_usd: float
    delivery_window: str
    feasibility_flag: FeasibilityFlag
    directive_summary: str


class OrchestrationOutput(BaseModel):
    """Final orchestration output."""

    replenishment_directives: list[ReplenishmentDirective]
    grand_total_cost_usd: float
    overall_system_summary: str
