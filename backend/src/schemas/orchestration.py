"""Schemas for Final Orchestration Agent."""

from typing import Literal

from pydantic import BaseModel, Field

FeasibilityFlag = Literal["FEASIBLE", "CONSTRAINED", "INFEASIBLE"]

# Severity from Module 1B (country risk_level)
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


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
    severity: Severity = Field(
        ...,
        description="Country risk level from Module 1B (risk_level): LOW, MEDIUM, HIGH, or CRITICAL",
    )
    priority_rank: int
    assigned_distributor_id: str = Field(
        ...,
        description="Must appear in routing_plan.available_distributors for this pharmacy (from Module 3)",
    )
    assigned_distributor_name: str
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
