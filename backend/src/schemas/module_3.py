"""Schemas for Module 3 - Logistics Routing Engine (distributors from Supabase)."""

from pydantic import BaseModel


class AvailableDistributor(BaseModel):
    """Distributor available for pharmacy delivery (from Supabase)."""

    distributor_id: str
    distributor_name: str
    distance_km: float
    can_fulfil_by_urgency: bool


class RoutingPlanItem(BaseModel):
    """Per-pharmacy routing plan."""

    pharmacy_id: str
    location: str
    delivery_urgency_score: float
    estimated_delivery_time_hours: int
    capacity_remaining_units: int
    available_distributors: list[AvailableDistributor]


class Module3Output(BaseModel):
    """Module 3 output."""

    routing_plan: list[RoutingPlanItem]
