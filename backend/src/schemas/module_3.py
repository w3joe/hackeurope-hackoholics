"""Schemas for Module 3 - Logistics Routing Engine stub."""

from pydantic import BaseModel


class AvailableSupplier(BaseModel):
    """Supplier available for pharmacy delivery."""

    supplier_id: str
    supplier_name: str
    distance_km: float
    can_fulfil_by_urgency: bool


class RoutingPlanItem(BaseModel):
    """Per-pharmacy routing plan."""

    pharmacy_id: str
    location: str
    delivery_urgency_score: float
    estimated_delivery_time_hours: int
    capacity_remaining_units: int
    available_suppliers: list[AvailableSupplier]


class Module3Output(BaseModel):
    """Module 3 stub output."""

    routing_plan: list[RoutingPlanItem]
