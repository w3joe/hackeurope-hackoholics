"""Schemas for Module 2 - Pharmacy Inventory Gap Analyzer."""

from typing import Literal

from pydantic import BaseModel, Field

IssueType = Literal["UNDERSTOCKED", "MISSING", "NEAR_EXPIRY"]
Urgency = Literal["LOW", "MEDIUM", "HIGH"]


class StockItem(BaseModel):
    """Pharmacy stock item."""

    drug_name: str
    category: str
    quantity: int
    unit_price_usd: float
    reorder_threshold: int
    reorder_quantity: int
    days_until_expiry: int
    supplier_id: str
    supplier_name: str
    supplier_lead_time_days: int
    supplier_unit_cost_usd: float


class Pharmacy(BaseModel):
    """Pharmacy with stock."""

    pharmacy_id: str
    pharmacy_name: str
    location: str
    region_id: str
    stock: list[StockItem]


class InventoryDataset(BaseModel):
    """Full inventory dataset."""

    pharmacies: list[Pharmacy]


class RiskContextItem(BaseModel):
    """Risk context from Module 1B (subset for Module 2)."""

    region_id: str
    risk_level: str
    recommended_disease_focus: list[str]


class CriticalGap(BaseModel):
    """A critical inventory gap."""

    drug_name: str
    category: str
    issue: IssueType
    current_quantity: int
    recommended_quantity: int
    urgency: Urgency
    estimated_restock_cost_usd: float
    preferred_supplier_id: str
    preferred_supplier_name: str
    supplier_lead_time_days: int


class GapReport(BaseModel):
    """Per-pharmacy gap report."""

    pharmacy_id: str
    pharmacy_name: str
    location: str
    critical_gaps: list[CriticalGap]
    total_estimated_restock_cost_usd: float
    overall_readiness_score: float = Field(ge=0, le=1)
    summary: str


class Module2Output(BaseModel):
    """Module 2 output."""

    gap_reports: list[GapReport]
