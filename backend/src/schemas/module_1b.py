"""Schemas for Module 1B - Country Disease Spread Risk Narrator."""

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RiskAssessment(BaseModel):
    """Per-country risk assessment output."""

    country: str
    risk_level: RiskLevel
    spread_likelihood: float = Field(ge=0, le=1)
    reasoning: str
    recommended_disease_focus: list[str]


class Module1BOutput(BaseModel):
    """Module 1B output."""

    risk_assessments: list[RiskAssessment]
