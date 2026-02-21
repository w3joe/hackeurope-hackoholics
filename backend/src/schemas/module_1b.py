"""Schemas for Module 1B - Country Disease Spread Risk Narrator."""

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TwelveWeekForecast(BaseModel):
    """12-week forecast for map visualization."""

    weekly_cases_per_100k: list[float] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="Estimated cases per 100k population for each of next 12 weeks",
    )
    forecast_start_week: str = Field(
        ...,
        description="ISO week when forecast begins, e.g. 2026-W08",
    )


class RiskAssessment(BaseModel):
    """Per-country risk assessment output."""

    country: str
    risk_level: RiskLevel
    spread_likelihood: float = Field(ge=0, le=1)
    reasoning: str
    recommended_disease_focus: list[str]
    twelve_week_forecast: TwelveWeekForecast


class Module1BOutput(BaseModel):
    """Module 1B output."""

    risk_assessments: list[RiskAssessment]
