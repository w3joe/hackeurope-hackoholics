"""Shared schemas for Module 1A/1B."""

from pydantic import BaseModel, Field


class CountryInput(BaseModel):
    """Input country from TDA/Isolation Forest pipeline."""

    country: str
    anomaly_score: float = Field(ge=0, le=1)
    persistence_shift: float
    flagged: bool
    historical_outbreaks: list[str] = Field(default_factory=list)
    population_density: str
    season: str


class Module1AOutput(BaseModel):
    """Module 1A stub output."""

    countries: list[CountryInput]

