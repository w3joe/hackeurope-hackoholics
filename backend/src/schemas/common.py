"""Shared schemas for Module 1A/1B."""

from pydantic import BaseModel, Field


class RegionInput(BaseModel):
    """Input region from TDA/Isolation Forest pipeline."""

    region_id: str
    region_name: str
    anomaly_score: float = Field(ge=0, le=1)
    persistence_shift: float
    flagged: bool
    historical_outbreaks: list[str] = Field(default_factory=list)
    population_density: str
    season: str


class Module1AOutput(BaseModel):
    """Module 1A stub output."""

    regions: list[RegionInput]

