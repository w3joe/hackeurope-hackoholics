"""Schema for Alert (frontend push notification format)."""

from typing import Literal

from pydantic import BaseModel

Severity = Literal["low", "watch", "urgent"]


class AlertReformatItem(BaseModel):
    """LLM output: description and severity per risk assessment."""

    description: str
    severity: Severity


class AlertsReformatOutput(BaseModel):
    """LLM output: one reformat item per input risk assessment (same order)."""

    items: list[AlertReformatItem]
