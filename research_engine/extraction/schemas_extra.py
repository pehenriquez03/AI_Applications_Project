"""Model output schemas that are not part of contracts.py.

NarrativeOut: the narrative call's answer. Sections are fixed named fields
(not a free dict) because structured outputs turn `dict[str, str]` into an
empty object; the JSON keys are the human section names via aliases.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SECTION_NAMES = [
    "What the paper proposes",
    "What we implemented and changed",
    "What the backtest found",
    "What the simulation and economics found",
    "How reliable the interpretation was",
    "What to do next",
]


class NarrativeSections(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_proposes: str = Field(alias=SECTION_NAMES[0])
    implemented_and_changed: str = Field(alias=SECTION_NAMES[1])
    backtest_found: str = Field(alias=SECTION_NAMES[2])
    simulation_and_economics: str = Field(alias=SECTION_NAMES[3])
    interpretation_reliability: str = Field(alias=SECTION_NAMES[4])
    next_steps: str = Field(alias=SECTION_NAMES[5])


class NarrativeOut(BaseModel):
    summary: str = Field(description="4-6 sentences, first word is the recommendation")
    sections: NarrativeSections

    def to_contract_dict(self) -> dict:
        """{"summary": str, "sections": {section name: str}} as RunFiles.narrative expects."""
        return {"summary": self.summary, "sections": self.sections.model_dump(by_alias=True)}
