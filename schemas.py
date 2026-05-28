from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str            # one short human-readable line
    value: dict                # structured payload
    artifact_id: str | None    # handle into the artifact store
    source: str
    run_id: str
    goal_id: int | str | None
    confidence: float
    created_at: datetime


class Artifact(BaseModel):
    id: str                    # "art:<sha256-prefix>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


class Goal(BaseModel):
    id: int
    text: str                  # short imperative description
    done: bool
    attach_artifact_id: str | None


class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        return all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class DecisionOutput(BaseModel):
    answer: str | None         # exactly one of these two is populated
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def is_answer(self) -> bool:
        return self.answer is not None


class HistoryItem(BaseModel):
    iter: int
    kind: Literal["action", "answer"]
    goal_id: int | str | None = None
    tool: str | None = None
    arguments: dict | None = None
    result_descriptor: str | None = None
    artifact_id: str | None = None
    text: str | None = None


class AttachedArtifact(BaseModel):
    artifact_id: str
    content: bytes


class PerceptionInput(BaseModel):
    query: str
    hits: list[MemoryItem]
    history: list[HistoryItem]
    prior_goals: list[Goal]
    run_id: str


class DecisionInput(BaseModel):
    goal: Goal
    hits: list[MemoryItem]
    attached: list[AttachedArtifact]
    history: list[HistoryItem]
    mcp_tools: list[dict]


class ActionOutput(BaseModel):
    result_descriptor: str
    artifact_id: str | None = None


class RememberInput(BaseModel):
    raw_text: str
    source: str
    run_id: str
    goal_id: int | str | None = None


class RecordOutcomeInput(BaseModel):
    tool_call: ToolCall
    result_text: str
    artifact_id: str | None = None
    run_id: str
    goal_id: int | str | None = None


class ReadInput(BaseModel):
    query: str
    history: list[HistoryItem] = Field(default_factory=list)
    kinds: list[str] | None = None
    top_k: int = 8