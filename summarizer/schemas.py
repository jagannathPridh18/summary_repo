"""Pydantic request/response schemas."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Body for POST /summarize/chat."""
    dialogue: str = Field(
        ...,
        description="The chat/conversation text to summarize. Speaker labels like "
                    "'Name: message' per line improve task extraction.",
        examples=["Sarah: Mike can you fix the login crash by EOD?\nMike: Sure, I will."],
    )
    conversationId: str = Field(
        ...,
        description="Conversation identifier (any non-empty string).",
        examples=["csg_6a38deffd0ed9183d02b99db"],
    )
    userId: str = Field(
        ...,
        description="User identifier (any non-empty string).",
        examples=["69e37db7870fcd59cc78b844"],
    )

    @field_validator("dialogue")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        # empty / whitespace-only input makes the model hallucinate tasks — reject it
        if not v or not v.strip():
            raise ValueError("dialogue cannot be empty")
        return v


class TaskItem(BaseModel):
    title: str
    assigned_to: str
    assigned_from: str
    due: str
    message: str
    priority: str


class SummarizeResponse(BaseModel):
    source: str                          # "call" or "chat"
    summary: str
    notes: list[str]
    tasks: list[TaskItem]
    total_tasks: int
    speakers: str
    retried: bool
    transcript: Optional[str] = None     # populated for calls
    request_id: str
    timestamp: str
