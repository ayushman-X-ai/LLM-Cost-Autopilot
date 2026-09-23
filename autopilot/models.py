from typing import Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str

class CompletionRequest(BaseModel):
    messages: list[Message]
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=512, ge=1, le=32768)
    metadata: dict = Field(default_factory=dict)
    verification_mode: Literal["fast", "quality"] | None = Field(
        default=None,
        description="fast: return immediately, verify asynchronously. quality: verify before returning; escalate on failure.",
    )

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage
    latency_ms: float
    cost_usd: float
    finish_reason: str | None = None
    raw: dict = Field(default_factory=dict)

class RoutingDecision(BaseModel):
    tier: str
    model_name: str
    reason: str
    estimated_cost_usd: float
    features: dict

class CompletionResponse(BaseModel):
    id: str
    response: str
    model: str
    provider: str
    complexity_tier: str
    routing_reason: str
    usage: Usage
    cost_usd: float
    latency_ms: float
    verification_queued: bool
    verification_mode: str = "fast"
    quality_score: float | None = None
    verified: bool | None = None
    escalated: bool = False
    escalation_model: str | None = None
    routing_event: str | None = None
    original_model: str | None = None
    fallback_model: str | None = None
    reason: str | None = None
