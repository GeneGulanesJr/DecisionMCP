"""Tool: prompt-injection / jailbreak detection."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import require_dict, yes_prob
from .base import Tool


class GuardInput(BaseModel):
    prompt: str = Field(..., description="Text to check for prompt injection / jailbreak.")


class GuardOutput(BaseModel):
    is_injection: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class GuardTool(Tool):
    name = "decision_guard"
    description = "Check whether a prompt contains injection or jailbreak attempts. Returns is_injection + confidence."
    input_schema = GuardInput
    output_schema = GuardOutput

    async def run(self, bridge: DecisionBridge, prompt: str) -> GuardOutput:
        raw = bridge.predict(prompt, preset="guard")
        require_dict(raw, self.name)
        # Flag if either attack question says yes; confidence is in whichever verdict we return.
        p_attack = max(
            yes_prob(raw, "jailbreak", self.name),
            yes_prob(raw, "prompt_injection", self.name),
        )
        is_injection = p_attack >= 0.5
        return GuardOutput(
            is_injection=is_injection,
            confidence=p_attack if is_injection else 1.0 - p_attack,
            details=raw,
        )
