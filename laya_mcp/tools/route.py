"""Tool: cheap-vs-frontier model routing decision."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import LayaBridge
from ._helpers import bin_mass, require_dict
from .base import Tool

# router_questions "difficulty" levels: 0 trivial, 1 easy, 2 moderate, 3 hard.
_FRONTIER_LEVELS = {2, 3}


class RouteInput(BaseModel):
    prompt: str = Field(..., description="The prompt that needs a model.")


class RouteOutput(BaseModel):
    tier: str = Field(..., description="'small' or 'frontier' (lowercase).")
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class RouteTool(Tool):
    name = "laya_route"
    description = "Decide whether a prompt needs a small/cheap model or a frontier/smart model."
    input_schema = RouteInput
    output_schema = RouteOutput

    async def run(self, bridge: LayaBridge, prompt: str) -> RouteOutput:
        raw = bridge.predict(prompt, preset="route")
        require_dict(raw, self.name)
        # Frontier when the model puts most of its mass on "moderate" or "hard".
        p_frontier = bin_mass(raw, "difficulty", _FRONTIER_LEVELS, self.name)
        is_frontier = p_frontier >= 0.5
        return RouteOutput(
            tier="frontier" if is_frontier else "small",
            confidence=p_frontier if is_frontier else 1.0 - p_frontier,
            details=raw,
        )
