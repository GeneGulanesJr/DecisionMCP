"""Base class for all DecisionMCP tools.

Subclass :class:`Tool` to add a new tool. Required attributes:

- ``name`` (str): unique MCP tool name (e.g. ``"decision_guard"``).
- ``description`` (str): one-line description shown to the agent.
- ``input_schema``: Pydantic model for input arguments.
- ``output_schema``: Pydantic model for output.

Required method:

- ``run(bridge, **kwargs) -> output_schema instance``: async, calls
  ``bridge.predict(...)`` and returns a populated output model.

The :meth:`to_mcp_schema` method renders the tool in MCP wire format.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..bridge import DecisionBridge


class Tool(ABC):
    """Abstract base for all DecisionMCP tools."""

    name: str = ""
    description: str = ""
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]

    @abstractmethod
    async def run(self, bridge: "DecisionBridge", **kwargs) -> BaseModel:
        """Execute the tool against ``bridge`` and return the output model."""
        raise NotImplementedError

    def to_mcp_schema(self) -> dict:
        """Render this tool in MCP wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema.model_json_schema(),
        }