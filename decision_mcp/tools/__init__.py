"""Tool registry.

To add a new tool:

1. Create ``decision_mcp/tools/mytool.py`` with a class extending :class:`Tool`.
2. Add an import + entry below.

That's the only edit needed — schemas, routes, and dispatch all derive
from the class.
"""
from .base import Tool
from .bug_severity import BugSeverityTool
from .commit_classify import CommitClassifyTool
from .diff_intent import DiffIntentTool
from .email import EmailTool
from .guard import GuardTool
from .moderate import ModerateTool
from .review_tone import ReviewToneTool
from .route import RouteTool
from .secret_risk import SecretRiskTool
from .test_priority import TestPriorityTool
from .triage import TriageTool
from .update import UpdateTool
from .usage_report import UsageTool

# Tools backed by engine presets (Laya upstream).
_PRESET_TOOLS: list[Tool] = [
    GuardTool(),
    RouteTool(),
    ModerateTool(),
    TriageTool(),
    EmailTool(),
]

# Tools backed by custom question schemas (coding-specific).
_CUSTOM_TOOLS: list[Tool] = [
    ReviewToneTool(),
    BugSeverityTool(),
    CommitClassifyTool(),
    TestPriorityTool(),
    SecretRiskTool(),
    DiffIntentTool(),
]

# Maintenance tools (no model inference).
_ADMIN_TOOLS: list[Tool] = [
    UpdateTool(),
    UsageTool(),
]

TOOLS: list[Tool] = _PRESET_TOOLS + _CUSTOM_TOOLS + _ADMIN_TOOLS

__all__ = ["TOOLS", "Tool"]