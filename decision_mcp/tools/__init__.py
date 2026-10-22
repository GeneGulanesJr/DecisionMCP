"""Tool registry.

To add a new tool:

1. Create ``decision_mcp/tools/mytool.py`` with a class extending :class:`Tool`.
2. Add an import + entry below.

That's the only edit needed — schemas, routes, and dispatch all derive
from the class.
"""
from .advisory import AdvisoryTool
from .base import Tool
from .bug_severity import BugSeverityTool
from .commit_classify import CommitClassifyTool
from .diff_intent import DiffIntentTool
from .email import EmailTool
from .guard import GuardTool
from .moderate import ModerateTool
from .music_mood import MusicMoodTool
from .music_sequence import MusicSequenceTool
from .review_tone import ReviewToneTool
from .road_report import RoadReportTool
from .route import RouteTool
from .secret_risk import SecretRiskTool
from .test_priority import TestPriorityTool
from .triage import TriageTool
from .update import UpdateTool
from .usage_report import UsageTool
from .video_boundary import VideoBoundaryTool
from .video_continuity import VideoContinuityTool
from .video_region import VideoRegionTool
from .video_segment import VideoSegmentTool

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

# Tools backed by custom question schemas (road and traffic, built for KAirOS).
_ROAD_TOOLS: list[Tool] = [
    RoadReportTool(),
    AdvisoryTool(),
]

# Tools backed by custom question schemas (video analysis, built for AutoVid).
# The vision tools require a supports_images engine (d1-omni / d1-omni-tf);
# decision_video_region and decision_video_continuity also run text-only.
_VIDEO_TOOLS: list[Tool] = [
    VideoRegionTool(),
    VideoContinuityTool(),
    VideoBoundaryTool(),
    VideoSegmentTool(),
]

# Tools backed by custom question schemas (music-driven editing, AutoVid).
# decision_music_mood requires a supports_audio engine (d1-omni / d1-omni-tf);
# decision_music_sequence is text-only.
_MUSIC_TOOLS: list[Tool] = [
    MusicSequenceTool(),
    MusicMoodTool(),
]

# Maintenance tools (no model inference).
_ADMIN_TOOLS: list[Tool] = [
    UpdateTool(),
    UsageTool(),
]

TOOLS: list[Tool] = (
    _PRESET_TOOLS + _CUSTOM_TOOLS + _ROAD_TOOLS + _VIDEO_TOOLS + _MUSIC_TOOLS
    + _ADMIN_TOOLS
)

__all__ = ["TOOLS", "Tool"]