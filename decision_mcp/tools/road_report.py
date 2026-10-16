"""Tool: road-user report triage (hazard / severity / blocked road / spam → suggestion).

Built for KAirOS, where a road user taps a hazard type, picks a road and may
add a short note in English, Tagalog or Taglish, and an MMDA planner then
confirms or dismisses the report. The tool reads the note and returns typed
answers plus a deterministic ``suggestion`` for the planner.

What the model sees
-------------------

Only the note (``text``) goes into the state. ``road`` and ``depth`` never
reach the model, and ``reported_type`` reaches it only inside one question:

- With the tapped type written into the state, the model echoed it back even
  when the note described something else, which defeats the consistency check.
- A ``Road: ...`` line lowered hazard accuracy.

Notes are often short and lean on the tapped type ("Knee-deep near UST" under
Flood). Read alone they name no hazard, so when ``reported_type`` is given the
tool adds one yes/no question, ``fits_reported_type``: could this note be about
the tapped type? ``depth`` only sets a floor on ``severity``, in code.

How the outputs are derived
---------------------------

- ``hazard_from_note``: the model's choice among the six report types,
  ``other`` and ``not_a_report``, from the note alone.
- ``fits_reported_type``: P(yes) of the added question; ``None`` without a
  ``reported_type``.
- ``matches_reported_type`` (``None`` without a ``reported_type``):
    - ``True`` when ``hazard_from_note == reported_type``;
    - otherwise ``True`` when the note does not confidently name another of
      the six types (confidence below 0.5, or ``other`` / ``not_a_report``),
      ``fits_reported_type >= 0.5`` and ``spam < 0.5``;
    - ``False`` in every other case.
- ``hazard``: ``hazard_from_note``, or ``reported_type`` when the match came
  from ``fits_reported_type``.
- ``confidence``: confidence of the answer that decided ``hazard``.
- ``blocks_road``: P(yes) that the note says a lane or the road is blocked.
- ``spam``: probability of the ``not_road`` option of a two-option question.
- ``severity``: the model's choice, then raised to at least ``medium`` when
  ``blocks_road >= 0.5``, and to ``high`` when ``depth`` says the flood is
  not passable (to light vehicles or to all).
- ``suggestion``:
    - ``dismiss``: ``spam >= 0.5`` and ``hazard_from_note == "not_a_report"``
      (both questions agree the note is not a road report).
    - ``confirm``: ``spam < 0.2``, ``hazard`` is one of the six report types,
      ``matches_reported_type`` is not ``False``, and ``confidence`` is at
      least 0.5 (at least 0.7 when the match came from ``fits_reported_type``).
    - ``verify``: everything else.

The thresholds are plain cut-offs on probabilities the engine does not
calibrate; treat ``suggestion`` as an ordering hint for a human queue, not as
a decision. Accuracy on Tagalog notes is much lower than on English ones with
the stock Laya checkpoints (see ``docs/TOOLS.md``).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, choice_prob, require_dict, yes_prob
from .base import Tool

_STATE_KEY = "report"

REPORT_TYPES = ("flood", "crash", "roadwork", "parking", "stalled", "signal")

_ROAD_REPORT_QUESTIONS = {
    "hazard": {
        "type": "choice",
        "instructions": "What road problem does `report` describe?",
        "criteria": {
            "flood": "floodwater or deep water on the road",
            "crash": "a collision or accident between vehicles",
            "roadwork": "road repair, digging, excavation or construction work",
            "parking": "vehicles parked illegally or double parked",
            "stalled": "a vehicle that broke down, stalled or has a flat tire",
            "signal": "a traffic light that is off, broken or stuck",
            "other": "another road problem such as a pothole, fallen tree, rally or heavy traffic",
            "not_a_report": (
                "not about road conditions: a joke, insult, advertisement or unrelated message"
            ),
        },
    },
    # Opaque keys: they scored a little better than low/medium/high keys on the
    # development notes. They are mapped back through _SEVERITY below.
    "severity": {
        "type": "choice",
        "instructions": "How much does the problem in `report` disrupt traffic?",
        "criteria": {
            "A": "vehicles pass normally; the problem is on the side or shoulder; no lane is blocked",
            "B": "one lane is blocked, or traffic is slow but moving",
            "C": (
                "the road is closed or impassable, two or more lanes are blocked, "
                "people are injured, or the flood is knee-deep or deeper"
            ),
        },
    },
    "blocks_road": {
        "type": "noul",
        "instructions": (
            "Does `report` say that a lane or the road is blocked, closed or impassable?"
        ),
        "criteria": {
            "true": "a lane or the whole road is blocked, closed or impassable",
            "false": "traffic can pass and no lane is blocked",
        },
        "labels": {"true": "A", "false": "B"},
    },
    "message_kind": {
        "type": "choice",
        "instructions": "What kind of message is `report`?",
        "criteria": {
            "road_report": (
                "describes a condition on a road: flood, crash, roadwork, parked or "
                "stalled vehicle, traffic light, obstruction or traffic"
            ),
            "not_road": (
                "a joke, an insult, advertising or selling something, small talk, "
                "or anything unrelated to roads"
            ),
        },
    },
}

_TYPE_DESCRIPTIONS = {
    "flood": "a flood or deep water on the road",
    "crash": "a collision or accident between vehicles",
    "roadwork": "road repair, digging or construction work",
    "parking": "illegally parked or double parked vehicles",
    "stalled": "a vehicle that broke down or stalled",
    "signal": "a traffic light that is off, broken or stuck",
}

# One extra yes/no question per tapped type, added only when `reported_type` is given.
_FIT_QUESTIONS = {
    kind: {
        "type": "noul",
        "instructions": f"A road user filed `report` as: {desc}. Could the note be about that?",
        "criteria": {
            "true": f"the note describes {desc}, or adds details that are consistent with it",
            "false": (
                "the note clearly describes a different problem, or is not about a road problem"
            ),
        },
        "labels": {"true": "A", "false": "B"},
    }
    for kind, desc in _TYPE_DESCRIPTIONS.items()
}

_SEVERITY = {"A": "low", "B": "medium", "C": "high"}
_SEVERITY_ORDER = ("low", "medium", "high")

# Cut-offs used by `suggestion` (documented in the module docstring and docs/TOOLS.md).
SPAM_DISMISS = 0.5
SPAM_CLEAR = 0.2
HAZARD_CONFIDENT = 0.5
FIT_MATCH = 0.5
FIT_CLEAR = 0.7
BLOCKED = 0.5


def _depth_floor(depth: str | None) -> str | None:
    """Severity floor implied by a flood-depth label, or ``None``.

    Accepts the KAirOS keys (``passable``, ``no-light``, ``impassable``) and
    their display labels ("not passable to light vehicles", "not passable").
    Anything that does not say the flood stops vehicles sets no floor.
    """
    if not depth:
        return None
    d = depth.strip().lower()
    if d in {"no-light", "no_light", "impassable", "no-all", "no_all"}:
        return "high"
    if "not passable" in d or "impassable" in d or "hindi madaanan" in d:
        return "high"
    return None


def _at_least(severity: str, floor: str | None) -> str:
    if floor is None:
        return severity
    return max(severity, floor, key=_SEVERITY_ORDER.index)


class RoadReportInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The report note, in English, Tagalog or Taglish. Only this reaches the model.",
    )
    reported_type: Literal["flood", "crash", "roadwork", "parking", "stalled", "signal"] | None = (
        Field(
            default=None,
            description="The hazard type the road user tapped. Checked against the note.",
        )
    )
    road: str | None = Field(
        default=None,
        description="Road name from the report record. Accepted so a client can pass the "
        "whole record; not shown to the model and not used.",
    )
    depth: str | None = Field(
        default=None,
        description="Flood depth label from the report record (e.g. 'not passable to light "
        "vehicles'). Not shown to the model; raises `severity` to high when it says not passable.",
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain a note; there is nothing to triage in a blank one")
        return v


class RoadReportOutput(BaseModel):
    hazard: str  # flood | crash | roadwork | parking | stalled | signal | other | not_a_report
    hazard_from_note: str  # same labels: what the note says when read alone
    matches_reported_type: bool | None
    fits_reported_type: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: str  # low | medium | high
    blocks_road: float = Field(..., ge=0.0, le=1.0)
    spam: float = Field(..., ge=0.0, le=1.0)
    suggestion: str  # confirm | verify | dismiss
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class RoadReportTool(Tool):
    name = "decision_road_report"
    description = (
        "Triage a road-user report note (English, Tagalog or Taglish): hazard type "
        "(flood / crash / roadwork / parking / stalled / signal / other / not_a_report), "
        "whether it matches the type the user tapped, severity, P(road blocked), P(spam) "
        "and a confirm / verify / dismiss suggestion for a human reviewer."
    )
    input_schema = RoadReportInput
    output_schema = RoadReportOutput

    async def run(
        self,
        bridge: DecisionBridge,
        text: str,
        reported_type: str | None = None,
        road: str | None = None,
        depth: str | None = None,
    ) -> RoadReportOutput:
        questions = _ROAD_REPORT_QUESTIONS
        if reported_type is not None:
            questions = {**questions, "fits_reported_type": _FIT_QUESTIONS[reported_type]}
        raw = bridge.predict_custom(text, questions=questions, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        from_note, confidence = choice_of(raw, "hazard", self.name)
        level, _ = choice_of(raw, "severity", self.name)
        if level not in _SEVERITY:
            raise ToolError(self.name, f"Unexpected severity label {level!r}")
        blocks_road = yes_prob(raw, "blocks_road", self.name)
        spam = choice_prob(raw, "message_kind", "not_road", self.name)

        severity = _SEVERITY[level]
        if blocks_road >= BLOCKED:
            severity = _at_least(severity, "medium")
        severity = _at_least(severity, _depth_floor(depth))

        hazard = from_note
        matches: bool | None = None
        fits: float | None = None
        by_context = False
        if reported_type is not None:
            fits = yes_prob(raw, "fits_reported_type", self.name)
            names_another_type = (
                from_note in REPORT_TYPES
                and from_note != reported_type
                and confidence >= HAZARD_CONFIDENT
            )
            if from_note == reported_type:
                matches = True
            elif not names_another_type and fits >= FIT_MATCH and spam < SPAM_DISMISS:
                # The note names no hazard of its own but is consistent with the tapped type.
                matches, by_context = True, True
                hazard, confidence = reported_type, fits
            else:
                matches = False

        if spam >= SPAM_DISMISS and from_note == "not_a_report":
            suggestion = "dismiss"
        elif (
            spam < SPAM_CLEAR
            and hazard in REPORT_TYPES
            and matches is not False
            and confidence >= (FIT_CLEAR if by_context else HAZARD_CONFIDENT)
        ):
            suggestion = "confirm"
        else:
            suggestion = "verify"

        return RoadReportOutput(
            hazard=hazard,
            hazard_from_note=from_note,
            matches_reported_type=matches,
            fits_reported_type=fits,
            severity=severity,
            blocks_road=blocks_road,
            spam=spam,
            suggestion=suggestion,
            confidence=confidence,
            details=raw,
        )
