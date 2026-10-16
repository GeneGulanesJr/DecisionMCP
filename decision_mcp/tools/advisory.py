"""Tool: public traffic advisory classification (road advisory? / kind / routing effect).

Built for KAirOS, where posts by MMDA, DPWH, PAGASA and local governments are
read into a queue and a planner turns the relevant ones into routing rules.
The tool answers three questions about one post:

- ``is_road_advisory`` / ``p_road_advisory``: is the post about roads or
  traffic at all (as opposed to a weather bulletin, greeting or agency news)?
- ``kind``: what the post is mainly about.
- ``effect``: the KAirOS routing-rule type the post would map to:
    - ``close``: the road is shut to all vehicles
    - ``restrict``: shut to some vehicle classes (trucks, light vehicles)
    - ``divert``: the road stays open and an official alternate route is given
    - ``weight``: open but slow (lanes occupied or reduced, heavy traffic)
    - ``none``: no road is closed, restricted or slowed

``effect`` is forced to ``none`` when the post is not a road advisory; that is
the only rule applied on top of the model's answers. ``confidence`` is the
weakest of the three answers. A person still writes the rule: the tool does
not extract the road name, the dates or the alternate route. Accuracy on
Tagalog posts is much lower than on English ones with the stock Laya
checkpoints (see ``docs/TOOLS.md``).
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ._helpers import choice_of, choice_prob, require_dict
from .base import Tool

_STATE_KEY = "post"

_ADVISORY_QUESTIONS = {
    "scope": {
        "type": "choice",
        "instructions": "Is `post` an advisory about roads or traffic?",
        "criteria": {
            "road_advisory": (
                "about a road, lane, bridge or traffic: closure, flood, crash, roadwork, "
                "rerouting, or vehicle restrictions such as number coding"
            ),
            "not_road": (
                "a weather bulletin, greeting, job fair, health or public-service "
                "announcement, or news with no road affected"
            ),
        },
    },
    "kind": {
        "type": "choice",
        "instructions": "What is `post` mainly about?",
        "criteria": {
            "closure": "a road, bridge or flyover closed for a reason not listed in the other options",
            "flood": "flooding or flood depth on a road",
            "crash": "a vehicular accident or collision",
            "roadwork": "road repair, reblocking, excavation, retrofitting or construction",
            "rerouting": "a rerouting scheme or alternate routes for motorists",
            "number_coding": "number coding or truck ban: in effect, suspended or changed",
            "weather": "a weather bulletin: rain, thunderstorm or typhoon warning",
            "event": "a parade, procession, rally, fun run or concert affecting traffic",
            "other": "not about roads or traffic: greeting, job fair, health, news",
        },
    },
    "effect": {
        "type": "choice",
        "instructions": "What does `post` mean for vehicles on the road it names?",
        "criteria": {
            "close": "the road is closed to all vehicles",
            "restrict": "only some vehicles are barred, such as trucks or light vehicles",
            "divert": "the road stays open but motorists are told to take a named alternate route",
            "weight": "the road is open but slow: lanes occupied or reduced, heavy traffic expected",
            "none": "no road is closed, restricted or slowed by this post",
        },
    },
}


class AdvisoryInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="A public post by MMDA, DPWH, PAGASA or a local government, in "
        "English, Tagalog or Taglish.",
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain the post")
        return v


class AdvisoryOutput(BaseModel):
    is_road_advisory: bool
    p_road_advisory: float = Field(..., ge=0.0, le=1.0)
    # closure | flood | crash | roadwork | rerouting | number_coding | weather | event | other
    kind: str
    effect: str  # close | restrict | divert | weight | none
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class AdvisoryTool(Tool):
    name = "decision_advisory"
    description = (
        "Classify a public traffic or agency post (English, Tagalog or Taglish): is it a "
        "road advisory, its kind (closure / flood / crash / roadwork / rerouting / "
        "number_coding / weather / event / other) and the routing-rule type it maps to "
        "(close / restrict / divert / weight / none)."
    )
    input_schema = AdvisoryInput
    output_schema = AdvisoryOutput

    async def run(self, bridge: DecisionBridge, text: str) -> AdvisoryOutput:
        raw = bridge.predict_custom(text, questions=_ADVISORY_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        scope, scope_conf = choice_of(raw, "scope", self.name)
        p_road = choice_prob(raw, "scope", "road_advisory", self.name)
        kind, kind_conf = choice_of(raw, "kind", self.name)
        effect, effect_conf = choice_of(raw, "effect", self.name)
        is_road = scope == "road_advisory"
        return AdvisoryOutput(
            is_road_advisory=is_road,
            p_road_advisory=p_road,
            kind=kind,
            # A post that is not about roads maps to no routing rule.
            effect=effect if is_road else "none",
            # Weakest of the three answers: all three are at least this sure.
            confidence=min(scope_conf, kind_conf, effect_conf),
            details=raw,
        )
