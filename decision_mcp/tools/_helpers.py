"""Helpers shared across tools for validating and extracting engine output.

Every engine returns the normalized decision shape::

    {"model": ..., "answers": {<question>: <answer>, ...}, "usage": {...}, "routing": {...}}

where each answer is one of three typed shapes (chosen by the question type):

- ``choice``: ``{"type": "choice", "choice": "<label>", "probabilities": {...}, ...}``
- ``noul``:   ``{"type": "noul", "noul": <P(yes)>, ...}``
- ``score``:  ``{"type": "score", "score": <0..N-1>, "legend": {"0": "...", ...}, ...}``

and every answer carries ``confidence`` and ``answer_confidence`` in [0, 1].
The helpers below validate exactly the parts we depend on and raise
:class:`ToolError` with context on failure, instead of letting bad data
silently produce empty/default results.
"""
from __future__ import annotations

from ..errors import ToolError

_VALUE_KEY = {"choice": "choice", "noul": "noul", "score": "score"}


def require_dict(raw, tool_name: str) -> dict:
    """Validate that ``raw`` is a non-empty dict.

    Raises:
        ToolError: if ``raw`` is not a dict, or is empty.
    """
    if not isinstance(raw, dict):
        raise ToolError(
            tool_name,
            f"Expected dict from engine, got {type(raw).__name__}: {raw!r}",
        )
    if not raw:
        raise ToolError(tool_name, "Engine returned an empty result.")
    return raw


def extract_answer(raw: dict, key: str, kind: str, tool_name: str) -> dict:
    """Extract and validate the answer to question ``key`` of type ``kind``.

    Validates that ``raw["answers"][key]`` is a dict of the expected ``type``,
    carries the type's value field, and has a numeric ``answer_confidence``.

    Raises:
        ToolError: on any of the above.

    Returns:
        The validated answer dict.
    """
    answers = raw.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ToolError(
            tool_name,
            f"Missing 'answers' in engine result. Got keys: {sorted(raw)}",
        )
    if key not in answers:
        raise ToolError(
            tool_name,
            f"Missing expected key {key!r} in engine answers. Got keys: {sorted(answers)}",
        )
    entry = answers[key]
    if not isinstance(entry, dict):
        raise ToolError(
            tool_name,
            f"Expected dict at {key!r}, got {type(entry).__name__}: {entry!r}",
        )
    if entry.get("type") != kind:
        raise ToolError(
            tool_name,
            f"Expected {kind!r} answer at {key!r}, got type {entry.get('type')!r}",
        )
    value_key = _VALUE_KEY[kind]
    if value_key not in entry:
        raise ToolError(tool_name, f"Missing {value_key!r} at {key!r}: {entry!r}")
    if "answer_confidence" not in entry:
        raise ToolError(tool_name, f"Missing 'answer_confidence' at {key!r}: {entry!r}")
    try:
        float(entry["answer_confidence"])
    except (TypeError, ValueError) as e:
        raise ToolError(
            tool_name,
            f"Confidence at {key!r} is not numeric: {entry['answer_confidence']!r}",
        ) from e
    if kind == "score" and not isinstance(entry.get("legend"), dict):
        raise ToolError(tool_name, f"Missing 'legend' at {key!r}: {entry!r}")
    if kind == "choice" and not isinstance(entry["choice"], str):
        raise ToolError(tool_name, f"Non-string 'choice' at {key!r}: {entry!r}")
    return entry


def choice_of(raw: dict, key: str, tool_name: str) -> tuple[str, float]:
    """``(label, confidence)`` of a ``choice`` question."""
    entry = extract_answer(raw, key, "choice", tool_name)
    return entry["choice"], float(entry["answer_confidence"])


def yes_prob(raw: dict, key: str, tool_name: str) -> float:
    """P(yes) of a ``noul`` question."""
    entry = extract_answer(raw, key, "noul", tool_name)
    try:
        return float(entry["noul"])
    except (TypeError, ValueError) as e:
        raise ToolError(tool_name, f"'noul' at {key!r} is not numeric: {entry['noul']!r}") from e


def score_level(raw: dict, key: str, tool_name: str) -> tuple[int, str, float]:
    """``(level, legend_text, confidence)`` of a ``score`` question.

    ``level`` is the nearest integer level to the model's expected score,
    clamped to the legend's range.
    """
    entry = extract_answer(raw, key, "score", tool_name)
    legend = entry["legend"]
    try:
        top = max(int(k) for k in legend)
        level = min(max(round(float(entry["score"])), 0), top)
        text = str(legend[str(level)])
    except (TypeError, ValueError, KeyError) as e:
        raise ToolError(
            tool_name, f"Malformed score/legend at {key!r}: {entry!r}", cause=e
        ) from e
    return level, text, float(entry["answer_confidence"])


def bin_mass(raw: dict, key: str, levels: set[int], tool_name: str) -> float:
    """Probability mass the model puts on ``levels`` of a ``score`` question."""
    entry = extract_answer(raw, key, "score", tool_name)
    probs = entry.get("probabilities")
    if not isinstance(probs, dict):
        raise ToolError(tool_name, f"Missing 'probabilities' at {key!r}: {entry!r}")
    try:
        return sum(float(probs[str(i)]) for i in levels)
    except (TypeError, ValueError, KeyError) as e:
        raise ToolError(
            tool_name, f"Malformed score probabilities at {key!r}: {probs!r}", cause=e
        ) from e
