"""Builders for engine-shaped results (the normalized DecisionEngine contract)."""
from __future__ import annotations


def choice(label: str, conf: float = 0.9) -> dict:
    return {
        "type": "choice",
        "choice": label,
        "probabilities": {label: conf},
        "confidence": conf,
        "answer_confidence": conf,
    }


def noul(p_yes: float) -> dict:
    conf = max(p_yes, 1.0 - p_yes)
    return {"type": "noul", "noul": p_yes, "confidence": conf, "answer_confidence": conf}


def score(level: float, legend: list[str], conf: float = 0.8, probs: list[float] | None = None) -> dict:
    if probs is None:
        rest = (1.0 - conf) / max(len(legend) - 1, 1)
        peak = min(max(round(level), 0), len(legend) - 1)
        probs = [conf if i == peak else rest for i in range(len(legend))]
    return {
        "type": "score",
        "score": level,
        "legend": {str(i): t for i, t in enumerate(legend)},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "confidence": conf,
        "answer_confidence": conf,
    }


def result(**answers: dict) -> dict:
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "routing": {"model": "english", "repo": "convaiinnovations/laya"},
    }


DIFFICULTY = [
    "trivial: a lookup or one-liner",
    "easy: short answer, no reasoning",
    "moderate: several steps",
    "hard: long multi-step reasoning or specialist knowledge",
]
FRUSTRATION = [
    "calm and neutral",
    "concerned but civil",
    "clearly annoyed",
    "very angry or using strong language",
]
EMAIL_URGENCY = ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]
