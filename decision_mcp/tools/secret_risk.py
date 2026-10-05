"""Tool: detect leaked secrets / credentials in text."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "text"

_SECRET_QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of secret or credential, if any, does `text` contain?",
        "criteria": {
            "none": "no secret or credential",
            "api_key": "an API key or access key for a service",
            "password": "a password or passphrase",
            "token": "a bearer, OAuth, JWT or session token",
            "cert": "a certificate or TLS private key",
            "ssh_key": "an SSH private key",
            "aws_creds": "AWS access key id or secret access key",
            "other": "some other secret",
        },
    },
    "risk": {
        "type": "choice",
        "instructions": "How damaging would it be if `text` were published publicly?",
        "criteria": {
            "none": "nothing sensitive",
            "low": "placeholder, example or expired-looking value",
            "medium": "internal identifier or low-privilege credential",
            "high": "a working-looking credential",
            "critical": "private key or cloud/admin credential",
        },
    },
}


class SecretRiskInput(BaseModel):
    text: str = Field(..., description="Text to scan for secrets / API keys / credentials.")


class SecretRiskOutput(BaseModel):
    kind: str  # none | api_key | password | token | cert | ssh_key | aws_creds | other
    risk: str  # none | low | medium | high | critical
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class SecretRiskTool(Tool):
    name = "decision_secret_risk"
    description = (
        "Scan text for leaked credentials (API keys, passwords, tokens, certs, SSH keys, "
        "AWS creds). Returns the kind of secret detected and a risk level. "
        "Use BEFORE saving any text to memory, posting to public channels, or "
        "including in commit messages."
    )
    input_schema = SecretRiskInput
    output_schema = SecretRiskOutput

    async def run(self, bridge: DecisionBridge, text: str) -> SecretRiskOutput:
        raw = bridge.predict_custom(text, questions=_SECRET_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        kind, _ = choice_of(raw, "kind", self.name)
        risk, confidence = choice_of(raw, "risk", self.name)
        return SecretRiskOutput(
            kind=kind,
            risk=risk,
            confidence=confidence,
            details=raw,
        )
