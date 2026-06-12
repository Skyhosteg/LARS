"""
update_intent.py — The structured form of "user wants to change something"

ΔU (delta-U) is the user's interrupt, but it has to become a *structured*
intent before f(S, ΔU) can do anything useful. This module defines the
schema for that intent.

We model 9 intent types. They cover ~95% of real-user interruptions we
see in collaborative reasoning sessions.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Closed set of user intent types LARS knows how to handle."""

    # Scope changes
    SCOPE_NARROW = "scope_narrow"   # "focus on X only"
    SCOPE_EXPAND = "scope_expand"   # "also consider Y"

    # Element-level changes
    CORRECTION = "correction"       # "actually use blue, not red"
    REPLACE = "replace"             # "use Twitter instead of Facebook"
    ADD = "add"                     # "also include TikTok"
    REMOVE = "remove"               # "drop the influencer budget"

    # Structural changes
    REPRIORITIZE = "reprioritize"   # "do budget first"

    # Meta
    CLARIFY = "clarify"             # "what do you mean by 'young'?"
    ABORT = "abort"                 # "stop, restart from scratch"


class UpdateIntent(BaseModel):
    """
    The structured form of a user interrupt.

    Examples:
        "focus on Cairo only"
            → type=SCOPE_NARROW, target="geo", new_value="Cairo"
        "use Twitter instead of Facebook"
            → type=REPLACE, target="channel", old_value="Facebook", new_value="Twitter"
        "drop the influencer budget"
            → type=REMOVE, target="channel", value="influencer"
    """

    type: IntentType
    target: Optional[str] = Field(
        default=None,
        description="The aspect this intent applies to: geo, audience, channel, budget, kpi, step, etc.",
    )
    old_value: Optional[str] = Field(
        default=None,
        description="For REPLACE/CORRECTION: the value being replaced.",
    )
    value: Optional[str] = Field(
        default=None,
        description="For ADD/REMOVE: the value to add/remove.",
    )
    new_value: Optional[str] = Field(
        default=None,
        description="For SCOPE_*/REPLACE/CORRECTION: the new value.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Optional user-supplied reasoning for the change.",
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    def short(self) -> str:
        bits = [f"type={self.type.value}"]
        if self.target:
            bits.append(f"target={self.target}")
        if self.old_value:
            bits.append(f"old={self.old_value!r}")
        if self.new_value:
            bits.append(f"new={self.new_value!r}")
        if self.value:
            bits.append(f"value={self.value!r}")
        bits.append(f"conf={self.confidence:.2f}")
        return " ".join(bits)
