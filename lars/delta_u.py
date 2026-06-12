"""
delta_u.py — The ΔU Parser

Takes raw user text and emits a structured UpdateIntent.

Two backends, same interface:
  - DeltaUParserLLM  : uses the LLM (gpt-4o-mini) with a few-shot prompt
  - DeltaUParserMock : returns canned intents for testing / demo

In production you'd use the LLM. The mock exists so the demo and tests
run offline.
"""

from __future__ import annotations

import json
import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from .llm import LLMClient, LLMError
from .prompts import SYSTEM_DELTA_U
from .update_intent import UpdateIntent

T = TypeVar("T", bound=BaseModel)


class DeltaUParser:
    """Base interface."""

    def parse(self, user_text: str) -> UpdateIntent:
        raise NotImplementedError


class DeltaUParserLLM(DeltaUParser):
    """LLM-backed parser."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def parse(self, user_text: str) -> UpdateIntent:
        user_msg = (
            f"USER INTERRUPT (raw text):\n{user_text}\n\n"
            f"Classify into a structured UpdateIntent. Return JSON only."
        )
        return self.llm.complete_json(SYSTEM_DELTA_U, user_msg, UpdateIntent)


class DeltaUParserMock(DeltaUParser):
    """
    Heuristic, regex-based parser. Deterministic, no network.

    Handles the demo's known phrases. For production, swap in
    DeltaUParserLLM.

    NOTE: This parser is English-only. For multilingual support
    (Arabic, etc.) use DeltaUParserLLM, which handles any language
    via the underlying model's multilingual capabilities.
    """

    # Order matters: more specific patterns first.
    RULES = [
        # REPLACE: "use X instead of Y"
        (re.compile(r"\buse\s+(?P<new>\w+)\s+instead\s+of\s+(?P<old>\w+)", re.I),
         lambda m: UpdateIntent(type="replace", target=None, old_value=m["old"], new_value=m["new"], confidence=0.9)),

        # SCOPE_NARROW: "focus on X only" / "only X" / "X only" (Cairo, etc.)
        (re.compile(r"\b(?:focus|restrict|limit)\s+on\s+(?P<v>[\w\s]+?)(?:\s+only|\.|$)", re.I),
         lambda m: UpdateIntent(type="scope_narrow", target="geo", new_value=m["v"].strip(), confidence=0.9)),
        (re.compile(r"\bonly\s+(?P<v>[\w\s]+?)(?:[,.\n]|$)", re.I),
         lambda m: UpdateIntent(type="scope_narrow", target="geo", new_value=m["v"].strip(), confidence=0.75)),

        # SCOPE_EXPAND: "also consider X" / "include X" / "add X"
        (re.compile(r"\balso\s+(?:consider|include|cover|add)\s+(?P<v>[\w\s]+?)(?:[,.\n]|$)", re.I),
         lambda m: UpdateIntent(type="scope_expand", target="geo", new_value=m["v"].strip(), confidence=0.85)),

        # REMOVE: "drop the X" / "remove X"
        (re.compile(r"\b(?:drop|remove|cut|skip)\s+(?:the\s+)?(?P<v>[\w\s]+?)(?:[,.\n]|$)", re.I),
         lambda m: UpdateIntent(type="remove", value=m["v"].strip(), confidence=0.8)),

        # CORRECTION: "actually use X" / "no, use X"
        (re.compile(r"\b(?:actually|no,?)\s+use\s+(?P<v>[\w\s]+?)(?:[,.\n]|$)", re.I),
         lambda m: UpdateIntent(type="correction", new_value=m["v"].strip(), confidence=0.8)),

        # REPRIORITIZE: "do X first" / "X first"
        (re.compile(r"\b(?:do|make)\s+(?P<v>[\w\s]+?)\s+first\b", re.I),
         lambda m: UpdateIntent(type="reprioritize", value=m["v"].strip(), confidence=0.8)),

        # CLARIFY: "what do you mean by X" / "explain X"
        (re.compile(r"\b(?:what\s+do\s+you\s+mean|explain)\s+(?:by\s+)?(?P<v>[\w\s]+?)\??$", re.I),
         lambda m: UpdateIntent(type="clarify", target="term", value=m["v"].strip(), confidence=0.85)),

        # ABORT: "stop" / "restart" / "start over"
        (re.compile(r"\b(?:stop|restart|start\s+over|abort|cancel)\b", re.I),
         lambda m: UpdateIntent(type="abort", confidence=0.95)),
    ]

    def parse(self, user_text: str) -> UpdateIntent:
        text = user_text.strip()
        for pattern, builder in self.RULES:
            m = pattern.search(text)
            if m:
                return builder(m)
        # Default: treat unknown as a free-form correction
        return UpdateIntent(
            type="correction",
            new_value=text,
            confidence=0.4,
        )
