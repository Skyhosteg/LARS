"""
llm.py — Thin LLM client with two backends:

  1. OpenAILLM  — calls gpt-4o-mini (or any chat-completions model)
  2. MockLLM    — returns deterministic JSON for testing without API keys

Both expose a single method:

    complete_json(system: str, user: str, schema: type[BaseModel]) -> BaseModel

The schema argument is a Pydantic model class. The client is responsible
for asking the model to emit JSON matching that schema and parsing it
back into a validated Pydantic instance.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Base interface. Subclass and implement complete_json."""

    def complete_json(self, system: str, user: str, schema: Type[T]) -> T:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------------- #


class OpenAILLM(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "No OpenAI API key. Set OPENAI_API_KEY env var or pass api_key=...\n"
                "Or use MockLLM for offline testing."
            )
        # Lazy import — openai is optional for mock-only users
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key)

    def complete_json(self, system: str, user: str, schema: Type[T]) -> T:
        schema_name = schema.__name__
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        # Append the schema spec to the system prompt. We don't rely on
        # response_format=json_schema here to keep this portable across
        # any chat-completions-compatible endpoint (Together, vLLM, etc.)
        full_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"Schema name: {schema_name}\n"
            f"Schema:\n{schema_json}\n"
            f"Return ONLY the JSON object. No prose, no markdown fences."
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {e}") from e

        raw = resp.choices[0].message.content or ""

        # Strip code fences if the model slipped them in
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Model did not return valid JSON. Raw:\n{raw[:500]}") from e

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMError(f"JSON did not match schema {schema_name}: {e}\nData: {data}") from e


# --------------------------------------------------------------------------- #
# Mock backend — deterministic, no network
# --------------------------------------------------------------------------- #


class MockLLM(LLMClient):
    """
    Deterministic LLM stub. Returns canned JSON for known inputs.

    Useful for:
      - Running the demo without API keys
      - Unit tests
      - Reproducible benchmarks
    """

    def __init__(self, fixtures: dict[str, dict] | None = None):
        """
        Args:
            fixtures: optional dict of {substring: json_response}.
                      When the user prompt contains the substring, that
                      fixture is returned. Otherwise a generic state
                      fixture is returned.
        """
        self.fixtures = fixtures or {}
        self._default = {
            "goal": "[mock] example goal",
            "steps_completed": [
                {"step_id": 1, "description": "[mock] step 1 done", "status": "completed", "dependencies": []},
                {"step_id": 2, "description": "[mock] step 2 done", "status": "completed", "dependencies": [1]},
            ],
            "steps_pending": [
                {"step_id": 3, "description": "[mock] step 3 pending", "status": "pending", "dependencies": [2]},
            ],
            "assumptions": ["[mock] assumption 1"],
            "decisions": [{"decision": "[mock] decision", "rationale": "[mock] rationale"}],
            "confidence": 0.75,
        }
        # Method-specific fixtures — used when prompt contains method markers.
        # These simulate what each baseline would return in production.
        self._restart_state = {
            "goal": "[mock] example goal",
            "steps_completed": [
                {"step_id": 1, "description": "[restart] regenerated step 1 based on user input", "status": "completed", "dependencies": []},
                {"step_id": 2, "description": "[restart] regenerated step 2 ignoring original", "status": "completed", "dependencies": [1]},
            ],
            "steps_pending": [
                {"step_id": 3, "description": "[restart] new pending from scratch", "status": "pending", "dependencies": [2]},
            ],
            "assumptions": ["[restart] fresh assumption from user redirect"],
            "decisions": [{"decision": "[restart] follow user instruction", "rationale": "user redirected"}],
            "confidence": 0.6,
        }
        self._langgraph_state = {
            "goal": "[mock] example goal",
            "steps_completed": [
                {"step_id": 1, "description": "[langgraph] original step 1 kept", "status": "completed", "dependencies": []},
                {"step_id": 2, "description": "[langgraph] original step 2 kept", "status": "completed", "dependencies": [1]},
                {"step_id": 3, "description": "[langgraph] new step appended from interrupt", "status": "completed", "dependencies": []},
            ],
            "steps_pending": [],
            "assumptions": ["[langgraph] original kept", "[langgraph] added: user-interrupt constraint"],
            "decisions": [
                {"decision": "[langgraph] original kept", "rationale": "from checkpoint"},
                {"decision": "[langgraph] append user instruction", "rationale": "no merge, just append"},
            ],
            "confidence": 0.55,
        }

    def complete_json(self, system: str, user: str, schema: Type[T]) -> T:
        # Method-aware matching: if the prompt contains markers
        # for a specific method, use the method-specific fixture.
        for needle, payload in self.fixtures.items():
            if needle in user:
                return schema.model_validate(payload)
        # If the prompt mentions the user's interrupt, it means
        # we're in restart_from_scratch or langgraph_checkpoint mode.
        if "USER INSTRUCTION" in user:
            return schema.model_validate(self._restart_state)
        if "USER INTERRUPT AT CHECKPOINT" in user:
            return schema.model_validate(self._langgraph_state)
        return schema.model_validate(self._default)


def default_client() -> LLMClient:
    """Pick the right LLM client based on env vars.

    Priority:
      1. OPENROUTER_API_KEY  → OpenRouterLLM
      2. OPENAI_API_KEY       → OpenAILLM
      3. (none)               → MockLLM
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterLLM()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    return MockLLM()


# --------------------------------------------------------------------------- #
# OpenRouter backend (multi-model proxy, OpenAI-compatible API)
# --------------------------------------------------------------------------- #


class OpenRouterLLM(LLMClient):
    """OpenRouter — a unified API for many LLMs (OpenAI, Anthropic, etc.).

    OpenRouter is mostly OpenAI-compatible, so we reuse the OpenAI client
    with a different base_url and an `Authorization: Bearer` header.

    Set the model via the `OPENROUTER_MODEL` env var, e.g.:
        openai/gpt-4o-mini
        anthropic/claude-3-haiku
        meta-llama/llama-3.1-8b-instruct
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY env var.\n"
                "Get a key at https://openrouter.ai/keys"
            )
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/Skyhosteg/LARS",
                "X-Title": "LARS - Live Adaptive Reasoning",
            },
        )

    def complete_json(self, system: str, user: str, schema: Type[T]) -> T:
        schema_name = schema.__name__
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        full_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"Schema name: {schema_name}\n"
            f"Schema:\n{schema_json}\n"
            f"Return ONLY the JSON object. No prose, no markdown fences."
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
        except Exception as e:
            raise LLMError(f"OpenRouter call failed: {e}") from e

        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Model did not return valid JSON. Raw:\n{raw[:500]}"
            ) from e

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMError(
                f"JSON did not match schema {schema_name}: {e}\nData: {data}"
            ) from e
