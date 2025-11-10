"""Unit tests for provider rotation helpers."""

import unittest
from unittest import mock

import llm_provider_rotation as rotation
from llm_provider_rotation import ProviderCandidate, RotatingLLM, _parse_rotation_entries


class TestLLMProviderRotation(unittest.TestCase):
    def test_codex_provider_marked_tool_free(self) -> None:
        entries = _parse_rotation_entries(
            raw_value="codex|gpt-5|https://ccproxy.emottet.com/codex/v1|key=abc",
            default_model="openai/claude-sonnet-4.5",
            default_base="https://primary.example/v1",
            default_key="fallback",
            source_key="LLM_PROVIDER_ROTATION",
        )

        self.assertTrue(entries)
        self.assertTrue(entries[0].tool_free_only)

    def test_rotating_llm_skips_tool_free_provider_when_tools(self) -> None:
        call_log: list[str] = []

        class DummyLLM:
            def __init__(self, model: str, api_base: str, api_key: str):
                self.model = model
                self.base_url = api_base
                self.api_key = api_key
                self.provider = None
                self.temperature = 0

            def call(self, **_: object) -> dict[str, str]:
                call_log.append(self.model)
                return {"model": self.model}

        providers = [
            ProviderCandidate(
                label="codex",
                model="openai/gpt-5",
                api_base="https://ccproxy.emottet.com/codex/v1",
                api_key="key-codex",
                disable_system_prompt=True,
                tool_free_only=True,
            ),
            ProviderCandidate(
                label="copilot",
                model="claude-sonnet-4.5",
                api_base="https://copilot.example/v1",
                api_key="key-copilot",
                disable_system_prompt=False,
                tool_free_only=False,
            ),
        ]

        with mock.patch.object(rotation, "LLM", DummyLLM):
            llm = RotatingLLM("TestAgent", providers)
            response = llm.call(messages=[], tools=[{"name": "demo"}])

        self.assertEqual(response, {"model": "claude-sonnet-4.5"})
        self.assertEqual(call_log, ["claude-sonnet-4.5"])


if __name__ == "__main__":
    unittest.main()
