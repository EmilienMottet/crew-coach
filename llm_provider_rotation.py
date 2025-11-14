"""Provider rotation helpers for CrewAI LLM instances."""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from crewai import LLM
from crewai.agent.core import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel


RATE_LIMIT_KEYWORDS = ("rate limit", "quota", "429")

# Codex endpoint requires system prompts to be stripped and merged into user message
# This is specific to the /codex/v1 endpoint optimization for code generation
PROMPTLESS_ENDPOINT_HINTS = ("codex",)
PROMPTLESS_MODEL_HINTS: tuple = ()  # No model-level restrictions on our endpoints

# IMPORTANT: ALL our endpoints support function calling (tools/MCP):
#   ✅ /copilot/v1  - supports tools + system prompts
#   ✅ /codex/v1    - supports tools (strips system prompts)
#   ✅ /claude/v1   - supports tools + system prompts
#
# Available models ALL support function calling:
#   - GPT family: gpt-5, gpt-5-mini, gpt-5-codex, gpt-4.1, gpt-4o, etc.
#   - Claude family: claude-sonnet-4.5, claude-haiku-4.5, claude-3.5-sonnet, etc.
#   - Other: gemini-2.5-pro, grok-code-fast-1
#
# No restrictions needed - all combinations work with CrewAI tools/MCP
TOOL_FREE_ENDPOINT_HINTS: tuple = ()  # No endpoint blocks tools
TOOL_FREE_MODEL_HINTS: tuple = ()     # No model blocks tools


@dataclass(frozen=True)
class ProviderCandidate:
    """Represents a single provider/model endpoint in the rotation chain."""

    label: str
    model: str
    api_base: str
    api_key: str
    disable_system_prompt: bool = False
    tool_free_only: bool = False


def _requires_promptless_mode(api_base: str, model_name: str) -> bool:
    """Return True when an endpoint/model rejects system prompts."""

    base = (api_base or "").lower()
    model = (model_name or "").lower()

    if any(keyword in base for keyword in PROMPTLESS_ENDPOINT_HINTS):
        return True

    if any(keyword in model for keyword in PROMPTLESS_MODEL_HINTS):
        return True

    return False


def _requires_tool_free_context(api_base: str, model_name: str) -> bool:
    """Return True when the provider cannot execute tool/function calls."""

    base = (api_base or "").lower()
    model = (model_name or "").lower()

    if not any(keyword in base for keyword in TOOL_FREE_ENDPOINT_HINTS):
        return False

    return any(keyword in model for keyword in TOOL_FREE_MODEL_HINTS)


def create_llm_with_rotation(
    *, agent_name: str, model_name: str, api_base: str, api_key: str
) -> BaseLLM:
    """Create an LLM wrapped with provider rotation logic when enabled."""

    normalized_model = _normalize_model_name(model_name, api_base)
    provider_chain = _build_provider_chain(
        agent_name=agent_name,
        normalized_model=normalized_model,
        api_base=api_base,
        api_key=api_key,
    )

    if len(provider_chain) == 1 and not provider_chain[0].disable_system_prompt:
        provider = provider_chain[0]
        return LLM(model=provider.model, api_base=provider.api_base, api_key=provider.api_key)

    labels = ", ".join(candidate.label for candidate in provider_chain)
    print(
        f"\n🔁 {agent_name or 'LLM'} provider chain: {labels}\n",
        file=sys.stderr,
    )
    return RotatingLLM(agent_name or "LLM", provider_chain)


def _build_provider_chain(
    *, agent_name: str, normalized_model: str, api_base: str, api_key: str
) -> List[ProviderCandidate]:
    """Assemble the ordered provider list for a given agent."""

    rotation_enabled = os.getenv("ENABLE_LLM_PROVIDER_ROTATION", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    primary = ProviderCandidate(
        label=_provider_label(agent_name, api_base, suffix="primary"),
        model=normalized_model,
        api_base=api_base,
        api_key=api_key,
        disable_system_prompt=_requires_promptless_mode(api_base, normalized_model),
        tool_free_only=_requires_tool_free_context(api_base, normalized_model),
    )

    if not rotation_enabled:
        return [primary]

    candidates: List[ProviderCandidate] = [primary]
    env_keys = []
    if agent_name:
        env_keys.append(f"{agent_name}_PROVIDER_ROTATION")
    env_keys.append("LLM_PROVIDER_ROTATION")

    for env_key in env_keys:
        raw_value = os.getenv(env_key, "").strip()
        if not raw_value:
            continue
        candidates.extend(
            _parse_rotation_entries(
                raw_value=raw_value,
                default_model=normalized_model,
                default_base=api_base,
                default_key=api_key,
                source_key=env_key,
            )
        )

    candidates = _deduplicate_candidates(candidates)
    candidates = _randomize_provider_order(candidates)
    candidates = _ensure_copilot_fallback(candidates, api_key)

    for candidate in candidates:
        if candidate.disable_system_prompt:
            print(
                f"ℹ️  {agent_name or 'LLM'} provider {candidate.label} strips system prompts per endpoint requirements",
                file=sys.stderr,
            )

    return candidates


def _parse_rotation_entries(
    *,
    raw_value: str,
    default_model: str,
    default_base: str,
    default_key: str,
    source_key: str,
) -> List[ProviderCandidate]:
    """Parse rotation entries defined in an environment variable."""

    entries: List[ProviderCandidate] = []
    for chunk in raw_value.split(";"):
        token = chunk.strip()
        if not token:
            continue

        parts = [part.strip() for part in token.split("|")]
        while len(parts) < 4:
            parts.append("")

        label_part, model_part, base_part, key_part = parts[:4]
        resolved_base = base_part or default_base
        resolved_model = (
            default_model if _uses_default_marker(model_part) else _normalize_model_name(model_part, resolved_base)
        )
        resolved_key = _resolve_api_key_hint(key_part, default_key)

        if not resolved_key:
            print(
                f"⚠️  Skipping provider '{token}' from {source_key}: no API key available",
                file=sys.stderr,
            )
            continue

        label = label_part or _provider_label(source_key, resolved_base)
        entries.append(
            ProviderCandidate(
                label=label,
                model=resolved_model,
                api_base=resolved_base,
                api_key=resolved_key,
                disable_system_prompt=_requires_promptless_mode(resolved_base, resolved_model),
                tool_free_only=_requires_tool_free_context(resolved_base, resolved_model),
            )
        )

    return entries


def _resolve_api_key_hint(hint: str, fallback_key: str) -> str:
    if not hint:
        return fallback_key

    normalized = hint.strip()
    if not normalized:
        return fallback_key

    if normalized.lower().startswith("env:"):
        env_name = normalized.split(":", 1)[1].strip()
        return os.getenv(env_name, fallback_key)

    if normalized.lower().startswith("key="):
        return normalized.split("=", 1)[1]

    return os.getenv(normalized, fallback_key)


def _ensure_copilot_fallback(
    candidates: Sequence[ProviderCandidate], default_key: str
) -> List[ProviderCandidate]:
    """Append the mandatory Copilot fallback unless already present."""

    copilot_base = os.getenv("COPILOT_API_BASE", "https://ccproxy.emottet.com/copilot/v1")
    copilot_key = os.getenv("COPILOT_API_KEY", default_key)
    copilot_model = _normalize_model_name(os.getenv("COPILOT_FALLBACK_MODEL", "gpt-5-mini"), copilot_base)

    fallback = ProviderCandidate(
        label="copilot-fallback",
        model=copilot_model,
        api_base=copilot_base,
        api_key=copilot_key,
        disable_system_prompt=_requires_promptless_mode(copilot_base, copilot_model),
        tool_free_only=_requires_tool_free_context(copilot_base, copilot_model),
    )

    extended = list(candidates) + [fallback]
    return _deduplicate_candidates(extended)


def _deduplicate_candidates(
    candidates: Sequence[ProviderCandidate],
) -> List[ProviderCandidate]:
    """Remove duplicate provider definitions while preserving order."""

    unique: List[ProviderCandidate] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.model,
            candidate.api_base,
            candidate.api_key,
            candidate.disable_system_prompt,
            candidate.tool_free_only,
        )
        if key in seen:
            continue
        unique.append(candidate)
        seen.add(key)
    return unique


def _randomize_provider_order(
    candidates: Sequence[ProviderCandidate],
) -> List[ProviderCandidate]:
    """Shuffle provider order to distribute load across endpoints."""

    randomized = list(candidates)
    if len(randomized) <= 1:
        return randomized

    random.shuffle(randomized)
    return randomized


def _normalize_model_name(model_name: str, api_base: str = "") -> str:
    """Normalize model name based on the endpoint requirements."""
    cleaned = (model_name or "").strip()
    if not cleaned:
        raise ValueError("Model name cannot be empty")

    base_lower = (api_base or "").lower()

    # Debug logging
    print(f"🔍 Normalizing model: '{cleaned}' for endpoint: '{api_base}'", file=sys.stderr)

    # Detect endpoint type from URL
    if "/copilot/v1" in base_lower:
        # Copilot endpoint: use short names
        normalized = _normalize_for_copilot(cleaned)
        print(f"   ✓ Copilot endpoint detected → '{normalized}'", file=sys.stderr)
        return normalized
    elif "/codex/v1" in base_lower:
        # Codex endpoint: use short names (gpt-5, gpt-5-codex)
        normalized = _normalize_for_codex(cleaned)
        print(f"   ✓ Codex endpoint detected → '{normalized}'", file=sys.stderr)
        return normalized
    elif "/claude/v1" in base_lower:
        # Claude endpoint: use full versioned names
        normalized = _normalize_for_claude_endpoint(cleaned)
        print(f"   ✓ Claude endpoint detected → '{normalized}'", file=sys.stderr)
        return normalized
    elif "ccproxy" in base_lower or "ghcopilot" in base_lower:
        # Generic ccproxy/ghcopilot: assume copilot behavior
        normalized = _normalize_for_copilot(cleaned)
        print(f"   ✓ Generic ccproxy/ghcopilot endpoint detected → '{normalized}'", file=sys.stderr)
        return normalized

    # Default: add openai/ prefix if needed
    print(f"   ✓ Generic OpenAI endpoint detected", file=sys.stderr)
    if "/" not in cleaned:
        return f"openai/{cleaned}"
    return cleaned


def _normalize_for_copilot(model_name: str) -> str:
    """Normalize model names for the copilot endpoint."""
    # Copilot accepts short names like: gpt-5, gpt-5-mini, claude-sonnet-4.5
    # Map common variations to canonical short names
    model_lower = model_name.lower()

    # GPT models
    if "gpt-5-codex" in model_lower:
        return "gpt-5-codex"
    if "gpt-5-mini" in model_lower:
        return "gpt-5-mini"
    if "gpt-5" in model_lower:
        return "gpt-5"
    if "gpt-4.1" in model_lower or "gpt-41" in model_lower:
        return "gpt-4.1"
    if "gpt-4o" in model_lower:
        return "gpt-4o"

    # Claude models - map to short names
    if "claude-sonnet-4.5" in model_lower or "claude-sonnet-4-5" in model_lower:
        return "claude-sonnet-4.5"
    if "claude-haiku-4.5" in model_lower or "claude-haiku-4-5" in model_lower:
        return "claude-haiku-4.5"
    if "claude-sonnet-4" in model_lower and "4.5" not in model_lower and "4-5" not in model_lower:
        return "claude-sonnet-4"
    if "claude-3.5-sonnet" in model_lower or "claude-35-sonnet" in model_lower:
        return "claude-3.5-sonnet"

    # Gemini models
    if "gemini-2.5-pro" in model_lower or "gemini-25-pro" in model_lower:
        return "gemini-2.5-pro"

    # Grok models
    if "grok-code-fast" in model_lower:
        return "grok-code-fast-1"

    return model_name


def _normalize_for_codex(model_name: str) -> str:
    """Normalize model names for the codex endpoint."""
    # Codex only accepts: gpt-5, gpt-5-codex
    model_lower = model_name.lower()

    if "gpt-5-codex" in model_lower:
        return "gpt-5-codex"
    if "gpt-5" in model_lower:
        return "gpt-5"

    # Default to gpt-5 for unrecognized models
    return "gpt-5"


def _normalize_for_claude_endpoint(model_name: str) -> str:
    """Normalize model names for the claude endpoint (requires full versioned names)."""
    model_lower = model_name.lower()

    # Map short names to full versioned names
    claude_version_map = {
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "claude-opus-4.1": "claude-opus-4-1-20250805",
        "claude-opus-4-1": "claude-opus-4-1-20250805",
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    # Check if the model name is already a full versioned name
    if model_lower.endswith(tuple(m[-8:] for m in claude_version_map.values() if len(m) > 8)):
        return model_name

    # Find matching short name and return full version
    for short_name, full_name in claude_version_map.items():
        if short_name in model_lower:
            return full_name

    # If no match, return as-is (might be already a full version)
    return model_name


def _uses_default_marker(value: str) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return lowered in {"", "same", "default", "~"}


def _provider_label(prefix: str, base_url: str, suffix: str | None = None) -> str:
    host = base_url.split("//")[-1]
    host = host.rstrip("/")
    if suffix:
        return f"{prefix or 'primary'}:{suffix}@{host}"
    return f"{prefix or 'fallback'}@{host}"


class RotatingLLM(BaseLLM):
    """BaseLLM wrapper that retries calls across multiple providers on 429 errors."""

    def __init__(self, agent_name: str, providers: Sequence[ProviderCandidate]) -> None:
        if not providers:
            raise ValueError("At least one provider is required for rotation")

        self._agent_name = agent_name or "LLM"
        self._providers = list(providers)
        self._llms: List[LLM | None] = [None] * len(self._providers)
        self._last_success_index = 0

        primary_llm = self._instantiate_llm(self._providers[0])
        self._llms[0] = primary_llm

        super().__init__(
            model=primary_llm.model,
            temperature=getattr(primary_llm, "temperature", None),
            api_key=getattr(primary_llm, "api_key", None),
            base_url=getattr(primary_llm, "base_url", None),
            provider=getattr(primary_llm, "provider", None),
        )

    def call(
        self,
        messages: str | List[LLMMessage],
        tools: List[Dict[str, BaseTool]] | None = None,
        callbacks: List[Any] | None = None,
        available_functions: Dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        base_messages = messages
        
        # CRITICAL FIX: CrewAI doesn't pass tools parameter, extract from agent
        if not tools and from_agent:
            agent_tools = getattr(from_agent, 'tools', None)
            if agent_tools:
                # Build available_functions dict from agent tools
                if not available_functions:
                    available_functions = {
                        getattr(tool, 'name', str(tool)): tool 
                        for tool in agent_tools
                    }
                    print(
                        f"   ℹ️  Extracted {len(available_functions)} tools from agent\n",
                        file=sys.stderr,
                    )
                
                # ALWAYS try to convert to tools format for function calling
                try:
                    from crewai.tools.base_tool import BaseTool as CrewBaseTool
                    import json
                    
                    def clean_schema(schema):
                        """Recursively clean JSON schema to be Claude/OpenAI compatible"""
                        if not isinstance(schema, dict):
                            return schema
                        
                        cleaned = {}
                        for key, value in schema.items():
                            # Skip unsupported fields
                            if key in ['title', 'definitions', '$defs', 'allOf', 'anyOf', 'oneOf']:
                                continue
                            
                            # Skip None values
                            if value is None:
                                continue
                            
                            # Skip empty lists
                            if isinstance(value, list) and len(value) == 0:
                                continue
                            
                            # Skip empty dicts (except for 'properties' and 'required' which are allowed)
                            if isinstance(value, dict) and len(value) == 0 and key not in ['properties', 'required']:
                                continue
                            
                            # Recursively clean nested dicts
                            if isinstance(value, dict):
                                cleaned[key] = clean_schema(value)
                            # Recursively clean dicts in lists
                            elif isinstance(value, list):
                                cleaned[key] = [clean_schema(item) if isinstance(item, dict) else item for item in value]
                            # Keep other values as-is
                            else:
                                cleaned[key] = value
                        
                        return cleaned
                    
                    tools = []
                    for tool in agent_tools:
                        tool_def = None
                        
                        # PRIORITY 1: Use args_schema if available (most reliable)
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            try:
                                schema = tool.args_schema
                                if hasattr(schema, 'model_json_schema'):
                                    json_schema = schema.model_json_schema()
                                    # Deep clean the schema
                                    json_schema = clean_schema(json_schema)
                                    
                                    # Ensure required fields exist after cleaning
                                    if 'type' not in json_schema:
                                        json_schema['type'] = 'object'
                                    if 'properties' not in json_schema:
                                        json_schema['properties'] = {}
                                    
                                    tool_def = {
                                        "type": "function",
                                        "function": {
                                            "name": getattr(tool, 'name', str(tool)),
                                            "description": getattr(tool, 'description', ''),
                                            "parameters": json_schema
                                        }
                                    }
                            except Exception as e:
                                print(f"⚠️  Failed to use args_schema for {getattr(tool, 'name', 'unknown')}: {e}", file=sys.stderr)
                        
                        # FALLBACK: Try to_function if args_schema failed
                        if not tool_def and hasattr(tool, 'to_function') and callable(tool.to_function):
                            try:
                                func_def = tool.to_function()
                                
                                # CRITICAL: Clean up schema for Claude/OpenAI compatibility
                                if 'function' in func_def and 'parameters' in func_def['function']:
                                    params = func_def['function']['parameters']
                                    # Deep clean the entire schema
                                    func_def['function']['parameters'] = clean_schema(params)
                                    
                                    # Ensure required fields exist after cleaning
                                    if 'type' not in func_def['function']['parameters']:
                                        func_def['function']['parameters']['type'] = 'object'
                                    if 'properties' not in func_def['function']['parameters']:
                                        func_def['function']['parameters']['properties'] = {}
                                
                                tool_def = func_def
                            except Exception as e:
                                print(f"⚠️  Failed to use to_function for {getattr(tool, 'name', 'unknown')}: {e}", file=sys.stderr)
                        
                        if tool_def:
                            tools.append(tool_def)
                    
                    if tools:
                        print(
                            f"   ℹ️  Converted {len(tools)} tools to function calling format\n",
                            file=sys.stderr,
                        )
                        # Debug: print first tool schema
                        if tools:
                            print(
                                f"   🔍 First tool schema sample:\n"
                                f"      {json.dumps(tools[0], indent=2)[:500]}...\n",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            f"   ⚠️  Failed to convert any of {len(agent_tools)} tools (no to_function or args_schema)\n",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(f"   ⚠️  Failed to convert tools: {e}\n", file=sys.stderr)
        
        has_tools = bool(tools) or bool(available_functions)
        attempted_provider = False

        # Log agent call with tool info
        tool_count = len(tools) if tools else (len(available_functions) if available_functions else 0)
        print(
            f"🤖 {self._agent_name} calling LLM (has_tools={has_tools}, tool_count={tool_count})",
            file=sys.stderr,
        )

        for index, provider in enumerate(self._providers):
            if has_tools and provider.tool_free_only:
                print(
                    f"\n⏭️  {self._agent_name}: SKIPPING {provider.label} ({provider.model})\n"
                    f"   Reason: Agent requires tool calling but this endpoint does not support it\n"
                    f"   (Codex endpoints cannot execute MCP tool calls)\n",
                    file=sys.stderr,
                )
                continue

            attempted_provider = True
            llm = self._ensure_llm(index)
            try:
                if index > 0:
                    print(
                        f"\n🔁 {self._agent_name}: retrying with {provider.label} ({provider.model})\n",
                        file=sys.stderr,
                    )

                effective_messages = _prepare_messages_for_provider(
                    base_messages,
                    provider.disable_system_prompt,
                )

                # CRITICAL FIX: When tools are present, call litellm.completion directly
                # CrewAI's LLM.call() ignores the tools parameter
                if tools and len(tools) > 0:
                    print(
                        f"   🚀 Calling litellm.completion directly with {len(tools)} tools\n"
                        f"      Model: {provider.model}\n"
                        f"      API Base: {provider.api_base}\n",
                        file=sys.stderr
                    )
                    
                    # Ensure messages are in correct format
                    if isinstance(effective_messages, str):
                        formatted_messages = [{"role": "user", "content": effective_messages}]
                    else:
                        formatted_messages = effective_messages
                    
                    # Prepare API key - use Bearer token format (standard for OpenAI-compatible APIs)
                    api_key_str = str(provider.api_key) if provider.api_key else ""
                    
                    # Determine custom_llm_provider based on API base and model
                    # All our endpoints are OpenAI-compatible
                    custom_provider = "openai"
                    
                    # LiteLLM needs provider prefix for some models
                    model_for_litellm = provider.model
                    if "claude" in provider.model.lower():
                        # Don't add prefix - our proxy handles it
                        pass
                    
                    # Call litellm directly with the necessary parameters
                    import litellm
                    try:
                        litellm_response = litellm.completion(
                            model=model_for_litellm,
                            messages=formatted_messages,
                            tools=tools,
                            api_base=provider.api_base,
                            api_key=api_key_str,
                            custom_llm_provider=custom_provider,
                            drop_params=True,
                            request_timeout=180,  # 180 seconds timeout (increased from 60)
                            max_tokens=16000,  # Allow longer responses for complex JSON outputs
                        )
                            
                    except Exception as e:
                        print(
                            f"   ❌ litellm.completion failed: {e}\n"
                            f"      Falling back to CrewAI LLM.call()\n",
                            file=sys.stderr
                        )
                        # Fall back to CrewAI's normal flow (without tools)
                        response = llm.call(
                            messages=effective_messages,
                            callbacks=callbacks,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )
                        self._last_success_index = index
                        self._sync_metadata(llm)
                        print(
                            f"✅ {self._agent_name}: successfully used {provider.label} ({provider.model}) [fallback]\n",
                            file=sys.stderr,
                        )
                        return response
                    
                    # Check for tool calls
                    choice = litellm_response.choices[0]
                    if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        print(
                            f"   ✅ LLM returned {len(choice.message.tool_calls)} tool calls!\n",
                            file=sys.stderr
                        )
                        
                        # Execute each tool call
                        tool_results = []
                        for tool_call in choice.message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args_str = tool_call.function.arguments
                            
                            print(
                                f"   🔧 Executing tool: {tool_name}\n"
                                f"      Arguments: {tool_args_str[:200]}...\n",
                                file=sys.stderr
                            )
                            
                            # Find the tool in available_functions
                            if available_functions and tool_name in available_functions:
                                try:
                                    # Parse arguments
                                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                    
                                    # Get the tool callable
                                    tool_func = available_functions[tool_name]
                                    
                                    # Execute the tool
                                    if hasattr(tool_func, '_run'):
                                        tool_result = tool_func._run(tool_args)
                                    elif hasattr(tool_func, 'run'):
                                        tool_result = tool_func.run(tool_args)
                                    elif callable(tool_func):
                                        tool_result = tool_func(**tool_args) if isinstance(tool_args, dict) else tool_func(tool_args)
                                    else:
                                        tool_result = {"error": f"Tool {tool_name} is not callable"}
                                    
                                    print(
                                        f"   ✅ Tool {tool_name} executed successfully\n"
                                        f"      Result: {str(tool_result)[:200]}...\n",
                                        file=sys.stderr
                                    )
                                    
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                                    })
                                except Exception as tool_error:
                                    print(
                                        f"   ❌ Tool {tool_name} failed: {tool_error}\n",
                                        file=sys.stderr
                                    )
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps({"error": str(tool_error)})
                                    })
                            else:
                                print(
                                    f"   ⚠️  Tool {tool_name} not found in available_functions\n",
                                    file=sys.stderr
                                )
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps({"error": f"Tool {tool_name} not available"})
                                })
                        
                        # Add tool results to conversation and call LLM again
                        if tool_results:
                            print(
                                f"   🔁 Sending {len(tool_results)} tool results back to LLM...\n",
                                file=sys.stderr
                            )
                            
                            # Build new messages with tool results
                            new_messages = formatted_messages + [
                                {
                                    "role": "assistant",
                                    "content": choice.message.content,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        }
                                        for tc in choice.message.tool_calls
                                    ]
                                }
                            ] + tool_results
                            
                            # Call LLM again with tool results
                            final_response = litellm.completion(
                                model=model_for_litellm,
                                messages=new_messages,
                                tools=tools,
                                api_base=provider.api_base,
                                api_key=api_key_str,
                                custom_llm_provider=custom_provider,
                                drop_params=True,
                                request_timeout=240,  # Longer timeout for second call with tool results (increased from 90)
                                max_tokens=16000,  # Allow longer responses for complex JSON outputs
                            )
                            
                            print(
                                f"   ✅ LLM generated final response with tool results\n",
                                file=sys.stderr
                            )
                            
                            response = final_response.choices[0].message.content or str(final_response.choices[0].message)
                        else:
                            response = choice.message.content or str(choice.message)
                    else:
                        response = choice.message.content or str(choice.message)
                else:
                    # No tools, use normal CrewAI flow
                    response = llm.call(
                        messages=effective_messages,
                        tools=tools,
                        callbacks=callbacks,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_model=response_model,
                    )

                self._last_success_index = index
                self._sync_metadata(llm)

                # Log which provider was successfully used
                print(
                    f"✅ {self._agent_name}: successfully used {provider.label} ({provider.model})\n",
                    file=sys.stderr,
                )

                return response

            except Exception as exc:  # noqa: BLE001
                if not _is_rate_limit_error(exc):
                    raise

                last_error = exc
                if index == len(self._providers) - 1:
                    break

                print(
                    f"⚠️  {self._agent_name}: provider {provider.label} hit quota, rotating...",
                    file=sys.stderr,
                )

        if last_error:
            raise last_error
        if not attempted_provider:
            raise RuntimeError(
                f"{self._agent_name}: no compatible provider available for tool-enabled calls"
            )
        raise RuntimeError("Provider rotation exhausted without capturing exception")

    def _ensure_llm(self, index: int) -> LLM:
        llm = self._llms[index]
        if llm is None:
            llm = self._instantiate_llm(self._providers[index])
            self._llms[index] = llm
        return llm

    @staticmethod
    def _instantiate_llm(target: ProviderCandidate) -> LLM:
        return LLM(model=target.model, api_base=target.api_base, api_key=target.api_key)

    def _sync_metadata(self, llm: LLM) -> None:
        self.model = getattr(llm, "model", self.model)
        self.api_key = getattr(llm, "api_key", self.api_key)
        self.base_url = getattr(llm, "base_url", self.base_url)

    def __getattr__(self, item: str) -> Any:
        current = self._llms[self._last_success_index]
        if current is not None and hasattr(current, item):
            return getattr(current, item)
        raise AttributeError(item)

    @staticmethod
    def set_callbacks(callbacks: List[Any]) -> None:  # pragma: no cover - delegation helper
        LLM.set_callbacks(callbacks)

    @staticmethod
    def set_env_callbacks() -> None:  # pragma: no cover - delegation helper
        LLM.set_env_callbacks()


def _is_rate_limit_error(exc: Exception) -> bool:
    exc_name = exc.__class__.__name__
    if exc_name in {"RateLimitError", "RateLimitException"}:
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status_code == 429:
        return True

    try:
        message = str(exc).lower()
    except Exception:  # pragma: no cover - defensive
        message = ""

    return any(keyword in message for keyword in RATE_LIMIT_KEYWORDS)


def _prepare_messages_for_provider(
    messages: str | List[LLMMessage], disable_system_prompt: bool
) -> str | List[LLMMessage]:
    if not disable_system_prompt or isinstance(messages, str):
        return messages
    return _merge_system_prompt_into_user(messages)


def _merge_system_prompt_into_user(messages: List[LLMMessage]) -> List[LLMMessage]:
    sanitized: List[LLMMessage] = []
    system_chunks: List[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            stripped = content.strip()
            if stripped:
                system_chunks.append(stripped)
            continue

        sanitized.append({"role": role, "content": content})

    if not system_chunks:
        return sanitized

    merged_prompt = "\n\n".join(system_chunks).strip()
    if not merged_prompt:
        return sanitized

    merged_entry: LLMMessage = {"role": "user", "content": merged_prompt}
    if sanitized and sanitized[0]["role"] == "user":
        first_content = sanitized[0]["content"].strip()
        combined_content = f"{merged_prompt}\n\n{first_content}".strip()
        sanitized[0] = {"role": "user", "content": combined_content}
        return sanitized

    return [merged_entry, *sanitized]
