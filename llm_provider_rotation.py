"""Provider rotation helpers for CrewAI LLM instances."""

from __future__ import annotations

import os
import sys
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from crewai import LLM
from crewai.agent.core import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel


RATE_LIMIT_KEYWORDS = ("rate limit", "quota", "429", "token_expired", "no quota", "unknown provider", "404", "not found", "model not available")

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
# Model Categories (from ccproxy /models endpoint)
# All models use https://ccproxy.emottet.com/v1 endpoint
COMPLEX_MODELS = (
    "gemini-claude-sonnet-4-5-thinking",
    "gemini-3-pro-high",
    "gpt-5.1-high",
)

INTERMEDIATE_MODELS = (
    "claude-sonnet-4.5-copilot",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "gpt-5.1-medium",
    "gpt-5.1-codex-medium",
    "gemini-2.5-pro",
)

SIMPLE_MODELS = (
    "gpt-5.1-low",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4.5-copilot",
)

FALLBACK_MODELS = (
    "gpt-5-mini",
    "raptor-mini",
)

# Category cascade order (from highest to lowest tier)
CATEGORY_CASCADE = ["complex", "intermediate", "simple", "fallback"]

TOOL_FREE_ENDPOINT_HINTS: tuple = ()  # No endpoint blocks tools
TOOL_FREE_MODEL_HINTS: tuple = ()     # No model blocks tools


def get_models_for_category(category: str) -> tuple[str, ...]:
    """Return all models for a given category."""
    category = category.lower()
    if category == "complex":
        return COMPLEX_MODELS
    elif category == "intermediate":
        return INTERMEDIATE_MODELS
    elif category == "simple":
        return SIMPLE_MODELS
    elif category == "fallback":
        return FALLBACK_MODELS
    return FALLBACK_MODELS


def get_model_for_category(category: str) -> str:
    """Return a random available model for a given category."""
    models = get_models_for_category(category)
    return random.choice(models)


def build_category_cascade(starting_category: str, endpoint_url: str, api_key: str, agent_name: str = "") -> List[ProviderCandidate]:
    """Build a provider chain with category cascade (random selection + fallback to lower tiers).

    Logic:
    1. Start with requested category (e.g., "complex")
    2. Add ALL models from that category in random order (excluding blacklisted models)
    3. Cascade to next lower tier (complex → intermediate → simple → fallback)
    4. If fallback fails → crash (no recovery)

    Args:
        starting_category: One of "complex", "intermediate", "simple", "fallback"
        endpoint_url: API endpoint URL (typically https://ccproxy.emottet.com/v1)
        api_key: API key for authentication
        agent_name: Name of the agent (for agent-specific blacklists)

    Returns:
        List of ProviderCandidate in cascade order
    """
    starting_category = starting_category.lower()

    # Find starting position in cascade
    try:
        start_index = CATEGORY_CASCADE.index(starting_category)
    except ValueError:
        # Unknown category → default to fallback
        print(f"⚠️  Unknown category '{starting_category}', defaulting to 'fallback'", file=sys.stderr)
        start_index = CATEGORY_CASCADE.index("fallback")

    # Get blacklisted models from global and agent-specific sources
    global_blacklist = os.getenv("GLOBAL_BLACKLISTED_MODELS", "").split(",") if os.getenv("GLOBAL_BLACKLISTED_MODELS") else []
    agent_blacklist = os.getenv(f"{agent_name}_BLACKLISTED_MODELS", "").split(",") if agent_name and os.getenv(f"{agent_name}_BLACKLISTED_MODELS") else []

    # Combine and clean blacklists
    all_blacklisted = [m.strip() for m in global_blacklist + agent_blacklist if m.strip()]

    # Apply default blacklist for specific agents
    if agent_name == "HEXIS_ANALYSIS" and not agent_blacklist:
        default_blacklist = ["gpt-5", "gpt-5-codex"]
        all_blacklisted.extend(default_blacklist)

    if all_blacklisted:
        print(f"   🚫 Filtering blacklisted models: {', '.join(all_blacklisted[:3])}{'...' if len(all_blacklisted) > 3 else ''}", file=sys.stderr)

    providers = []

    # Build cascade from starting category to fallback
    for category in CATEGORY_CASCADE[start_index:]:
        models = list(get_models_for_category(category))

        # Filter out blacklisted models
        models = [m for m in models if m not in all_blacklisted]

        if not models:
            print(f"   ⚠️  All models in category '{category}' are blacklisted, skipping category", file=sys.stderr)
            continue

        # Randomize order within category for load balancing
        random.shuffle(models)

        for model in models:
            providers.append(
                ProviderCandidate(
                    label=f"{category}:{model}",
                    model=model,
                    api_base=endpoint_url,
                    api_key=api_key,
                    disable_system_prompt=_requires_promptless_mode(endpoint_url, model),
                    tool_free_only=_requires_tool_free_context(endpoint_url, model),
                )
            )

    return providers


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


def _apply_blacklist_if_needed(agent_name: str, model_name: str, api_base: str) -> str:
    """Apply blacklist after model normalization to ensure it's effective.

    This function must be called AFTER _normalize_model_name() to ensure
    the blacklist is applied to the final model name that will be used.

    Args:
        agent_name: Name of the agent (e.g., "DESCRIPTION", "HEXIS_ANALYSIS")
        model_name: Normalized model name after endpoint-specific normalization
        api_base: API base URL

    Returns:
        Model name to use (either original or fallback if blacklisted)
    """
    # Get blacklisted models from global and agent-specific sources
    global_blacklist = os.getenv("GLOBAL_BLACKLISTED_MODELS", "").split(",") if os.getenv("GLOBAL_BLACKLISTED_MODELS") else []
    agent_blacklist = os.getenv(f"{agent_name}_BLACKLISTED_MODELS", "").split(",") if os.getenv(f"{agent_name}_BLACKLISTED_MODELS") else []

    # Combine and clean blacklists (agent-specific takes precedence)
    all_blacklisted = [m.strip() for m in global_blacklist + agent_blacklist if m.strip()]

    # Apply default blacklist for specific agents
    if agent_name == "HEXIS_ANALYSIS" and not agent_blacklist:
        default_blacklist = ["gpt-5", "gpt-5-codex"]
        all_blacklisted.extend(default_blacklist)

    # Check if model is blacklisted
    if model_name in all_blacklisted:
        source = "global" if model_name in global_blacklist else ("default" if agent_name == "HEXIS_ANALYSIS" and not agent_blacklist else "agent-specific")
        print(f"⚠️  {agent_name}: Model '{model_name}' is blacklisted ({source}) after normalization, using fallback", file=sys.stderr)

        # Use safer fallback models
        if "codex" in api_base.lower():
            return "gpt-5-mini"  # Safer alternative for codex endpoint
        else:
            return "claude-sonnet-4.5"  # General fallback

    return model_name


def create_llm_with_rotation(
    *, agent_name: str, model_name: str, api_base: str, api_key: str
) -> BaseLLM:
    """Create an LLM wrapped with provider rotation logic when enabled.

    Auto-detects if model_name is a category and uses category cascade if so.
    """

    # Check if model_name is actually a category
    category_names = ["complex", "intermediate", "simple", "fallback"]
    if model_name.lower() in category_names:
        # Use category-based cascade
        return create_llm_with_category(
            agent_name=agent_name,
            category=model_name,
            api_base=api_base,
            api_key=api_key,
        )

    # Normal model-based rotation
    normalized_model = _normalize_model_name(model_name, api_base)

    # Apply blacklist after model normalization (this fixes the issue)
    normalized_model = _apply_blacklist_if_needed(agent_name, normalized_model, api_base)

    provider_chain = _build_provider_chain(
        agent_name=agent_name,
        normalized_model=normalized_model,
        api_base=api_base,
        api_key=api_key,
    )

    # ALWAYS use RotatingLLM to ensure tools are passed correctly
    # The standard LLM class doesn't pass tools to the API
    labels = ", ".join(candidate.label for candidate in provider_chain)
    print(
        f"\n🔁 {agent_name or 'LLM'} provider chain: {labels}\n",
        file=sys.stderr,
    )
    return RotatingLLM(agent_name or "LLM", provider_chain)


def create_llm_with_category(
    *, agent_name: str, category: str, api_base: str, api_key: str
) -> BaseLLM:
    """Create an LLM with category-based cascade (random selection + auto-fallback).

    New unified approach:
    - All models use the same endpoint (https://ccproxy.emottet.com/v1)
    - Random selection within category for load balancing
    - Automatic cascade to lower tiers on quota/404 errors
    - Crash if fallback category fails (no silent failures)

    Args:
        agent_name: Name of the agent (for logging)
        category: Starting category ("complex", "intermediate", "simple", "fallback")
        api_base: API endpoint URL (typically https://ccproxy.emottet.com/v1)
        api_key: API key for authentication

    Returns:
        RotatingLLM instance with category cascade chain

    Example:
        llm = create_llm_with_category(
            agent_name="HEXIS_ANALYSIS",
            category="complex",
            api_base="https://ccproxy.emottet.com/v1",
            api_key=api_key
        )
    """
    # Build cascade chain from category
    provider_chain = build_category_cascade(
        starting_category=category,
        endpoint_url=api_base,
        api_key=api_key,
        agent_name=agent_name,
    )

    # Log cascade for transparency
    labels = ", ".join(f"{p.label.split(':')[0]}:{len([x for x in provider_chain if x.label.startswith(p.label.split(':')[0])])}" for p in provider_chain[:1])  # Show count per category
    category_counts = {}
    for p in provider_chain:
        cat = p.label.split(':')[0]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    cascade_summary = " → ".join(f"{cat}({count})" for cat, count in category_counts.items())

    print(
        f"\n🎯 {agent_name or 'LLM'} category cascade: {cascade_summary}\n"
        f"   Starting with: {category}\n"
        f"   Total providers: {len(provider_chain)}\n",
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
    candidates = _randomize_primary_providers(candidates)  # Randomize primary providers only
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

    copilot_base = os.getenv("COPILOT_API_BASE", "https://ccproxy.emottet.com/v1")
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


def _randomize_primary_providers(
    candidates: Sequence[ProviderCandidate],
) -> List[ProviderCandidate]:
    """Shuffle primary providers (non-fallback) to distribute load across endpoints.

    The fallback provider (copilot-fallback with gpt-5-mini) is preserved as the last provider.
    This ensures load balancing across primary endpoints while maintaining a stable fallback.
    """
    randomized = list(candidates)
    if len(randomized) <= 1:
        return randomized

    # Separate fallback from primary providers
    fallback_providers = []
    primary_providers = []

    for candidate in randomized:
        # Identify fallback providers by label (copilot-fallback) or model (gpt-5-mini)
        is_fallback = (
            "fallback" in candidate.label.lower() or
            "gpt-5-mini" in candidate.model.lower()
        )

        if is_fallback:
            fallback_providers.append(candidate)
        else:
            primary_providers.append(candidate)

    # Shuffle only the primary providers
    if len(primary_providers) > 1:
        random.shuffle(primary_providers)

    # Combine: randomized primaries + fallbacks at the end
    return primary_providers + fallback_providers


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
    elif "/v1" in base_lower and "ccproxy" in base_lower:
        # New single endpoint: assume copilot behavior (supports short names)
        normalized = _normalize_for_copilot(cleaned)
        print(f"   ✓ New ccproxy endpoint detected → '{normalized}'", file=sys.stderr)
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
        return "claude-sonnet-4-5"
    if "claude-haiku-4.5" in model_lower or "claude-haiku-4-5" in model_lower:
        return "claude-haiku-4-5"
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


def _normalize_for_zai(model_name: str) -> str:
    """Normalize model names for the Z.ai endpoint."""
    # Z.ai accepts exact model names like: glm-4.6
    # Map common variations to canonical names
    model_lower = model_name.lower()

    # GLM models
    if "glm-4.6" in model_lower or "glm4.6" in model_lower:
        return "glm-4.6"
    if "glm-4" in model_lower:
        return "glm-4.6"  # Default to latest version

    # Return as-is for unrecognized models
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

    # Class-level tracking of disabled providers (shared across all instances)
    _disabled_providers: Dict[str, float] = {}  # key = provider_key, value = disabled_at timestamp
    _provider_lock = threading.Lock()  # Thread-safe access to _disabled_providers

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

        # DEBUG: Check what CrewAI passes to call()
        print(
            f"   🔍 RotatingLLM.call() invoked:\n"
            f"      tools param: {len(tools) if tools else 'None'}\n"
            f"      from_agent: {from_agent.role if from_agent else 'None'}\n"
            f"      available_functions: {len(available_functions) if available_functions else 'None'}\n",
            file=sys.stderr
        )

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
            # Check if provider is disabled due to previous quota/rate limit errors
            if self._is_provider_disabled(provider):
                print(
                    f"\n⏭️  {self._agent_name}: SKIPPING {provider.label} ({provider.model})\n"
                    f"   Reason: Provider is temporarily disabled due to quota/rate limit\n",
                    file=sys.stderr,
                )
                continue

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
                    if "claude" in provider.model.lower() and not provider.model.lower().startswith("openai/"):
                        # Force openai/ prefix to ensure LiteLLM uses OpenAI format for tools
                        # This prevents LiteLLM from trying to adapt tools for Anthropic API
                        model_for_litellm = f"openai/{provider.model}"
                    
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
                            max_tokens=32000,  # Increased for weekly meal plans (7 days with full recipes)
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
                    
                    # Tool call loop - continue until LLM stops requesting tools
                    current_response = litellm_response
                    current_messages = formatted_messages
                    max_tool_iterations = 60  # Safety limit (supports 7-day meal plans)
                    iteration = 0

                    while iteration < max_tool_iterations:
                        choice = current_response.choices[0]

                        # Check if LLM wants to call tools
                        if not (hasattr(choice.message, 'tool_calls') and choice.message.tool_calls):
                            # No more tool calls - we have the final response
                            response = choice.message.content or str(choice.message)
                            break

                        iteration += 1
                        print(
                            f"   ✅ LLM returned {len(choice.message.tool_calls)} tool calls (iteration {iteration})!\n",
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
                                        tool_result = tool_func._run(**tool_args) if isinstance(tool_args, dict) else tool_func._run(tool_args)
                                    elif hasattr(tool_func, 'run'):
                                        tool_result = tool_func.run(**tool_args) if isinstance(tool_args, dict) else tool_func.run(tool_args)
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

                        # Build new messages with assistant response and tool results
                        print(
                            f"   🔁 Sending {len(tool_results)} tool results back to LLM...\n",
                            file=sys.stderr
                        )

                        current_messages = current_messages + [
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
                        current_response = litellm.completion(
                            model=model_for_litellm,
                            messages=current_messages,
                            tools=tools,
                            api_base=provider.api_base,
                            api_key=api_key_str,
                            custom_llm_provider=custom_provider,
                            drop_params=True,
                            request_timeout=240,
                            max_tokens=32000,
                        )
                    else:
                        # Max iterations reached
                        print(
                            f"   ⚠️  Max tool iterations ({max_tool_iterations}) reached, using last response\n",
                            file=sys.stderr
                        )
                        response = current_response.choices[0].message.content or str(current_response.choices[0].message)
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

                # Disable the provider to prevent future retry attempts
                self._disable_provider(provider)

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
        model = target.model
        # Force LiteLLM routing for Claude models with custom api_base
        # This prevents CrewAI from using native Anthropic provider which would bypass our proxy
        if target.api_base and "claude" in model.lower() and not model.startswith("openai/"):
            model = f"openai/{model}"
            print(f"   ℹ️  Forcing LiteLLM routing for {target.model} → {model}", file=sys.stderr)
        
        # Force LiteLLM routing for Gemini models with custom api_base
        # This prevents CrewAI from trying to use native Google Gen AI provider
        if target.api_base and "gemini" in model.lower() and not model.startswith("openai/"):
            model = f"openai/{model}"
            print(f"   ℹ️  Forcing LiteLLM routing for {target.model} → {model}", file=sys.stderr)
        return LLM(model=model, api_base=target.api_base, api_key=target.api_key)

    def _sync_metadata(self, llm: LLM) -> None:
        self.model = getattr(llm, "model", self.model)
        self.api_key = getattr(llm, "api_key", self.api_key)
        self.base_url = getattr(llm, "base_url", self.base_url)

    @staticmethod
    def _provider_key(provider: ProviderCandidate) -> str:
        """Generate unique identifier for a provider based on model and endpoint."""
        return f"{provider.model}@{provider.api_base}"

    def _is_provider_disabled(self, provider: ProviderCandidate) -> bool:
        """Check if a provider is currently disabled due to quota/rate limit errors."""
        ttl_seconds = int(os.getenv("PROVIDER_DISABLED_TTL_SECONDS", "0"))  # Default: 0 = disabled for entire execution
        now = time.time()

        with self._provider_lock:
            key = self._provider_key(provider)
            if key not in self._disabled_providers:
                return False

            # If TTL is 0, provider stays disabled for the entire execution
            if ttl_seconds == 0:
                return True

            disabled_at = self._disabled_providers[key]

            # Check if TTL has expired - if so, re-enable the provider
            if now - disabled_at >= ttl_seconds:
                del self._disabled_providers[key]
                ttl_minutes = ttl_seconds // 60
                print(
                    f"✅ {self._agent_name}: provider {provider.label} re-enabled after {ttl_minutes}min TTL\n",
                    file=sys.stderr,
                )
                return False

            return True

    def _disable_provider(self, provider: ProviderCandidate) -> None:
        """Mark a provider as disabled due to quota/rate limit error."""
        ttl_seconds = int(os.getenv("PROVIDER_DISABLED_TTL_SECONDS", "0"))

        with self._provider_lock:
            key = self._provider_key(provider)
            self._disabled_providers[key] = time.time()

        if ttl_seconds == 0:
            print(
                f"\n🚫 {self._agent_name}: provider {provider.label} DISABLED due to quota/rate limit\n"
                f"   Disabled for the remainder of this execution\n"
                f"   Will be available again on next execution (next process start)\n"
                f"   (Configure TTL via PROVIDER_DISABLED_TTL_SECONDS env var, 0 = permanent)\n",
                file=sys.stderr,
            )
        else:
            ttl_minutes = ttl_seconds // 60
            print(
                f"\n🚫 {self._agent_name}: provider {provider.label} DISABLED due to quota/rate limit\n"
                f"   Will be re-enabled after {ttl_minutes} minutes\n"
                f"   (Configure via PROVIDER_DISABLED_TTL_SECONDS env var)\n",
                file=sys.stderr,
            )

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


def _is_codex_error_quota_related(exc: Exception) -> bool:
    """Check if a 401 error from codex endpoint is quota-related (not real auth failure)."""
    try:
        message = str(exc).lower()
        # Codex endpoint returns "token_expired" when quota is exceeded
        return "token_expired" in message
    except Exception:  # pragma: no cover - defensive
        return False


def _is_rate_limit_error(exc: Exception) -> bool:
    exc_name = exc.__class__.__name__
    if exc_name in {"RateLimitError", "RateLimitException"}:
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status_code in {402, 429}:  # 402 = quota exceeded, 429 = rate limit
        return True

    # Handle codex endpoint 401 "token_expired" as quota error (not real auth failure)
    if status_code == 401 and _is_codex_error_quota_related(exc):
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
