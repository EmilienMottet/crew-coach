"""Provider rotation helpers for CrewAI LLM instances."""

from __future__ import annotations

import os
import sys
import random
import threading
import time
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from crewai import LLM
from crewai.agent.core import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel
import json

# Import centralized LLM configuration
from llm_config import (
    COMPLEX_MODELS,
    INTERMEDIATE_MODELS,
    SIMPLE_MODELS,
    FALLBACK_MODELS,
    CATEGORY_CASCADE,
    THINKING_MODELS,
    DEFAULT_AGENT_BLACKLISTS,
    DEFAULT_BASE_URL,
    TOOL_FREE_ENDPOINT_HINTS,
    TOOL_FREE_MODEL_HINTS,
    is_thinking_model,
)


RATE_LIMIT_KEYWORDS = ("rate limit", "quota", "429", "token_expired", "no quota", "unknown provider", "404", "not found", "model not available", "auth_unavailable", "no auth available", "authorized for use with claude code", "403", "forbidden", "permission_denied", "subscription_required")

# Persistent blacklist configuration
DEFAULT_BLACKLIST_FILE = ".disabled_providers.json"
DEFAULT_BASE_TTL_SECONDS = 3600  # 1 hour (strike #1)
DEFAULT_MAX_TTL_SECONDS = 259200  # 72 hours (3 days) - cap for repeated failures
BLACKLIST_FILE_VERSION = 1


class PersistentProviderBlacklist:
    """Persistent storage for disabled providers with strike-based TTL.

    Stores disabled providers in a JSON file with exponential backoff TTL:
    - Strike 1: 1 hour
    - Strike 2: 6 hours
    - Strike 3: 24 hours
    - Strike 4+: 72 hours (cap)

    Thread-safe and handles concurrent access via file locking.
    """

    _instance: Optional["PersistentProviderBlacklist"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "PersistentProviderBlacklist":
        """Get or create singleton instance."""
        with cls._instance_lock:
            if cls._instance is None:
                file_path = os.getenv("DISABLED_PROVIDERS_FILE", DEFAULT_BLACKLIST_FILE)
                cls._instance = cls(file_path)
            return cls._instance

    def __init__(self, file_path: str):
        """Initialize blacklist with given file path.

        Args:
            file_path: Path to JSON file for persistence
        """
        self._file_path = Path(file_path)
        self._lock = threading.Lock()
        self._base_ttl = int(os.getenv("PROVIDER_BASE_TTL_SECONDS", str(DEFAULT_BASE_TTL_SECONDS)))
        self._max_ttl = int(os.getenv("PROVIDER_MAX_TTL_SECONDS", str(DEFAULT_MAX_TTL_SECONDS)))

        # In-memory cache (synced with file)
        self._providers: Dict[str, Dict[str, Any]] = {}

        # Load existing state
        self._load()

    def _load(self) -> None:
        """Load state from JSON file."""
        if not self._file_path.exists():
            self._providers = {}
            return

        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)

            # Validate version
            if data.get("version") != BLACKLIST_FILE_VERSION:
                print(f"⚠️  Blacklist file version mismatch, resetting", file=sys.stderr)
                self._providers = {}
                return

            self._providers = data.get("providers", {})

            # Cleanup expired entries on load
            self._cleanup_expired()

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Failed to load blacklist file: {e}, resetting", file=sys.stderr)
            self._providers = {}

    def _save(self) -> None:
        """Save state to JSON file atomically."""
        data = {
            "version": BLACKLIST_FILE_VERSION,
            "providers": self._providers,
        }

        # Atomic write: write to temp file then rename
        try:
            # Create temp file in same directory for atomic rename
            fd, temp_path = tempfile.mkstemp(
                dir=self._file_path.parent if self._file_path.parent.exists() else None,
                suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                # Atomic rename
                shutil.move(temp_path, self._file_path)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except IOError as e:
            print(f"⚠️  Failed to save blacklist file: {e}", file=sys.stderr)

    def _cleanup_expired(self) -> None:
        """Mark expired entries as re-enabled but keep strike count for future failures.

        Note: We don't delete expired entries - we mark them as "re-enabled" by setting
        disabled_at to 0. This preserves the strike count so that if the provider fails
        again, we can increment the strike and apply a longer TTL.
        """
        now = time.time()
        reenabled_keys = []

        for key, entry in self._providers.items():
            disabled_at = entry.get("disabled_at", 0)
            if disabled_at == 0:
                # Already re-enabled, skip
                continue

            ttl_seconds = entry.get("ttl_seconds", self._base_ttl)

            if now - disabled_at >= ttl_seconds:
                # Mark as re-enabled but keep strike count
                entry["disabled_at"] = 0
                reenabled_keys.append(key)

        if reenabled_keys:
            self._save()
            print(f"✅ Blacklist: {len(reenabled_keys)} expired provider(s) now available", file=sys.stderr)

    def _extract_resets_at(self, error_msg: str) -> Optional[int]:
        """Extract resets_at timestamp from provider error message.

        Some providers return quota reset information in their error responses:
        - OpenAI-style: "resets_at": 1766324470
        - Alternative: "resets_in_seconds": 5971

        Args:
            error_msg: Error message from provider

        Returns:
            Unix timestamp when quota resets, or None if not found
        """
        import re

        if not error_msg:
            return None

        # Try to extract "resets_at": <timestamp>
        resets_at_match = re.search(r'"resets_at"\s*:\s*(\d+)', error_msg)
        if resets_at_match:
            try:
                return int(resets_at_match.group(1))
            except ValueError:
                pass

        # Try to extract "resets_in_seconds": <seconds> and convert to timestamp
        resets_in_match = re.search(r'"resets_in_seconds"\s*:\s*(\d+)', error_msg)
        if resets_in_match:
            try:
                seconds = int(resets_in_match.group(1))
                return int(time.time()) + seconds
            except ValueError:
                pass

        return None

    def _calculate_ttl(self, strike_count: int) -> int:
        """Calculate TTL based on strike count with exponential backoff.

        Uses multipliers based on _base_ttl (default 1h = 3600s):
        - Strike 1: 1x base = 1h
        - Strike 2: 6x base = 6h
        - Strike 3: 24x base = 24h
        - Strike 4+: capped at _max_ttl (default 72h)

        The multipliers ensure consistent ratios even if base TTL is changed
        (e.g., for testing with shorter TTLs).
        """
        # Multipliers relative to base TTL: 1x, 6x, 24x
        multipliers = {
            1: 1,    # 1h if base=3600s
            2: 6,    # 6h if base=3600s
            3: 24,   # 24h if base=3600s
        }

        if strike_count in multipliers:
            ttl = self._base_ttl * multipliers[strike_count]
            return min(ttl, self._max_ttl)

        # Strike 4+: cap at max TTL
        return self._max_ttl

    def is_disabled(self, provider_key: str) -> bool:
        """Check if a provider is currently disabled.

        Args:
            provider_key: Unique provider identifier (model@endpoint)

        Returns:
            True if provider is disabled and TTL hasn't expired
        """
        # Check env var to disable persistence
        if os.getenv("DISABLE_PERSISTENT_BLACKLIST", "").lower() in ("true", "1", "yes"):
            return False

        with self._lock:
            # Reload from file to get latest state
            self._load()

            if provider_key not in self._providers:
                return False

            entry = self._providers[provider_key]
            disabled_at = entry.get("disabled_at", 0)

            # disabled_at == 0 means provider was re-enabled (TTL expired earlier)
            # Strike count is preserved for future failures
            if disabled_at == 0:
                return False

            ttl_seconds = entry.get("ttl_seconds", self._base_ttl)
            now = time.time()

            # Check if TTL has expired
            if now - disabled_at >= ttl_seconds:
                # Mark as re-enabled but preserve strike count
                strike_count = entry.get("strike_count", 1)
                entry["disabled_at"] = 0
                self._save()

                print(
                    f"✅ Blacklist: provider {provider_key} re-enabled after TTL expiry\n"
                    f"   Previous strikes: {strike_count}\n",
                    file=sys.stderr,
                )
                return False

            return True

    def get_remaining_ttl(self, provider_key: str) -> int:
        """Get remaining time before provider is re-enabled.

        Args:
            provider_key: Unique provider identifier

        Returns:
            Remaining seconds, or 0 if not disabled
        """
        with self._lock:
            if provider_key not in self._providers:
                return 0

            entry = self._providers[provider_key]
            disabled_at = entry.get("disabled_at", 0)

            # disabled_at == 0 means already re-enabled
            if disabled_at == 0:
                return 0

            ttl_seconds = entry.get("ttl_seconds", self._base_ttl)

            remaining = ttl_seconds - (time.time() - disabled_at)
            return max(0, int(remaining))

    def get_strike_count(self, provider_key: str) -> int:
        """Get current strike count for a provider.

        Args:
            provider_key: Unique provider identifier

        Returns:
            Current strike count, or 0 if not in blacklist
        """
        with self._lock:
            if provider_key not in self._providers:
                return 0
            return self._providers[provider_key].get("strike_count", 0)

    def disable(self, provider_key: str, error_msg: str = "") -> int:
        """Disable a provider and increment strike count.

        Args:
            provider_key: Unique provider identifier
            error_msg: Error message that triggered the disable

        Returns:
            New strike count
        """
        # Check env var to disable persistence
        if os.getenv("DISABLE_PERSISTENT_BLACKLIST", "").lower() in ("true", "1", "yes"):
            return 0

        with self._lock:
            # Reload to get latest state
            self._load()

            now = time.time()

            # Get existing entry or create new one
            if provider_key in self._providers:
                entry = self._providers[provider_key]
                strike_count = entry.get("strike_count", 0) + 1
            else:
                strike_count = 1

            # Calculate TTL: prefer provider's resets_at timestamp if available
            provider_resets_at = self._extract_resets_at(error_msg)
            if provider_resets_at:
                # Use provider's reset timestamp for more accurate TTL
                ttl_seconds = max(60, int(provider_resets_at - now))  # Min 60s
                ttl_seconds = min(ttl_seconds, self._max_ttl)  # Cap at max TTL
                print(
                    f"   ℹ️  Using provider's resets_at timestamp (TTL: {ttl_seconds}s)",
                    file=sys.stderr,
                )
            else:
                # Fallback to exponential backoff
                ttl_seconds = self._calculate_ttl(strike_count)

            # Update entry
            self._providers[provider_key] = {
                "disabled_at": now,
                "strike_count": strike_count,
                "last_error": error_msg[:500] if error_msg else "",
                "ttl_seconds": ttl_seconds,
            }

            # Save to file
            self._save()

            # Calculate re-enable time
            reenable_at = datetime.fromtimestamp(now + ttl_seconds)
            ttl_human = self._format_duration(ttl_seconds)
            next_ttl = self._calculate_ttl(strike_count + 1)
            next_ttl_human = self._format_duration(next_ttl)

            print(
                f"\n🚫 Blacklist: provider DISABLED (strike #{strike_count})\n"
                f"   Provider: {provider_key}\n"
                f"   Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}\n"
                f"   TTL: {ttl_human} (until {reenable_at.strftime('%Y-%m-%d %H:%M:%S')})\n"
                f"   Next TTL will be: {next_ttl_human}\n",
                file=sys.stderr,
            )

            return strike_count

    def reset_strikes(self, provider_key: str) -> int:
        """Reset strike count after successful call.

        Args:
            provider_key: Unique provider identifier

        Returns:
            Previous strike count (0 if wasn't in blacklist)
        """
        # Check env var to disable persistence
        if os.getenv("DISABLE_PERSISTENT_BLACKLIST", "").lower() in ("true", "1", "yes"):
            return 0

        with self._lock:
            # Reload to get latest state
            self._load()

            if provider_key not in self._providers:
                return 0

            previous_strikes = self._providers[provider_key].get("strike_count", 0)

            # Remove from blacklist entirely on success
            del self._providers[provider_key]
            self._save()

            if previous_strikes > 0:
                print(
                    f"✅ Blacklist: provider {provider_key} SUCCESS, resetting {previous_strikes} strike(s)\n",
                    file=sys.stderr,
                )

            return previous_strikes

    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Format duration in human-readable form."""
        if seconds < 3600:
            return f"{seconds // 60} minutes"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            days = seconds // 86400
            return f"{days} day{'s' if days > 1 else ''}"

# Minimum expected response length for non-JSON responses
# JSON responses are validated by parsing, not length
# Lowered from 500 to 100 to avoid false positives on valid short responses
MIN_EXPECTED_RESPONSE_LENGTH = 100

# Truncation retry configuration
# Retry on same provider before rotating (truncation can be transient)
TRUNCATION_MAX_SAME_PROVIDER_RETRIES = 2
TRUNCATION_RETRY_DELAY_SECONDS = 2.0

# All models are now accessible via the unified /v1 endpoint
# No endpoint requires system prompts to be stripped
PROMPTLESS_ENDPOINT_HINTS: tuple = ()
PROMPTLESS_MODEL_HINTS: tuple = ()  # No model-level restrictions on our endpoints

# NOTE: Model lists (COMPLEX_MODELS, INTERMEDIATE_MODELS, SIMPLE_MODELS, FALLBACK_MODELS,
# CATEGORY_CASCADE, THINKING_MODELS, DEFAULT_AGENT_BLACKLISTS, TOOL_FREE_*) are now
# imported from llm_config.py - single source of truth for all LLM configuration.


def _is_thinking_model_name(model: str) -> bool:
    """Check if a model name is a thinking/reasoning model that hallucinates tool calls.

    Delegates to is_thinking_model() from llm_config.py.
    """
    return is_thinking_model(model)


def _validate_llm_response(response: Any, context: str = "LLM call") -> Any:
    """Validate that an LLM response has valid choices.

    Args:
        response: The response from litellm.completion()
        context: Description of where this validation is happening

    Returns:
        The validated response

    Raises:
        ValueError: If response is malformed (no choices)
    """
    if response is None:
        raise ValueError(f"{context}: Response is None")

    if not hasattr(response, 'choices'):
        raise ValueError(f"{context}: Response has no 'choices' attribute. Response type: {type(response)}")

    if response.choices is None:
        raise ValueError(f"{context}: response.choices is None")

    if len(response.choices) == 0:
        raise ValueError(f"{context}: response.choices is empty")

    return response


def _parse_react_format(content: str) -> dict | None:
    """Parse ReAct format from content if present.

    Some models (like gemini-2.5-flash-lite) output tool calls in ReAct text format:
        Action: tool_name
        Action Input: {"arg": "value"}

    Instead of using the proper tool_calls API field. This function detects and
    parses this format so the tool can still be executed.

    Args:
        content: The message content to parse

    Returns:
        dict with 'name' and 'arguments' if ReAct format found, None otherwise
    """
    if not content:
        return None

    import re

    # Look for Action: and Action Input: pattern (case-insensitive)
    # Action can be at the end of reasoning text, so we look for the last occurrence
    action_match = re.search(r'Action:\s*(\S+)', content, re.IGNORECASE)

    if not action_match:
        return None

    # Look for Action Input: with JSON object
    # The JSON might be on the same line or the next line
    input_match = re.search(
        r'Action Input:\s*(\{.*?\})',
        content,
        re.DOTALL | re.IGNORECASE
    )

    if not input_match:
        return None

    tool_name = action_match.group(1).strip()
    arguments = input_match.group(1).strip()

    # Validate that arguments is valid JSON
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return None

    return {
        'name': tool_name,
        'arguments': arguments
    }


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


def build_category_cascade(starting_category: str, endpoint_url: str, api_key: str, agent_name: str = "", has_tools: bool = False) -> List[ProviderCandidate]:
    """Build a provider chain with category cascade (random selection + fallback to lower tiers).

    Logic:
    1. Start with requested category (e.g., "complex")
    2. Add ALL models from that category in random order (excluding blacklisted models)
    3. If has_tools=True, also exclude thinking models (they hallucinate tool calls)
    4. Cascade to next lower tier (complex → intermediate → simple → fallback)
    5. If fallback fails → crash (no recovery)

    Args:
        starting_category: One of "complex", "intermediate", "simple", "fallback"
        endpoint_url: API endpoint URL (typically https://ccproxy.emottet.com/v1)
        api_key: API key for authentication
        agent_name: Name of the agent (for agent-specific blacklists)
        has_tools: If True, exclude thinking models that hallucinate tool calls

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

    # Apply default blacklist from centralized constant (only if no env-specific blacklist)
    if agent_name and not agent_blacklist and agent_name in DEFAULT_AGENT_BLACKLISTS:
        all_blacklisted.extend(DEFAULT_AGENT_BLACKLISTS[agent_name])

    if all_blacklisted:
        print(f"   🚫 Blacklist active: {', '.join(all_blacklisted[:5])}{'...' if len(all_blacklisted) > 5 else ''}", file=sys.stderr)

    providers = []

    # Build cascade from starting category to fallback
    for category in CATEGORY_CASCADE[start_index:]:
        models = list(get_models_for_category(category))

        # Filter out blacklisted models
        models = [m for m in models if m not in all_blacklisted]

        # CRITICAL: Exclude thinking models for agents with tools
        # Thinking models hallucinate tool calls (ReAct in content) instead of using tool_calls
        # This causes CrewAI to treat the response as "Final Answer" without executing tools
        if has_tools:
            before_count = len(models)
            models = [m for m in models if not _is_thinking_model_name(m)]
            excluded = before_count - len(models)
            if excluded > 0:
                print(f"   🧠 Excluded {excluded} thinking model(s) from '{category}' (agent has tools)", file=sys.stderr)

        if not models:
            print(f"   ⚠️  All models in category '{category}' are blacklisted/excluded, skipping category", file=sys.stderr)
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

    # Return True if endpoint doesn't support tools
    if any(keyword in base for keyword in TOOL_FREE_ENDPOINT_HINTS):
        return True

    # Return True if model doesn't support tools
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

    # Apply default blacklist from centralized constant (only if no env-specific blacklist)
    if agent_name and not agent_blacklist and agent_name in DEFAULT_AGENT_BLACKLISTS:
        all_blacklisted.extend(DEFAULT_AGENT_BLACKLISTS[agent_name])

    # Check if model is blacklisted
    if model_name in all_blacklisted:
        source = "global" if model_name in global_blacklist else ("default" if agent_name in DEFAULT_AGENT_BLACKLISTS else "agent-specific")
        print(f"⚠️  {agent_name}: Model '{model_name}' is blacklisted ({source}) after normalization, using fallback", file=sys.stderr)

        # Use safer fallback models
        if "codex" in api_base.lower():
            return "gpt-5-mini"  # Safer alternative for codex endpoint
        else:
            return "claude-sonnet-4.5"  # General fallback

    return model_name


def create_llm_with_rotation(
    *, agent_name: str, model_name: str, api_base: str, api_key: str, has_tools: bool = False
) -> BaseLLM:
    """Create an LLM wrapped with provider rotation logic when enabled.

    Auto-detects if model_name is a category and uses category cascade if so.

    Args:
        agent_name: Name of the agent (for logging)
        model_name: Model name or category ("complex", "intermediate", "simple", "fallback")
        api_base: API endpoint URL
        api_key: API key for authentication
        has_tools: If True, exclude thinking models that hallucinate tool calls
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
            has_tools=has_tools,
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
    *, agent_name: str, category: str, api_base: str, api_key: str, has_tools: bool = False
) -> BaseLLM:
    """Create an LLM with category-based cascade (random selection + auto-fallback).

    New unified approach:
    - All models use the same endpoint (https://ccproxy.emottet.com/v1)
    - Random selection within category for load balancing
    - If has_tools=True, exclude thinking models (they hallucinate tool calls)
    - Automatic cascade to lower tiers on quota/404 errors
    - Crash if fallback category fails (no silent failures)

    Args:
        agent_name: Name of the agent (for logging)
        category: Starting category ("complex", "intermediate", "simple", "fallback")
        api_base: API endpoint URL (typically https://ccproxy.emottet.com/v1)
        api_key: API key for authentication
        has_tools: If True, exclude thinking models that hallucinate tool calls

    Returns:
        RotatingLLM instance with category cascade chain

    Example:
        llm = create_llm_with_category(
            agent_name="HEXIS_ANALYSIS",
            category="complex",
            api_base="https://ccproxy.emottet.com/v1",
            api_key=api_key,
            has_tools=True
        )
    """
    # Build cascade chain from category
    provider_chain = build_category_cascade(
        starting_category=category,
        endpoint_url=api_base,
        api_key=api_key,
        agent_name=agent_name,
        has_tools=has_tools,
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

    # Debug logging
    print(f"🔍 Normalizing model: '{cleaned}' for endpoint: '{api_base}'", file=sys.stderr)

    # Default: add openai/ prefix if needed for LiteLLM to use OpenAI format
    # This is required for non-OpenAI models (like Claude) when using an OpenAI-compatible proxy
    if "/" not in cleaned and not cleaned.startswith("gpt-"):
         # For Claude models via proxy, we usually need openai/ prefix for LiteLLM
         # unless it's a GPT model which LiteLLM handles natively
         if "claude" in cleaned.lower() or "gemini" in cleaned.lower() or "deepseek" in cleaned.lower():
             return f"openai/{cleaned}"
             
    return cleaned





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



def _clean_json_schema(schema: Any) -> Any:
    """Recursively clean JSON schema to be Claude/OpenAI compatible.

    Removes unsupported fields, empty containers, and ensures structure is valid.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    
    # Handle optional fields (anyOf/oneOf with null) - common in Pydantic v2
    # We pick the first non-null option to ensure the field has a type
    if 'anyOf' in schema and isinstance(schema['anyOf'], list):
        for option in schema['anyOf']:
            if isinstance(option, dict) and option.get('type') != 'null':
                # Found a non-null option, use it as the base
                cleaned.update(_clean_json_schema(option))
                break
    elif 'oneOf' in schema and isinstance(schema['oneOf'], list):
        for option in schema['oneOf']:
            if isinstance(option, dict) and option.get('type') != 'null':
                cleaned.update(_clean_json_schema(option))
                break

    for key, value in schema.items():
        # Skip unsupported fields (we handled anyOf/oneOf above)
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
            cleaned[key] = _clean_json_schema(value)
        # Recursively clean dicts in lists
        elif isinstance(value, list):
            cleaned[key] = [_clean_json_schema(item) if isinstance(item, dict) else item for item in value]
        # Keep other values as-is
        else:
            cleaned[key] = value

    return cleaned


def _requires_anthropic_format(model_name: str, has_tools: bool = False, api_base: str = "") -> bool:
    """Check if model requires Anthropic-style API format.

    Returns True for:
    1. Claude thinking models (always need native Anthropic API)
    2. Claude models with tools on non-proxy endpoints

    Returns False for proxy endpoints (ccproxy, cliapi) which use OpenAI-compatible
    format for tool calling.

    Other models (kimi, deepseek, etc.) use OpenAI-compatible format even if they have "thinking".

    Args:
        model_name: Model name to check
        has_tools: Whether agent has tools
        api_base: API base URL to check for proxy endpoints
    """
    model_lower = model_name.lower()

    # CRITICAL: Never use Anthropic native API for proxy endpoints
    # Proxies like ccproxy use OpenAI-compatible format for tool calling
    # They don't support /v1/messages endpoint, only /v1/chat/completions
    if api_base:
        proxy_patterns = ["ccproxy", "cliapi", "proxy"]
        if any(pattern in api_base.lower() for pattern in proxy_patterns):
            return False

    # Claude/Anthropic thinking models always need native Anthropic API
    anthropic_thinking_patterns = [
        "claude-sonnet-4-5-thinking",     # Claude Sonnet 4.5 thinking
        "claude-opus-4-5-thinking",       # Claude Opus 4.5 thinking variants
        "gemini-claude-sonnet-4-5-thinking",  # Gemini-proxied Claude thinking
    ]
    if any(pattern in model_lower for pattern in anthropic_thinking_patterns):
        return True

    # Claude models with tools need Anthropic native API
    # CLIProxyAPI doesn't properly handle tools for Claude via /v1/chat/completions
    # but works correctly via /v1/messages (native Anthropic format)
    if has_tools and "claude" in model_lower:
        return True

    return False


def _is_thinking_model(model: str) -> bool:
    """Check if model has extended thinking enabled.

    These models require thinking blocks to be preserved in multi-turn conversations.
    When thinking is enabled, assistant messages must start with a thinking block,
    and previous thinking blocks must be passed back in subsequent turns.
    """
    model_lower = model.lower()
    thinking_patterns = [
        "thinking",      # All Claude thinking variants
        "-r1",           # deepseek-r1
    ]
    return any(pattern in model_lower for pattern in thinking_patterns)


def _convert_to_anthropic_format(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI function calling format to Anthropic tool format.

    OpenAI format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Anthropic format:
        {"name": "...", "description": "...", "input_schema": {...}}
    """
    anthropic_tools = []

    for i, tool in enumerate(openai_tools):
        if not isinstance(tool, dict):
            print(f"   ⚠️  Tool {i} is not a dict: {type(tool)}", file=sys.stderr)
            continue

        if tool.get("type") != "function":
            print(f"   ⚠️  Tool {i} has unexpected type: {tool.get('type')}", file=sys.stderr)
            continue

        func = tool.get("function", {})
        tool_name = func.get("name", "")

        if not tool_name:
            print(f"   ⚠️  Tool {i} has no name! Keys in function: {list(func.keys())}", file=sys.stderr)

        # Build Anthropic-style tool
        anthropic_tool = {
            "name": tool_name,
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {
                "type": "object",
                "properties": {},
            })
        }

        anthropic_tools.append(anthropic_tool)

    # Log first few converted tools for debugging
    if anthropic_tools:
        sample_names = [t.get("name", "?") for t in anthropic_tools[:5]]
        print(f"   🔍 Converted tool names sample: {sample_names}", file=sys.stderr)

    return anthropic_tools


class RotatingLLM(BaseLLM):

    """BaseLLM wrapper that retries calls across multiple providers on 429 errors."""

    # Use persistent blacklist singleton (shared across all instances)
    # This replaces the old in-memory _disabled_providers dict
    _blacklist: Optional[PersistentProviderBlacklist] = None

    @classmethod
    def _get_blacklist(cls) -> PersistentProviderBlacklist:
        """Get the persistent blacklist singleton."""
        if cls._blacklist is None:
            cls._blacklist = PersistentProviderBlacklist.get_instance()
        return cls._blacklist

    def __init__(self, agent_name: str, providers: Sequence[ProviderCandidate]) -> None:
        if not providers:
            raise ValueError("At least one provider is required for rotation")

        self._agent_name = agent_name or "LLM"
        self._providers = list(providers)
        self._llms: List[LLM | None] = [None] * len(self._providers)
        self._last_success_index = 0
        # Storage for thinking blocks across multi-turn conversations
        # Needed because CrewAI doesn't preserve thinking blocks in messages
        self._last_thinking_blocks: List[Dict[str, Any]] | None = None
        self._last_reasoning_content: str | None = None
        # Track truncation retry attempts per provider (transient issue mitigation)
        self._truncation_retry_count: Dict[str, int] = {}

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

        # Ensure tools passed directly are also cleaned
        if tools:
            cleaned_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    # It's already in tool format, clean the parameters
                    if "parameters" in tool["function"]:
                        tool["function"]["parameters"] = _clean_json_schema(tool["function"]["parameters"])
                        
                        # Ensure required fields exist
                        if "type" not in tool["function"]["parameters"]:
                            tool["function"]["parameters"]["type"] = "object"
                        if "properties" not in tool["function"]["parameters"]:
                            tool["function"]["parameters"]["properties"] = {}
                    cleaned_tools.append(tool)
                else:
                    # It might be a raw tool object, let the extraction logic handle it if needed
                    # But usually 'tools' arg is already formatted by CrewAI
                    cleaned_tools.append(tool)
            tools = cleaned_tools

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
                    )
                
                # Clean tools if they were extracted from agent
                # This ensures they are compatible with Claude/OpenAI
                if available_functions:
                    # We'll do the cleaning in the conversion loop below
                    pass
                
                # ALWAYS try to convert to tools format for function calling
                try:
                    from crewai.tools.base_tool import BaseTool as CrewBaseTool
                    import json
                    

                    
                    tools = []
                    for tool in agent_tools:
                        tool_def = None
                        
                        # PRIORITY 1: Use args_schema if available (most reliable)
                        # Check for _original_args_schema first (in case tool is wrapped)
                        schema_to_use = getattr(tool, '_original_args_schema', None) or getattr(tool, 'args_schema', None)
                        
                        if schema_to_use:
                            try:
                                schema = schema_to_use
                                if hasattr(schema, 'model_json_schema'):
                                    json_schema = schema.model_json_schema()
                                    # Deep clean the schema
                                    json_schema = _clean_json_schema(json_schema)
                                    
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
                                    func_def['function']['parameters'] = _clean_json_schema(params)
                                    
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

            # CRITICAL: Skip thinking models if messages have assistant without thinking blocks
            # and we don't have stored blocks to inject. This prevents API errors.
            if _is_thinking_model(provider.model):
                # Check if base_messages contains assistant messages without thinking blocks
                has_incompatible_assistant = False
                for msg in base_messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        if not msg.get("thinking_blocks"):
                            has_incompatible_assistant = True
                            break

                # If incompatible and we have no stored blocks, skip this provider
                if has_incompatible_assistant and not self._last_thinking_blocks:
                    print(
                        f"\n⏭️  {self._agent_name}: SKIPPING thinking model {provider.label} ({provider.model})\n"
                        f"   Reason: Messages contain assistant without thinking blocks\n"
                        f"   Cannot use thinking models in multi-turn without preserved thinking blocks\n",
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
                        formatted_messages = list(effective_messages)  # Make a copy to avoid mutation

                    # CRITICAL: Inject stored thinking blocks for thinking models
                    # CrewAI doesn't preserve thinking blocks in messages, so we need to
                    # inject them back into the last assistant message
                    if _is_thinking_model(provider.model) and self._last_thinking_blocks:
                        # Find the last assistant message that needs thinking blocks
                        for i in range(len(formatted_messages) - 1, -1, -1):
                            msg = formatted_messages[i]
                            if msg.get("role") == "assistant" and not msg.get("thinking_blocks"):
                                # Inject thinking blocks into this message
                                formatted_messages[i] = dict(msg)  # Make a copy
                                formatted_messages[i]["thinking_blocks"] = self._last_thinking_blocks
                                if self._last_reasoning_content:
                                    formatted_messages[i]["reasoning_content"] = self._last_reasoning_content
                                print(
                                    f"   🧠 Injected {len(self._last_thinking_blocks)} thinking block(s) into assistant message\n",
                                    file=sys.stderr
                                )
                                break

                    # Prepare API key - use Bearer token format (standard for OpenAI-compatible APIs)
                    api_key_str = str(provider.api_key) if provider.api_key else ""

                    # Determine if we need Anthropic native API
                    # Claude models with tools need /v1/messages endpoint (CLIProxyAPI limitation)
                    # BUT proxy endpoints (ccproxy) use OpenAI-compatible format for tool calling
                    has_tools = bool(tools)
                    use_anthropic_api = _requires_anthropic_format(provider.model, has_tools, provider.api_base)

                    # Adjust api_base for Anthropic (needs /v1/messages, not /v1/chat/completions)
                    # LiteLLM appends /v1/messages to api_base for Anthropic, so we need to remove /v1
                    api_base_for_call = provider.api_base
                    if use_anthropic_api and provider.api_base.endswith("/v1"):
                        api_base_for_call = provider.api_base[:-3]  # Remove /v1 suffix
                        print(
                            f"   📍 Adjusted api_base for Anthropic: {api_base_for_call}\n",
                            file=sys.stderr
                        )

                    # Set custom provider based on API format
                    custom_provider = "anthropic" if use_anthropic_api else "openai"

                    # LiteLLM needs provider prefix for some models
                    model_for_litellm = provider.model

                    if use_anthropic_api:
                        # Use anthropic/ prefix to route to /v1/messages
                        if not provider.model.lower().startswith("anthropic/"):
                            model_for_litellm = f"anthropic/{provider.model}"
                        print(
                            f"   🔄 Using Anthropic native API for {provider.model} with {len(tools) if tools else 0} tools\n",
                            file=sys.stderr
                        )
                    elif "claude" in provider.model.lower() and not provider.model.lower().startswith("openai/"):
                        # Claude without tools - use openai/ prefix (original behavior)
                        model_for_litellm = f"openai/{provider.model}"

                    # Tools are passed as-is - LiteLLM handles conversion automatically
                    # when using anthropic/ prefix and custom_llm_provider="anthropic"
                    tools_for_api = tools

                    # Call litellm directly with the necessary parameters
                    import litellm
                    try:
                        litellm_response = litellm.completion(
                            model=model_for_litellm,
                            messages=formatted_messages,
                            tools=tools_for_api,
                            api_base=api_base_for_call,  # Adjusted for Anthropic if needed
                            api_key=api_key_str,
                            custom_llm_provider=custom_provider,
                            drop_params=True,
                            timeout=180,  # 180 seconds timeout for httpx/openai client
                            max_tokens=32000,  # Increased for weekly meal plans (7 days with full recipes)
                        )
                            
                    except Exception as e:
                        print(
                            f"   ❌ litellm.completion failed: {e}\n"
                            f"      Falling back to CrewAI LLM.call()\n",
                            file=sys.stderr
                        )
                        # Log detailed error info for debugging
                        if hasattr(e, "response"):
                            print(f"      Response content: {getattr(e, 'response', 'N/A')}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

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

                    # Validate response before using
                    try:
                        _validate_llm_response(litellm_response, "Initial litellm.completion")
                    except ValueError as e:
                        # Invalid response (choices=None) - raise to try next provider
                        # Don't fallback to CrewAI's llm.call() as it has the same bug
                        print(f"   ❌ Invalid LLM response: {e}\n", file=sys.stderr)
                        raise RuntimeError(f"Provider returned invalid response: {e}")

                    # Tool call loop - continue until LLM stops requesting tools
                    current_response = litellm_response
                    current_messages = formatted_messages
                    max_tool_iterations = 60  # Safety limit (supports 7-day meal plans)
                    iteration = 0

                    while iteration < max_tool_iterations:
                        choice = current_response.choices[0]

                        # Check if LLM wants to call tools
                        print(f"   🔍 DEBUG: choice.message: {choice.message}", file=sys.stderr)

                        # ALWAYS capture thinking blocks from response for thinking models
                        # This is critical for multi-turn conversations where CrewAI
                        # will parse ReAct actions and call us again without the thinking
                        if _is_thinking_model(model_for_litellm):
                            thinking_blocks = getattr(choice.message, 'thinking_blocks', None)
                            reasoning_content = getattr(choice.message, 'reasoning_content', None)
                            # Also check provider_specific_fields
                            if not thinking_blocks:
                                provider_fields = getattr(choice.message, 'provider_specific_fields', {}) or {}
                                thinking_blocks = provider_fields.get('thinking_blocks')
                            if not reasoning_content:
                                reasoning_content = getattr(choice.message, 'reasoning', None)

                            if thinking_blocks:
                                self._last_thinking_blocks = thinking_blocks
                                print(f"   🧠 Captured {len(thinking_blocks)} thinking block(s) for future turns", file=sys.stderr)
                            if reasoning_content:
                                self._last_reasoning_content = reasoning_content

                        if not (hasattr(choice.message, 'tool_calls') and choice.message.tool_calls):
                            # No tool_calls in response - check for ReAct format in content
                            react_action = _parse_react_format(choice.message.content)

                            if react_action and available_functions:
                                tool_name = react_action['name']
                                tool_args_str = react_action['arguments']

                                if tool_name in available_functions:
                                    # Found ReAct format with valid tool - execute it
                                    iteration += 1
                                    print(
                                        f"   🔄 Detected ReAct format in content (iteration {iteration}): {tool_name}\n",
                                        file=sys.stderr
                                    )

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
                                            f"   ✅ Tool {tool_name} executed successfully (ReAct)\n"
                                            f"      Result: {str(tool_result)[:200]}...\n",
                                            file=sys.stderr
                                        )

                                        # Generate a pseudo tool_call_id for ReAct format
                                        react_tool_call_id = f"react_{tool_name}_{iteration}"

                                        tool_results = [{
                                            "tool_call_id": react_tool_call_id,
                                            "role": "tool",
                                            "name": tool_name,
                                            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                                        }]

                                        # Build assistant message with ReAct content (no tool_calls)
                                        assistant_message = {
                                            "role": "assistant",
                                            "content": choice.message.content,
                                            "tool_calls": [{
                                                "id": react_tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": tool_args_str
                                                }
                                            }]
                                        }

                                        current_messages = current_messages + [assistant_message] + tool_results

                                        # Call LLM again with tool results
                                        print(
                                            f"   🔁 Sending ReAct tool result back to LLM...\n",
                                            file=sys.stderr
                                        )

                                        current_response = litellm.completion(
                                            model=model_for_litellm,
                                            messages=current_messages,
                                            tools=tools_for_api,
                                            api_base=api_base_for_call,
                                            api_key=api_key_str,
                                            custom_llm_provider=custom_provider,
                                            drop_params=True,
                                            request_timeout=240,
                                            max_tokens=32000,
                                        )

                                        # Validate follow-up response
                                        try:
                                            _validate_llm_response(current_response, f"ReAct tool loop iteration {iteration}")
                                        except ValueError as e:
                                            print(f"   ❌ Invalid LLM response in ReAct tool loop: {e}\n", file=sys.stderr)
                                            response = "Error: LLM returned invalid response after ReAct tool execution"
                                            break

                                        continue  # Continue the tool loop

                                    except Exception as tool_error:
                                        print(
                                            f"   ❌ ReAct tool {tool_name} failed: {tool_error}\n",
                                            file=sys.stderr
                                        )
                                        # Fall through to treat as final response

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

                        # Build assistant message with thinking blocks preserved
                        assistant_message = {
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

                        # Preserve thinking blocks for thinking models (required by Anthropic API)
                        # When thinking is enabled, assistant messages must include the original
                        # thinking blocks with their cryptographic signatures for multi-turn
                        if _is_thinking_model(model_for_litellm):
                            thinking_blocks = getattr(choice.message, 'thinking_blocks', None)
                            if thinking_blocks:
                                assistant_message["thinking_blocks"] = thinking_blocks
                                print(f"   🧠 Preserved {len(thinking_blocks)} thinking block(s) for multi-turn", file=sys.stderr)

                            reasoning_content = getattr(choice.message, 'reasoning_content', None)
                            if reasoning_content:
                                assistant_message["reasoning_content"] = reasoning_content

                        current_messages = current_messages + [assistant_message] + tool_results

                        # Call LLM again with tool results
                        current_response = litellm.completion(
                            model=model_for_litellm,
                            messages=current_messages,
                            tools=tools_for_api,
                            api_base=api_base_for_call,  # Use adjusted api_base for Anthropic
                            api_key=api_key_str,
                            custom_llm_provider=custom_provider,
                            drop_params=True,
                            request_timeout=240,
                            max_tokens=32000,
                        )

                        # Validate follow-up response
                        try:
                            _validate_llm_response(current_response, f"Tool loop iteration {iteration}")
                        except ValueError as e:
                            print(f"   ❌ Invalid LLM response in tool loop: {e}\n", file=sys.stderr)
                            response = "Error: LLM returned invalid response after tool execution"
                            break
                    else:
                        # Max iterations reached
                        print(
                            f"   ⚠️  Max tool iterations ({max_tool_iterations}) reached, using last response\n",
                            file=sys.stderr
                        )
                        # Safely access response with validation
                        try:
                            _validate_llm_response(current_response, "Max iterations fallback")
                            response = current_response.choices[0].message.content or str(current_response.choices[0].message)
                        except ValueError as e:
                            print(f"   ❌ Invalid response at max iterations: {e}\n", file=sys.stderr)
                            response = "Error: LLM returned invalid response"
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

                # Log response length for debugging
                response_str = str(response) if response else ""
                response_len = len(response_str)
                print(
                    f"   🔍 Response length: {response_len} chars",
                    file=sys.stderr,
                )
                if response_len < 1000:
                    print(
                        f"   ⚠️  WARNING: Suspiciously short response ({response_len} chars)",
                        file=sys.stderr,
                    )

                # Check if response appears truncated
                if _is_response_truncated(response_str):
                    provider_key = self._provider_key(provider)

                    # Track truncation occurrences for debugging
                    self._truncation_retry_count[provider_key] = self._truncation_retry_count.get(provider_key, 0) + 1
                    truncation_count = self._truncation_retry_count[provider_key]

                    error_msg = f"Truncated response ({response_len} chars)"
                    print(
                        f"\n⚠️  {self._agent_name}: TRUNCATED RESPONSE DETECTED\n"
                        f"   Provider: {provider.label}\n"
                        f"   Model: {provider.model}\n"
                        f"   Response length: {response_len} chars\n"
                        f"   Truncation count for this provider: {truncation_count}\n"
                        f"   Disabling provider and rotating to next...\n",
                        file=sys.stderr,
                    )

                    # Disable provider and try next one
                    self._disable_provider(provider, error_msg)
                    last_error = RuntimeError(error_msg)
                    continue

                self._last_success_index = index
                self._sync_metadata(llm)

                # Reset truncation and strike counts on successful call
                provider_key = self._provider_key(provider)
                if provider_key in self._truncation_retry_count:
                    del self._truncation_retry_count[provider_key]
                self._reset_provider_strikes(provider)

                # Log which provider was successfully used
                print(
                    f"✅ {self._agent_name}: successfully used {provider.label} ({provider.model})\n",
                    file=sys.stderr,
                )

                return response

            except Exception as exc:  # noqa: BLE001
                is_rate_limit = _is_rate_limit_error(exc)
                is_context_length = _is_context_length_error(exc)
                is_invalid_response = _is_invalid_response_error(exc)

                if not is_rate_limit and not is_context_length and not is_invalid_response:
                    raise

                if is_rate_limit:
                    # Disable the provider to prevent future retry attempts
                    error_msg = str(exc)[:500]
                    self._disable_provider(provider, error_msg)
                    print(
                        f"⚠️  {self._agent_name}: provider {provider.label} hit quota, rotating...",
                        file=sys.stderr,
                    )
                elif is_context_length:
                    # Don't disable - just skip to a model with larger context
                    print(
                        f"⚠️  {self._agent_name}: provider {provider.label} context too small for this request\n"
                        f"   Rotating to next provider with larger context window...",
                        file=sys.stderr,
                    )
                elif is_invalid_response:
                    # Provider returned malformed response (choices=None, etc.)
                    # Don't disable permanently - might be a transient issue
                    print(
                        f"⚠️  {self._agent_name}: provider {provider.label} returned invalid response\n"
                        f"   Error: {str(exc)[:200]}\n"
                        f"   Rotating to next provider...",
                        file=sys.stderr,
                    )

                last_error = exc
                if index == len(self._providers) - 1:
                    break

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
        """Check if a provider is currently disabled due to quota/rate limit errors.

        Uses persistent blacklist with TTL-based expiry and strike counting.
        """
        key = self._provider_key(provider)
        blacklist = self._get_blacklist()

        if blacklist.is_disabled(key):
            # Get remaining TTL for logging
            remaining = blacklist.get_remaining_ttl(key)
            strike_count = blacklist.get_strike_count(key)
            remaining_human = PersistentProviderBlacklist._format_duration(remaining)
            print(
                f"⏭️  {self._agent_name}: SKIPPING {provider.label} ({provider.model})\n"
                f"   Reason: Disabled (strike #{strike_count}), {remaining_human} remaining\n",
                file=sys.stderr,
            )
            return True

        return False

    def _disable_provider(self, provider: ProviderCandidate, error_msg: str = "") -> None:
        """Mark a provider as disabled due to quota/rate limit error.

        Increments strike count and calculates TTL with exponential backoff.
        State is persisted to JSON file for cross-process recovery.
        """
        key = self._provider_key(provider)
        blacklist = self._get_blacklist()
        blacklist.disable(key, error_msg)

    def _reset_provider_strikes(self, provider: ProviderCandidate) -> None:
        """Reset strike count after a successful call.

        Called when a provider successfully handles a request to clear
        any accumulated strikes from previous failures.
        """
        key = self._provider_key(provider)
        blacklist = self._get_blacklist()
        blacklist.reset_strikes(key)

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
    if status_code in {402, 403, 429}:  # 402 = quota exceeded, 403 = forbidden/no license, 429 = rate limit
        return True

    # Handle codex endpoint 401 "token_expired" as quota error (not real auth failure)
    if status_code == 401 and _is_codex_error_quota_related(exc):
        return True

    try:
        message = str(exc).lower()
    except Exception:  # pragma: no cover - defensive
        message = ""

    return any(keyword in message for keyword in RATE_LIMIT_KEYWORDS)


def _is_response_truncated(
    response: str,
    min_length: int = MIN_EXPECTED_RESPONSE_LENGTH,
) -> bool:
    """Detect if a response appears to be truncated.

    Returns True if:
    - Response is empty or very short (1-10 chars) and not valid minimal JSON, OR
    - Response contains JSON that fails to parse (incomplete JSON), OR
    - Response is non-JSON and shorter than min_length

    IMPORTANT: Valid JSON is NEVER considered truncated, regardless of length.
    This prevents false positives on short but complete JSON responses.
    """
    if not response:
        return True

    stripped = response.strip()

    # Very short responses (< 10 chars) are almost certainly truncated
    # unless they are valid minimal JSON like "{}", "[]", "null", "true", "false"
    if len(stripped) < 10:
        # Check if it's valid minimal JSON
        if stripped in ("{}", "[]", "null", "true", "false"):
            return False  # Valid minimal JSON

        # Single char or very short non-JSON is definitely truncated
        if len(stripped) <= 2:
            return True

        # Try to parse as JSON
        try:
            json.loads(stripped)
            return False  # Valid short JSON
        except json.JSONDecodeError:
            return True  # Short non-JSON = truncated

    # If response looks like JSON, check JSON validity
    # Valid JSON = not truncated, regardless of length
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return False  # Valid JSON = not truncated
        except json.JSONDecodeError:
            # JSON is incomplete/malformed - likely truncated
            return True

    # For non-JSON responses, use length threshold
    response_len = len(response)
    if response_len < min_length:
        return True

    return False


# Keywords that indicate context length / token limit errors
CONTEXT_LENGTH_KEYWORDS = (
    "prompt is too long",
    "context_length_exceeded",
    "context length",
    "maximum context",
    "token limit",
    "tokens > ",
    "exceeds the maximum",
    "too many tokens",
    "input too long",
    "max_tokens",
)


def _is_context_length_error(exc: Exception) -> bool:
    """Check if an exception is due to context length / token limit exceeded.

    These errors should trigger rotation to a model with a larger context window,
    not be treated as fatal errors.
    """
    try:
        message = str(exc).lower()
    except Exception:  # pragma: no cover - defensive
        message = ""

    return any(keyword in message for keyword in CONTEXT_LENGTH_KEYWORDS)


def _is_invalid_response_error(exc: Exception) -> bool:
    """Check if an exception is due to an invalid/malformed LLM response.

    These errors should trigger rotation to the next provider, not be treated
    as fatal errors. Common cases:
    - response.choices is None
    - response.choices is empty
    - Provider returned non-standard response format
    """
    try:
        message = str(exc).lower()
    except Exception:  # pragma: no cover - defensive
        message = ""

    invalid_response_keywords = (
        "invalid response",
        "choices is none",
        "choices is empty",
        "response is none",
        "nonetype",
        "no 'choices' attribute",
    )
    return any(keyword in message for keyword in invalid_response_keywords)


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
