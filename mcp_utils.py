"""Utilities for configuring MCP references with discovery-first behaviour."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

CATALOG_ENV_VAR = "MCP_TOOL_CATALOG"
DEFAULT_CATALOG_FILENAME = "all-tool.json"


def _resolve_catalog_path() -> Path:
    """Return the path to the local MCP tool catalogue JSON file."""

    override = os.getenv(CATALOG_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / DEFAULT_CATALOG_FILENAME


@lru_cache(maxsize=1)
def _load_catalog_names() -> List[str]:
    """Load all tool names defined in the local catalogue, if available."""

    path = _resolve_catalog_path()
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    tools = data.get("tools", [])
    names: List[str] = []
    for item in tools:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def load_catalog_tool_names(prefixes: Sequence[str]) -> List[str]:
    """Return catalogue tool names matching any of the provided prefixes."""

    if not prefixes:
        return []

    matches: List[str] = []
    seen = set()
    for name in _load_catalog_names():
        for prefix in prefixes:
            if name.startswith(prefix) and name not in seen:
                matches.append(name)
                seen.add(name)
                break
    return matches


def build_mcp_references(raw_urls: str, tool_names: Iterable[str]) -> List[str]:
    """Compose MCP references for each server.

    Note: Tool discovery happens automatically when connecting to the MCP server.
    We don't need to append tool names as URL fragments - the MCP protocol
    handles tool listing via the list_tools() method.
    
    Args:
        raw_urls: Comma-separated list of MCP server URLs
        tool_names: Not used, kept for backwards compatibility
        
    Returns:
        List of unique base MCP server URLs without tool-specific fragments
    """

    if not raw_urls:
        return []

    references: List[str] = []
    seen = set()

    def _add(reference: str) -> None:
        if reference and reference not in seen:
            references.append(reference)
            seen.add(reference)

    for segment in raw_urls.split(","):
        entry = segment.strip()
        if not entry:
            continue

        if entry.startswith("crewai-amp:"):
            _add(entry)
            continue

        # Remove any existing fragment from the URL
        base_reference, _, _ = entry.partition("#")
        _add(base_reference)

    return references
