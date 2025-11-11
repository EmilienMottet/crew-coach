"""Initialize LLM authentication before any CrewAI imports.

This module MUST be imported first to ensure Basic Auth is properly configured
for all LLM instances, including those created by CrewAI's structured output system.
"""
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment immediately
load_dotenv(override=True)


def initialize_basic_auth() -> str:
    """
    Initialize Basic Auth configuration for LiteLLM/OpenAI clients.

    Returns:
        The configured API key (Basic Auth token or standard key)
    """
    auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")
    base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/copilot/v1")

    # Set base URL
    os.environ["OPENAI_API_BASE"] = base_url

    if not auth_token:
        # Use standard API key
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key

    # Configure Basic Auth
    basic_token = auth_token if auth_token.startswith("Basic ") else f"Basic {auth_token}"
    os.environ["OPENAI_API_KEY"] = basic_token

    # Apply monkey-patch to OpenAI clients
    from openai import AsyncOpenAI, OpenAI
    from litellm.llms.openai.openai import OpenAIConfig

    def _basic_auth_headers(self: Any) -> Dict[str, str]:
        key = getattr(self, "api_key", "") or ""
        if isinstance(key, str) and key.startswith("Basic "):
            return {"Authorization": key}
        return {"Authorization": f"Bearer {key}"}

    if not getattr(OpenAI, "_basic_auth_patched", False):
        OpenAI.auth_headers = property(_basic_auth_headers)  # type: ignore[assignment]
        AsyncOpenAI.auth_headers = property(_basic_auth_headers)  # type: ignore[assignment]
        setattr(OpenAI, "_basic_auth_patched", True)
        setattr(AsyncOpenAI, "_basic_auth_patched", True)

    if not getattr(OpenAIConfig, "_basic_auth_patched", False):
        original_validate_env = OpenAIConfig.validate_environment

        def validate_environment_with_basic(
            self: Any,
            headers: Dict[str, str],
            model: str,
            messages: List[Any],
            optional_params: Dict[str, Any],
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
        ) -> Dict[str, str]:
            result = original_validate_env(
                self,
                headers,
                model,
                messages,
                optional_params,
                api_key=api_key,
                api_base=api_base,
            )
            if isinstance(api_key, str) and api_key.startswith("Basic "):
                result["Authorization"] = api_key
            return result

        OpenAIConfig.validate_environment = validate_environment_with_basic  # type: ignore[assignment]
        setattr(OpenAIConfig, "_basic_auth_patched", True)

    # CRITICAL: Also patch Instructor's client creation for structured outputs
    try:
        import instructor
        from instructor.client import Instructor

        # Patch Instructor to use our auth headers
        if not getattr(Instructor, "_basic_auth_patched", False):
            original_create = Instructor.from_openai if hasattr(Instructor, "from_openai") else None

            if original_create:
                def from_openai_with_auth(*args, **kwargs):
                    # Ensure api_key is set in kwargs
                    if "api_key" not in kwargs:
                        kwargs["api_key"] = basic_token
                    return original_create(*args, **kwargs)

                Instructor.from_openai = staticmethod(from_openai_with_auth)  # type: ignore[assignment]
                setattr(Instructor, "_basic_auth_patched", True)
    except ImportError:
        # Instructor not installed or not used
        pass

    return basic_token


# Initialize authentication on module import
API_KEY = initialize_basic_auth()
