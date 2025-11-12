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


def patch_litellm_for_tools() -> None:
    """
    Monkey-patch litellm.completion to log when tools are passed.
    
    This helps debug whether CrewAI's LLM.call() is actually passing
    the tools parameter to the underlying LiteLLM completion call.
    """
    import sys
    import litellm
    from functools import wraps
    
    if getattr(litellm.completion, "_tools_logging_patched", False):
        return  # Already patched
    
    # Store original completion function
    _original_completion = litellm.completion
    
    @wraps(_original_completion)
    def _completion_with_tool_logging(*args, **kwargs):
        """Wrapper that logs when tools are passed to completion."""
        # Check if tools are present
        tools = kwargs.get('tools')
        if tools and isinstance(tools, list) and len(tools) > 0:
            print(
                f"🛠️  LiteLLM completion called with {len(tools)} tools\n"
                f"   Model: {kwargs.get('model', 'unknown')}\n"
                f"   Tool names: {[t.get('function', {}).get('name', 'unnamed') for t in tools[:5]]}\n",
                file=sys.stderr
            )
        else:
            # Log that NO tools were passed
            print(
                f"⚠️  LiteLLM completion called WITHOUT tools\n"
                f"   Model: {kwargs.get('model', 'unknown')}\n",
                file=sys.stderr
            )
        
        return _original_completion(*args, **kwargs)
    
    # Apply the patch
    litellm.completion = _completion_with_tool_logging
    setattr(litellm.completion, "_tools_logging_patched", True)
    
    print("✅ LiteLLM completion patched for tool logging\n", file=sys.stderr)


def patch_crewai_llm_call() -> None:
    """
    Monkey-patch CrewAI's LLM.call() to pass tools to litellm.completion.
    
    CrewAI's LLM.call() ignores the tools parameter completely.
    This patch intercepts the call and passes tools to litellm when present.
    """
    import sys
    
    try:
        from crewai.llm import LLM
        import litellm
    except ImportError:
        print("⚠️  Could not import CrewAI LLM for patching\n", file=sys.stderr)
        return
    
    if getattr(LLM, "_tool_calling_patched", False):
        return  # Already patched
    
    # Store original call method
    _original_call = LLM.call
    
    def _call_with_tools(self, messages, tools=None, **kwargs):
        """Patched LLM.call that passes tools to litellm.completion."""
        # Extract tools from agent if not provided
        if not tools and 'from_agent' in kwargs:
            agent = kwargs.get('from_agent')
            if hasattr(agent, 'tools') and agent.tools:
                # Convert CrewAI tools to OpenAI function calling format
                converted_tools = []
                for tool in agent.tools:
                    try:
                        # Try to_function() first
                        if hasattr(tool, 'to_function') and callable(tool.to_function):
                            func_def = tool.to_function()
                            converted_tools.append(func_def)
                        # Fall back to manual conversion
                        elif hasattr(tool, 'args_schema'):
                            tool_def = {
                                "type": "function",
                                "function": {
                                    "name": getattr(tool, 'name', str(tool)),
                                    "description": getattr(tool, 'description', ''),
                                    "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {}
                                }
                            }
                            converted_tools.append(tool_def)
                    except Exception:
                        pass  # Skip tools that can't be converted
                
                if converted_tools:
                    tools = converted_tools
                    print(
                        f"   🔧 Patched LLM.call extracted {len(tools)} tools from agent\n",
                        file=sys.stderr
                    )
        
        # If we have tools, call litellm.completion directly
        if tools and len(tools) > 0:
            print(
                f"   🚀 Calling litellm.completion with {len(tools)} tools\n",
                file=sys.stderr
            )
            
            # Prepare messages
            if isinstance(messages, str):
                formatted_messages = [{"role": "user", "content": messages}]
            else:
                formatted_messages = messages
            
            # Call litellm directly
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=formatted_messages,
                    tools=tools,
                    api_base=getattr(self, 'base_url', None),
                    api_key=getattr(self, 'api_key', None),
                    drop_params=True,  # Ignore unsupported params
                    **{k: v for k, v in kwargs.items() if k not in ['from_agent', 'from_task', 'tools', 'available_functions']}
                )
                
                # Return the content or function call
                choice = response.choices[0]
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    print(
                        f"   ✅ LLM returned {len(choice.message.tool_calls)} tool calls!\n",
                        file=sys.stderr
                    )
                
                return choice.message.content or str(choice.message)
                
            except Exception as e:
                print(f"   ⚠️  litellm.completion with tools failed: {e}\n", file=sys.stderr)
                # Fall back to original call
                return _original_call(self, messages, **kwargs)
        
        # No tools, use original call
        return _original_call(self, messages, **kwargs)
    
    # Apply the patch
    LLM.call = _call_with_tools  # type: ignore[assignment]
    setattr(LLM, "_tool_calling_patched", True)
    
    print("✅ CrewAI LLM.call() patched for tool calling support\n", file=sys.stderr)


# Initialize authentication on module import
API_KEY = initialize_basic_auth()

# Apply litellm.completion patch
patch_litellm_for_tools()

# Apply CrewAI LLM.call() patch
patch_crewai_llm_call()
