"""Main Crew definition for Strava activity description generation."""
import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv
from crewai import Crew, Process, LLM
import litellm

from agents import create_description_agent, create_privacy_agent, create_translation_agent
from tasks import create_description_task, create_privacy_task, create_translation_task
from tools import (
    get_intervals_activity_details,
    get_intervals_activity_intervals,
    get_recent_intervals_activities
)


class StravaDescriptionCrew:
    """Crew for generating and validating Strava activity descriptions."""
    
    def __init__(self):
        """Initialize the crew with LLM and agents."""
        load_dotenv()
        
        # Configure environment variables for LiteLLM/OpenAI
        base_url = os.getenv("OPENAI_API_BASE", "https://ghcopilot.emottet.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5mini")
        
        # Configure authentication - support both API key and Basic Auth
        auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")  # Base64-encoded Basic Auth
        
        # Set environment variables that LiteLLM expects
        os.environ["OPENAI_API_BASE"] = base_url
        
        # Configure litellm to add custom headers globally
        if auth_token:
            # Configure litellm to use Basic Auth
            # We need to monkey-patch the completion function to add the header
            litellm.add_function_to_prompt = False
            # Set a dummy API key (required by litellm but won't be used)
            os.environ["OPENAI_API_KEY"] = "not-used"
            
            # Monkey patch to add Basic Auth header
            original_completion = litellm.completion
            
            def completion_with_basic_auth(*args, **kwargs):
                # Add custom headers
                if "extra_headers" not in kwargs:
                    kwargs["extra_headers"] = {}
                kwargs["extra_headers"]["Authorization"] = f"Basic {auth_token}"
                # Remove the api_key parameter to prevent Bearer auth
                if "api_key" in kwargs:
                    del kwargs["api_key"]
                return original_completion(*args, **kwargs)
            
            litellm.completion = completion_with_basic_auth
            api_key = "not-used"
        else:
            # Use standard API key
            api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Create LLM instance
        self.llm = LLM(
            model=f"openai/{model_name}",
            api_base=base_url,
            api_key=api_key,
            drop_params=True,
            additional_drop_params=["stop"]
        )
        
        # Prepare tools for description agent
        description_tools = [
            get_intervals_activity_details,
            get_intervals_activity_intervals,
            get_recent_intervals_activities
        ]
        
        # Create agents
        self.description_agent = create_description_agent(self.llm, description_tools)
        self.privacy_agent = create_privacy_agent(self.llm)
        self.translation_agent = create_translation_agent(self.llm)
    
    def process_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a Strava activity webhook to generate title and description.
        
        Args:
            activity_data: The webhook payload from Strava (n8n input)
            
        Returns:
            Dictionary containing the final title, description, and privacy settings
        """
        # Step 1: Generate title and description
        print("\n🚀 Step 1: Generating activity description...\n", file=sys.stderr)
        
        description_task = create_description_task(
            self.description_agent,
            activity_data
        )
        
        description_crew = Crew(
            agents=[self.description_agent],
            tasks=[description_task],
            process=Process.sequential,
            verbose=True
        )
        
        description_result = description_crew.kickoff()
        
        # Parse the result
        try:
            generated_content = json.loads(str(description_result))
        except json.JSONDecodeError:
            # If not valid JSON, create a fallback structure
            result_str = str(description_result)
            generated_content = {
                "title": "Activity completed",
                "description": result_str[:500] if len(result_str) > 500 else result_str,
                "workout_type": "Unknown",
                "key_metrics": {}
            }
        
        print(f"\n✅ Generated content:\n{json.dumps(generated_content, indent=2)}\n", file=sys.stderr)
        
        # Step 2: Check privacy and compliance
        print("\n🔒 Step 2: Checking privacy and compliance...\n", file=sys.stderr)
        
        privacy_task = create_privacy_task(
            self.privacy_agent,
            activity_data,
            json.dumps(generated_content, indent=2)
        )
        
        privacy_crew = Crew(
            agents=[self.privacy_agent],
            tasks=[privacy_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Wrap the privacy crew execution in a try-except to catch agent errors
        try:
            privacy_result = privacy_crew.kickoff()
        except Exception as e:
            print(f"\n❌ Privacy crew execution failed: {str(e)}\n", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Create a safe default result
            privacy_check = {
                "privacy_approved": False,
                "during_work_hours": True,
                "should_be_private": True,
                "issues_found": [f"Privacy agent execution failed: {str(e)}"],
                "recommended_changes": {
                    "title": generated_content.get("title") if isinstance(generated_content, dict) else "Activity",
                    "description": generated_content.get("description") if isinstance(generated_content, dict) else ""
                },
                "reasoning": f"Privacy agent failed to execute: {str(e)}"
            }
            # Skip to result assembly
            recommended_changes = {}
            final_title = generated_content.get("title", "Activity") if isinstance(generated_content, dict) else "Activity"
            final_description = generated_content.get("description", "") if isinstance(generated_content, dict) else ""
        else:
            # Privacy crew executed successfully, parse the result
            print(f"\n🔍 DEBUG: privacy_result type: {type(privacy_result)}\n", file=sys.stderr)
            print(f"\n🔍 DEBUG: privacy_result repr: {repr(privacy_result)}\n", file=sys.stderr)
            print(f"\n🔍 DEBUG: privacy_result dir: {dir(privacy_result)}\n", file=sys.stderr)
            
            # Parse privacy result - handle both CrewOutput objects and strings
            privacy_check = None
            try:
                # Try to access different possible attributes of CrewOutput
                if hasattr(privacy_result, 'raw'):
                    privacy_result_str = str(privacy_result.raw)
                    print(f"\n🔍 DEBUG: Using privacy_result.raw: {privacy_result_str}\n", file=sys.stderr)
                elif hasattr(privacy_result, 'result'):
                    privacy_result_str = str(privacy_result.result)
                    print(f"\n🔍 DEBUG: Using privacy_result.result: {privacy_result_str}\n", file=sys.stderr)
                elif hasattr(privacy_result, 'output'):
                    privacy_result_str = str(privacy_result.output)
                    print(f"\n🔍 DEBUG: Using privacy_result.output: {privacy_result_str}\n", file=sys.stderr)
                else:
                    privacy_result_str = str(privacy_result)
                    print(f"\n🔍 DEBUG: Using str(privacy_result): {privacy_result_str}\n", file=sys.stderr)
                
                # Strip markdown code block markers if present
                privacy_result_str = privacy_result_str.strip()
                if privacy_result_str.startswith("```json"):
                    privacy_result_str = privacy_result_str[7:]  # Remove ```json
                if privacy_result_str.startswith("```"):
                    privacy_result_str = privacy_result_str[3:]  # Remove ```
                if privacy_result_str.endswith("```"):
                    privacy_result_str = privacy_result_str[:-3]  # Remove trailing ```
                privacy_result_str = privacy_result_str.strip()
                
                print(f"\n🔍 DEBUG: Cleaned JSON string: {privacy_result_str[:200]}...\n", file=sys.stderr)
                
                privacy_check = json.loads(privacy_result_str)
                print(f"\n🔍 DEBUG: Successfully parsed privacy_check\n", file=sys.stderr)
            except (json.JSONDecodeError, Exception) as e:
                # Default to safe settings if parsing fails
                print(f"\n⚠️  Warning: Could not parse privacy result: {str(e)}\n", file=sys.stderr)
                print(f"⚠️  Exception type: {type(e)}\n", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                privacy_check = {
                    "privacy_approved": False,
                    "during_work_hours": True,
                    "should_be_private": True,
                    "issues_found": [f"Could not parse privacy check: {str(e)}"],
                    "recommended_changes": {
                        "title": generated_content.get("title") if isinstance(generated_content, dict) else "Activity",
                        "description": generated_content.get("description") if isinstance(generated_content, dict) else ""
                    },
                    "reasoning": "Privacy check failed to parse, using safe defaults"
                }
            
            # Ensure privacy_check is a dict (defense in depth)
            if not isinstance(privacy_check, dict):
                print(f"\n⚠️  Warning: privacy_check is not a dict, got {type(privacy_check)}: {privacy_check}\n", file=sys.stderr)
                privacy_check = {
                    "privacy_approved": False,
                    "during_work_hours": True,
                    "should_be_private": True,
                    "issues_found": ["Invalid privacy check format - defaulting to private"],
                    "recommended_changes": {
                        "title": generated_content.get("title") if isinstance(generated_content, dict) else "Activity",
                        "description": generated_content.get("description") if isinstance(generated_content, dict) else ""
                    },
                    "reasoning": "Privacy check returned invalid format, using safe defaults"
                }
            
            print(f"\n✅ Privacy check result:\n{json.dumps(privacy_check, indent=2)}\n", file=sys.stderr)
            
            # Step 3: Combine results - safely access nested values with explicit type checking
            # Extra safety: ensure privacy_check is truly a dict before accessing
            if not isinstance(privacy_check, dict):
                print(f"\n⚠️  CRITICAL: privacy_check is still not a dict after parsing! Type: {type(privacy_check)}, Value: {privacy_check}\n", file=sys.stderr)
                privacy_check = {
                    "privacy_approved": False,
                    "during_work_hours": True,
                    "should_be_private": True,
                    "issues_found": [f"Privacy check type error: expected dict, got {type(privacy_check).__name__}"],
                    "recommended_changes": {
                        "title": generated_content.get("title") if isinstance(generated_content, dict) else "Activity",
                        "description": generated_content.get("description") if isinstance(generated_content, dict) else ""
                    },
                    "reasoning": "Privacy check type error - defaulting to safe private settings"
                }
            
            recommended_changes = privacy_check.get("recommended_changes", {})
            if not isinstance(recommended_changes, dict):
                recommended_changes = {}
            
            final_title = recommended_changes.get("title") or (generated_content.get("title", "Activity") if isinstance(generated_content, dict) else "Activity")
            final_description = recommended_changes.get("description") or (generated_content.get("description", "") if isinstance(generated_content, dict) else "")
        
        # Step 3: Translate if enabled
        print("\n🌐 Step 3: Translating content (if enabled)...\n", file=sys.stderr)
        
        translation_enabled = os.getenv("TRANSLATION_ENABLED", "false").lower() == "true"
        
        if translation_enabled:
            # Prepare content for translation
            content_to_translate = {
                "title": final_title,
                "description": final_description,
                "workout_type": generated_content.get("workout_type", "Unknown") if isinstance(generated_content, dict) else "Unknown",
                "key_metrics": generated_content.get("key_metrics", {}) if isinstance(generated_content, dict) else {}
            }
            
            translation_task = create_translation_task(
                self.translation_agent,
                json.dumps(content_to_translate, indent=2)
            )
            
            translation_crew = Crew(
                agents=[self.translation_agent],
                tasks=[translation_task],
                process=Process.sequential,
                verbose=True
            )
            
            try:
                translation_result = translation_crew.kickoff()
                
                # Parse translation result
                translation_result_str = str(translation_result)
                
                # Strip markdown code block markers if present
                translation_result_str = translation_result_str.strip()
                if translation_result_str.startswith("```json"):
                    translation_result_str = translation_result_str[7:]
                if translation_result_str.startswith("```"):
                    translation_result_str = translation_result_str[3:]
                if translation_result_str.endswith("```"):
                    translation_result_str = translation_result_str[:-3]
                translation_result_str = translation_result_str.strip()
                
                translated_content = json.loads(translation_result_str)
                
                # Use translated title and description
                final_title = translated_content.get("title", final_title)
                final_description = translated_content.get("description", final_description)
                
                print(f"\n✅ Translation successful:\nTitle: {final_title}\nDescription: {final_description[:100]}...\n", file=sys.stderr)
                
            except Exception as e:
                print(f"\n⚠️ Warning: Translation failed, using original content: {str(e)}\n", file=sys.stderr)
                # Keep original content if translation fails
        else:
            print("\n⏭️  Translation disabled, skipping...\n", file=sys.stderr)
        
        # Step 4: Combine results
        final_result = {
            "activity_id": activity_data.get("object_data", {}).get("id") if isinstance(activity_data.get("object_data"), dict) else None,
            "title": final_title,
            "description": final_description,
            "should_be_private": privacy_check.get("should_be_private", True) if isinstance(privacy_check, dict) else True,
            "privacy_check": {
                "approved": privacy_check.get("privacy_approved", False) if isinstance(privacy_check, dict) else False,
                "during_work_hours": privacy_check.get("during_work_hours", False) if isinstance(privacy_check, dict) else False,
                "issues": privacy_check.get("issues_found", []) if isinstance(privacy_check, dict) else ["Type error in privacy check"],
                "reasoning": privacy_check.get("reasoning", "Error in privacy processing") if isinstance(privacy_check, dict) else "Error in privacy processing"
            },
            "workout_analysis": {
                "type": generated_content.get("workout_type", "Unknown") if isinstance(generated_content, dict) else "Unknown",
                "metrics": generated_content.get("key_metrics", {}) if isinstance(generated_content, dict) else {}
            }
        }
        
        return final_result


def main():
    """Main entry point for n8n integration."""
    # Read input from stdin (n8n will provide this)
    input_data = sys.stdin.read()
    
    try:
        activity_data = json.loads(input_data)
        
        # Handle array input (from n8n)
        if isinstance(activity_data, list) and len(activity_data) > 0:
            activity_data = activity_data[0]
        
        # Process the activity
        crew = StravaDescriptionCrew()
        result = crew.process_activity(activity_data)
        
        # Output result as JSON to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # Log error to stderr with full traceback
        print(f"\n❌ Error: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Output error result to stdout
        error_result = {
            "error": str(e),
            "activity_id": None,
            "title": "Error processing activity",
            "description": "An error occurred while generating the description",
            "should_be_private": True,
            "privacy_check": {
                "approved": False,
                "during_work_hours": False,
                "issues": [str(e)],
                "reasoning": "Error during processing"
            }
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
