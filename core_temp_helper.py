"""Helper functions for extracting and analyzing CORE Body Temp data.

This module provides utilities to:
1. Extract CORE temperature streams from Intervals.icu
2. Calculate representative metrics
3. Generate description snippets for Strava
"""

import re
from typing import Dict, Optional


def extract_core_metrics_from_streams(streams_response: str) -> Optional[Dict]:
    """
    Extract CORE temperature metrics from activity streams response.

    Args:
        streams_response: Raw response from IntervalsIcu__get_activity_streams

    Returns:
        Dict with core temp metrics or None if no CORE data available
    """
    if not isinstance(streams_response, str):
        return None

    if "core_temperature" not in streams_response:
        return None

    # Extract core temperature values
    core_pattern = r"Stream:.*?\(core_temperature\).*?First 5 values: \[(.*?)\].*?Last 5 values: \[(.*?)\]"
    core_match = re.search(core_pattern, streams_response, re.DOTALL)

    if not core_match:
        return None

    # Parse first and last values
    def parse_temp_values(val_str):
        temps = []
        for v in val_str.split(","):
            v = v.strip()
            if v and v != "None":
                try:
                    temps.append(float(v))
                except ValueError:
                    pass
        return temps

    first_temps = parse_temp_values(core_match.group(1))
    last_temps = parse_temp_values(core_match.group(2))

    if not first_temps or not last_temps:
        return None

    start_temp = first_temps[0]
    end_temp = last_temps[-1]
    max_temp = max(last_temps)  # Approximate max from last values
    temp_rise = end_temp - start_temp

    # Extract skin temperature if available
    skin_start = None
    skin_end = None

    skin_pattern = r"Stream:.*?\(skin_temperature\).*?First 5 values: \[(.*?)\].*?Last 5 values: \[(.*?)\]"
    skin_match = re.search(skin_pattern, streams_response, re.DOTALL)

    if skin_match:
        first_skin = parse_temp_values(skin_match.group(1))
        last_skin = parse_temp_values(skin_match.group(2))
        if first_skin and last_skin:
            skin_start = first_skin[0]
            skin_end = last_skin[-1]

    return {
        "core_temp_start": start_temp,
        "core_temp_end": end_temp,
        "core_temp_max": max_temp,
        "core_temp_rise": temp_rise,
        "skin_temp_start": skin_start,
        "skin_temp_end": skin_end,
    }


def generate_core_description_snippet(
    core_metrics: Dict, ambient_temp: Optional[float] = None
) -> Optional[str]:
    """
    Generate a description snippet about CORE temperature data.

    Args:
        core_metrics: Dict from extract_core_metrics_from_streams()
        ambient_temp: Optional ambient temperature in Celsius

    Returns:
        String to add to activity description, or None if not notable
    """
    if not core_metrics:
        return None

    max_temp = core_metrics["core_temp_max"]
    temp_rise = core_metrics["core_temp_rise"]

    # Decision logic for what to include
    if max_temp >= 39.5:
        # Critical temperature
        return f"Core temp max {max_temp:.1f}°C (zone critique atteinte)"

    elif max_temp >= 39.0:
        # High temperature - important to mention
        return f"Core temp max {max_temp:.1f}°C, stress thermique important"

    elif temp_rise >= 1.5:
        # Significant temperature rise
        return f"Élévation core temp +{temp_rise:.1f}°C (max {max_temp:.1f}°C)"

    elif max_temp >= 38.8:
        # Moderate temperature, mention if hot conditions
        if ambient_temp and ambient_temp > 25:
            return f"Core temp bien gérée malgré chaleur (max {max_temp:.1f}°C)"
        else:
            return f"Thermorégulation optimale (core temp max {max_temp:.1f}°C)"

    else:
        # Normal temperature - only mention if very cold conditions
        if ambient_temp and ambient_temp < 5:
            return f"Thermorégulation parfaite par temps froid (core temp {max_temp:.1f}°C)"
        else:
            # Don't clutter description with normal temperatures
            return None


def should_include_core_in_description(
    core_metrics: Dict, ambient_temp: Optional[float] = None
) -> bool:
    """
    Decide if CORE data is notable enough to include in description.

    Args:
        core_metrics: Dict from extract_core_metrics_from_streams()
        ambient_temp: Optional ambient temperature in Celsius

    Returns:
        True if CORE data should be mentioned in description
    """
    if not core_metrics:
        return False

    max_temp = core_metrics["core_temp_max"]
    temp_rise = core_metrics["core_temp_rise"]

    # Always include if temperature is high or rise is significant
    if max_temp >= 39.0 or temp_rise >= 1.5:
        return True

    # Include if extreme ambient conditions
    if ambient_temp:
        if ambient_temp < 5 or ambient_temp > 28:
            return True

    # Otherwise, not notable enough
    return False


def format_core_metrics_for_llm(
    core_metrics: Optional[Dict], ambient_temp: Optional[float] = None
) -> str:
    """
    Format CORE metrics as a string for LLM context.

    Args:
        core_metrics: Dict from extract_core_metrics_from_streams()
        ambient_temp: Optional ambient temperature

    Returns:
        Formatted string with CORE data context
    """
    if not core_metrics:
        return "CORE Body Temperature: No data available for this activity."

    max_temp = core_metrics["core_temp_max"]
    temp_rise = core_metrics["core_temp_rise"]
    start_temp = core_metrics["core_temp_start"]
    end_temp = core_metrics["core_temp_end"]

    # Build context string
    lines = [
        "CORE Body Temperature Data:",
        f"  - Start: {start_temp:.1f}°C",
        f"  - Max: {max_temp:.1f}°C",
        f"  - End: {end_temp:.1f}°C",
        f"  - Rise: +{temp_rise:.1f}°C",
    ]

    # Add interpretation
    if max_temp >= 39.5:
        lines.append(f"  ⚠️  CRITICAL: Temperature reached critical zone")
    elif max_temp >= 39.0:
        lines.append(f"  🔴 HIGH: Significant thermal stress")
    elif temp_rise >= 1.5:
        lines.append(f"  📈 NOTABLE: Large temperature increase")
    else:
        lines.append(f"  ✅ NORMAL: Temperature well controlled")

    # Add recommendation
    snippet = generate_core_description_snippet(core_metrics, ambient_temp)
    if snippet:
        lines.append(f'\n  💡 Suggested mention: "{snippet}"')

    return "\n".join(lines)


# Example usage for testing
if __name__ == "__main__":
    # Example stream response (simplified)
    example_response = """
    Stream: None (core_temperature)
      Value Type: java.lang.Float
      Data Points: 5324
      First 5 values: [37.32, 37.32, 37.32, 37.32, 37.32]
      Last 5 values: [38.5, 38.5, 38.5, 38.5, 38.5]
    """

    metrics = extract_core_metrics_from_streams(example_response)
    print("Extracted metrics:", metrics)

    if metrics:
        print(
            "\nDescription snippet:",
            generate_core_description_snippet(metrics, ambient_temp=6),
        )
        print(
            "\nShould include?",
            should_include_core_in_description(metrics, ambient_temp=6),
        )
        print("\nFormatted for LLM:")
        print(format_core_metrics_for_llm(metrics, ambient_temp=6))
