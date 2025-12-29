#!/usr/bin/env python3
"""Analyze CORE Body Temp data and extract representative metrics."""

import os
import sys
import json
import re
from statistics import mean, stdev
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after env loading
from mcp_auth_wrapper import MetaMCPAdapter


def extract_stream_values(stream_response: str, stream_name: str) -> list:
    """Extract numeric values from stream response string."""
    # Look for the stream section
    pattern = rf"Stream:.*?\({stream_name}\).*?First 5 values: \[(.*?)\].*?Last 5 values: \[(.*?)\]"
    match = re.search(pattern, stream_response, re.DOTALL)

    if not match:
        return []

    first_vals = match.group(1)
    last_vals = match.group(2)

    # Parse the values
    def parse_values(val_str):
        vals = []
        for v in val_str.split(","):
            v = v.strip()
            if v and v != "None":
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    first = parse_values(first_vals)
    last = parse_values(last_vals)

    # Get data points count
    points_match = re.search(
        rf"Stream:.*?\({stream_name}\).*?Data Points: (\d+)", stream_response, re.DOTALL
    )
    total_points = int(points_match.group(1)) if points_match else 0

    return {
        "first_values": first,
        "last_values": last,
        "total_points": total_points,
        "start": first[0] if first else None,
        "end": last[-1] if last else None,
    }


def analyze_core_data(activity_id: str):
    """Analyze CORE temperature data and provide summary."""

    # Configuration
    mcp_url = os.getenv("INTERVALS_MCP_SERVER_URL", "")
    mcp_key = os.getenv("MCP_API_KEY", "")

    if not mcp_url or not mcp_key:
        print("❌ Missing configuration", file=sys.stderr)
        return None

    print(f"🔗 Analyzing CORE data for activity {activity_id}...\n", file=sys.stderr)

    try:
        # Connect to MCP
        adapter = MetaMCPAdapter(mcp_url, mcp_key, connect_timeout=30)
        adapter.start()

        # Find streams tool
        streams_tool = None
        for tool in adapter.tools:
            if "stream" in tool.name.lower():
                streams_tool = tool
                break

        if not streams_tool:
            print("❌ Streams tool not found", file=sys.stderr)
            adapter.stop()
            return None

        # Request CORE streams
        print("📞 Fetching CORE temperature streams...\n", file=sys.stderr)

        core_response = streams_tool._run(
            activity_id=activity_id,
            stream_types="core_temperature,skin_temperature,heartrate,watts",
        )

        if (
            not isinstance(core_response, str)
            or "core_temperature" not in core_response
        ):
            print("⚠️  No CORE data available for this activity", file=sys.stderr)
            adapter.stop()
            return None

        # Extract core temp data
        core_temp = extract_stream_values(core_response, "core_temperature")
        skin_temp = extract_stream_values(core_response, "skin_temperature")

        adapter.stop()

        print("=" * 80, file=sys.stderr)
        print("📊 CORE TEMPERATURE ANALYSIS", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

        if core_temp["start"] is None:
            print("⚠️  No valid CORE data", file=sys.stderr)
            return None

        # Calculate statistics
        start_temp = core_temp["start"]
        end_temp = core_temp["end"]
        temp_rise = end_temp - start_temp

        # Estimate max (using last values as proxy)
        max_temp = max(core_temp["last_values"])

        print(f"Core Temperature:", file=sys.stderr)
        print(f"  Start: {start_temp:.1f}°C", file=sys.stderr)
        print(f"  End: {end_temp:.1f}°C", file=sys.stderr)
        print(f"  Max (estimated): {max_temp:.1f}°C", file=sys.stderr)
        print(f"  Rise: +{temp_rise:.1f}°C", file=sys.stderr)
        print(
            f"  Duration: {core_temp['total_points']} seconds ({core_temp['total_points']//60} min)\n",
            file=sys.stderr,
        )

        if skin_temp["start"]:
            print(f"Skin Temperature:", file=sys.stderr)
            print(f"  Start: {skin_temp['start']:.1f}°C", file=sys.stderr)
            print(f"  End: {skin_temp['end']:.1f}°C", file=sys.stderr)
            print(
                f"  Change: {skin_temp['end'] - skin_temp['start']:.1f}°C\n",
                file=sys.stderr,
            )

        # Determine most representative metric
        print("=" * 80, file=sys.stderr)
        print("🎯 MOST REPRESENTATIVE METRIC FOR DESCRIPTION", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

        # Decision logic
        if max_temp >= 39.5:
            metric = f"Core temp max {max_temp:.1f}°C (zone critique)"
            importance = "HIGH"
            print(f"   🚨 {metric}", file=sys.stderr)
            print(f"      Raison: Température critique atteinte", file=sys.stderr)

        elif max_temp >= 39.0:
            metric = f"Core temp max {max_temp:.1f}°C"
            importance = "HIGH"
            print(f"   ⚠️  {metric}", file=sys.stderr)
            print(
                f"      Raison: Température élevée, stress thermique important",
                file=sys.stderr,
            )

        elif temp_rise >= 1.5:
            metric = f"Élévation core temp +{temp_rise:.1f}°C (max {max_temp:.1f}°C)"
            importance = "MEDIUM"
            print(f"   📈 {metric}", file=sys.stderr)
            print(f"      Raison: Forte élévation pendant l'effort", file=sys.stderr)

        elif max_temp >= 38.5:
            metric = f"Core temp bien contrôlée (max {max_temp:.1f}°C)"
            importance = "LOW"
            print(f"   ✅ {metric}", file=sys.stderr)
            print(f"      Raison: Température normale pour l'effort", file=sys.stderr)

        else:
            metric = f"Core temp stable (max {max_temp:.1f}°C)"
            importance = "LOW"
            print(f"   ✅ {metric}", file=sys.stderr)
            print(f"      Raison: Pas de stress thermique", file=sys.stderr)

        # Return summary
        summary = {
            "activity_id": activity_id,
            "core_temp_start": start_temp,
            "core_temp_end": end_temp,
            "core_temp_max": max_temp,
            "core_temp_rise": temp_rise,
            "skin_temp_start": skin_temp["start"],
            "skin_temp_end": skin_temp["end"],
            "duration_seconds": core_temp["total_points"],
            "recommended_metric": metric,
            "importance": importance,
        }

        print("\n" + "=" * 80, file=sys.stderr)
        print("📋 SUMMARY FOR INTEGRATION", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

        print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)

        return summary

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None


if __name__ == "__main__":
    # Test with the activity from input.json
    activity_id = "i107537962"

    if len(sys.argv) > 1:
        activity_id = sys.argv[1]

    result = analyze_core_data(activity_id)

    if result:
        print("\n✅ Analysis complete!", file=sys.stderr)
        print(f"\n💡 Recommended description addition:", file=sys.stderr)
        print(f"   {result['recommended_metric']}", file=sys.stderr)
        sys.exit(0)
    else:
        print("\n❌ Analysis failed", file=sys.stderr)
        sys.exit(1)
