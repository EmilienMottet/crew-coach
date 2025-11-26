#!/bin/bash
# Script to help set up GitHub secrets for CI/CD
# Usage: ./scripts/setup_github_secrets.sh

echo "=========================================="
echo "GitHub Secrets Setup Helper"
echo "=========================================="
echo ""
echo "GitHub Actions requires the following secrets to be configured:"
echo ""
echo "Repository: emottet/crew (assumed)"
echo "Path: Settings → Secrets and variables → Actions → New repository secret"
echo ""
echo "Required secrets:"
echo "----------------"
echo ""

# Read from .env file
if [ -f .env ]; then
    echo "✅ Found .env file, extracting values..."
    echo ""

    # Extract values
    OPENAI_API_BASE=$(grep "^OPENAI_API_BASE=" .env | cut -d'=' -f2)
    OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2)
    MCP_API_KEY=$(grep "^MCP_API_KEY=" .env | cut -d'=' -f2)
    STRAVA_MCP_SERVER_URL=$(grep "^STRAVA_MCP_SERVER_URL=" .env | cut -d'=' -f2)
    INTERVALS_MCP_SERVER_URL=$(grep "^INTERVALS_MCP_SERVER_URL=" .env | cut -d'=' -f2)
    METEO_MCP_SERVER_URL=$(grep "^METEO_MCP_SERVER_URL=" .env | cut -d'=' -f2)
    TOOLBOX_MCP_SERVER_URL=$(grep "^TOOLBOX_MCP_SERVER_URL=" .env | cut -d'=' -f2)

    echo "1. OPENAI_API_BASE"
    echo "   Value: $OPENAI_API_BASE"
    echo ""

    echo "2. OPENAI_API_KEY"
    echo "   Value: ${OPENAI_API_KEY:0:10}... (masked)"
    echo ""

    echo "3. MCP_API_KEY"
    echo "   Value: ${MCP_API_KEY:0:10}... (masked)"
    echo ""

    echo "4. STRAVA_MCP_SERVER_URL"
    echo "   Value: $STRAVA_MCP_SERVER_URL"
    echo ""

    echo "5. INTERVALS_MCP_SERVER_URL"
    echo "   Value: $INTERVALS_MCP_SERVER_URL"
    echo ""

    echo "6. METEO_MCP_SERVER_URL"
    echo "   Value: $METEO_MCP_SERVER_URL"
    echo ""

    echo "7. TOOLBOX_MCP_SERVER_URL"
    echo "   Value: $TOOLBOX_MCP_SERVER_URL"
    echo ""

    echo "=========================================="
    echo "Next steps:"
    echo "=========================================="
    echo ""
    echo "1. Go to: https://github.com/emottet/crew/settings/secrets/actions"
    echo "2. Click 'New repository secret'"
    echo "3. Add each secret above with its name and value"
    echo ""
    echo "Or use GitHub CLI (gh):"
    echo "----------------------"
    echo ""
    echo "gh secret set OPENAI_API_BASE --body \"$OPENAI_API_BASE\""
    echo "gh secret set OPENAI_API_KEY --body \"$OPENAI_API_KEY\""
    echo "gh secret set MCP_API_KEY --body \"$MCP_API_KEY\""
    echo "gh secret set STRAVA_MCP_SERVER_URL --body \"$STRAVA_MCP_SERVER_URL\""
    echo "gh secret set INTERVALS_MCP_SERVER_URL --body \"$INTERVALS_MCP_SERVER_URL\""
    echo "gh secret set METEO_MCP_SERVER_URL --body \"$METEO_MCP_SERVER_URL\""
    echo "gh secret set TOOLBOX_MCP_SERVER_URL --body \"$TOOLBOX_MCP_SERVER_URL\""
    echo ""
else
    echo "❌ .env file not found!"
    echo ""
    echo "Please create a .env file with the following variables:"
    echo "  OPENAI_API_BASE"
    echo "  OPENAI_API_KEY"
    echo "  MCP_API_KEY"
    echo "  STRAVA_MCP_SERVER_URL"
    echo "  INTERVALS_MCP_SERVER_URL"
    echo "  METEO_MCP_SERVER_URL"
    echo "  TOOLBOX_MCP_SERVER_URL"
    echo ""
fi
