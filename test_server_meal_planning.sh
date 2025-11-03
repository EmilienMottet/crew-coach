#!/bin/bash
# Test script for the meal planning server endpoint

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🍽️  Testing Meal Planning Server Endpoint"
echo "=========================================="
echo ""

# Check if server is running
echo "⏳ Checking if server is running..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not running at http://localhost:8000${NC}"
    echo ""
    echo "Start the server with:"
    echo "  python server.py"
    echo ""
    exit 1
fi

echo ""

# Test 1: Default meal plan (next Monday)
echo "=========================================="
echo "Test 1: Generate meal plan (default date)"
echo "=========================================="
echo ""

echo "⏳ Sending request to /meal-plan with empty body..."
response=$(curl -s -w "\n%{http_code}" -X POST \
  http://localhost:8000/meal-plan \
  -H "Content-Type: application/json" \
  -d '{}')

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo ""
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✅ Request successful (HTTP $http_code)${NC}"
    echo ""

    # Parse response
    week_start=$(echo "$body" | jq -r '.week_start_date')
    total_meals=$(echo "$body" | jq -r '.integration.total_meals_created // 0')
    approved=$(echo "$body" | jq -r '.validation.approved // false')
    error=$(echo "$body" | jq -r '.error // "none"')

    echo "📊 Results:"
    echo "  Week start: $week_start"
    echo "  Total meals: $total_meals"
    echo "  Validation approved: $approved"

    if [ "$error" != "none" ] && [ "$error" != "null" ]; then
        echo -e "  ${RED}Error: $error${NC}"
    fi

    echo ""
    echo "📝 Full response (first 500 chars):"
    echo "$body" | jq '.' | head -c 500
    echo "..."
else
    echo -e "${RED}❌ Request failed (HTTP $http_code)${NC}"
    echo ""
    echo "Response:"
    echo "$body" | jq '.' || echo "$body"
    exit 1
fi

echo ""
echo ""

# Test 2: Specific date
echo "=========================================="
echo "Test 2: Generate meal plan (specific date)"
echo "=========================================="
echo ""

# Calculate next Monday
next_monday=$(date -d "next Monday" +%Y-%m-%d)
echo "⏳ Sending request for week starting: $next_monday..."

response=$(curl -s -w "\n%{http_code}" -X POST \
  http://localhost:8000/meal-plan \
  -H "Content-Type: application/json" \
  -d "{\"week_start_date\": \"$next_monday\"}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo ""
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✅ Request successful (HTTP $http_code)${NC}"
    echo ""

    week_start=$(echo "$body" | jq -r '.week_start_date')
    if [ "$week_start" = "$next_monday" ]; then
        echo -e "${GREEN}✅ Week start date matches request${NC}"
    else
        echo -e "${YELLOW}⚠️  Week start date mismatch: expected $next_monday, got $week_start${NC}"
    fi
else
    echo -e "${RED}❌ Request failed (HTTP $http_code)${NC}"
    echo ""
    echo "Response:"
    echo "$body" | jq '.' || echo "$body"
    exit 1
fi

echo ""
echo ""

# Test 3: Invalid date format
echo "=========================================="
echo "Test 3: Invalid date format (should fail)"
echo "=========================================="
echo ""

echo "⏳ Sending request with invalid date..."
response=$(curl -s -w "\n%{http_code}" -X POST \
  http://localhost:8000/meal-plan \
  -H "Content-Type: application/json" \
  -d '{"week_start_date": "invalid-date"}')

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo ""
if [ "$http_code" = "400" ]; then
    echo -e "${GREEN}✅ Correctly rejected invalid date (HTTP $http_code)${NC}"
    echo ""
    echo "Error message:"
    echo "$body" | jq -r '.detail'
else
    echo -e "${YELLOW}⚠️  Expected HTTP 400, got $http_code${NC}"
    echo ""
    echo "Response:"
    echo "$body" | jq '.' || echo "$body"
fi

echo ""
echo ""
echo "=========================================="
echo -e "${GREEN}✅ All tests completed!${NC}"
echo "=========================================="
