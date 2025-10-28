#!/bin/bash

# Test script for the translation agent
# This script tests the crew with and without translation enabled

echo "=========================================="
echo "Testing Strava Description Crew"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Activating venv..."
    source venv/bin/activate || {
        echo "❌ Error: Could not activate virtual environment"
        echo "Please run: source venv/bin/activate"
        exit 1
    }
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "Please configure your API keys and MCP server URL in .env"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Test 1: Without translation
echo "=========================================="
echo "Test 1: WITHOUT translation"
echo "=========================================="
echo ""

# Temporarily disable translation
export TRANSLATION_ENABLED=false

echo "Running crew with translation disabled..."
cat test_translation.json | python crew.py > output_no_translation.json 2> logs_no_translation.txt

if [ $? -eq 0 ]; then
    echo "✅ Test 1 completed successfully"
    echo ""
    echo "Output saved to: output_no_translation.json"
    echo "Logs saved to: logs_no_translation.txt"
    echo ""
    echo "Generated title (original):"
    jq -r '.title' output_no_translation.json
    echo ""
else
    echo "❌ Test 1 failed"
    echo "Check logs_no_translation.txt for details"
    exit 1
fi

# Test 2: With translation
echo "=========================================="
echo "Test 2: WITH translation (English)"
echo "=========================================="
echo ""

# Enable translation
export TRANSLATION_ENABLED=true
export TRANSLATION_TARGET_LANGUAGE=English
export TRANSLATION_SOURCE_LANGUAGE=French

echo "Running crew with translation enabled..."
cat test_translation.json | python crew.py > output_with_translation.json 2> logs_with_translation.txt

if [ $? -eq 0 ]; then
    echo "✅ Test 2 completed successfully"
    echo ""
    echo "Output saved to: output_with_translation.json"
    echo "Logs saved to: logs_with_translation.txt"
    echo ""
    echo "Generated title (translated):"
    jq -r '.title' output_with_translation.json
    echo ""
else
    echo "❌ Test 2 failed"
    echo "Check logs_with_translation.txt for details"
    exit 1
fi

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Test 1 (No translation):"
echo "  Title: $(jq -r '.title' output_no_translation.json)"
echo ""
echo "Test 2 (With translation):"
echo "  Title: $(jq -r '.title' output_with_translation.json)"
echo ""
echo "Comparison:"
if [ "$(jq -r '.title' output_no_translation.json)" != "$(jq -r '.title' output_with_translation.json)" ]; then
    echo "  ✅ Titles are different (translation applied)"
else
    echo "  ⚠️  Titles are identical (translation may not have worked)"
fi
echo ""
echo "All tests completed!"
echo ""
echo "Files generated:"
echo "  - output_no_translation.json"
echo "  - output_with_translation.json"
echo "  - logs_no_translation.txt"
echo "  - logs_with_translation.txt"
