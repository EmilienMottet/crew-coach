#!/bin/bash
# Script to substitute secrets in n8n workflows before deployment
# Usage: ./substitute-n8n-secrets.sh

echo "=========================================="
echo "n8n Secret Substitution"
echo "=========================================="

# Create temporary directory for substituted files
mkdir -p /tmp/n8n-workflows
echo "Created temporary directory: /tmp/n8n-workflows"

# Check required environment variables
MISSING_SECRETS=0
if [ -z "${INTERVALS_ICU_API_KEY}" ]; then
    echo "❌ Error: INTERVALS_ICU_API_KEY is not set"
    MISSING_SECRETS=1
fi
if [ -z "${TELEGRAM_CHAT_ID}" ]; then
    echo "❌ Error: TELEGRAM_CHAT_ID is not set"
    MISSING_SECRETS=1
fi
if [ -z "${TELEGRAM_CREDENTIAL_ID}" ]; then
    echo "❌ Error: TELEGRAM_CREDENTIAL_ID is not set"
    MISSING_SECRETS=1
fi

if [ $MISSING_SECRETS -eq 1 ]; then
    echo "Missing required secrets. Exiting."
    exit 1
fi

# Process all JSON files in n8n/workflows/
count=0
for file in n8n/workflows/*.json; do
    [ -e "$file" ] || continue

    filename=$(basename "$file")
    echo "Processing $filename..."

    # Perform substitutions using sed
    # We use | as delimiter to avoid issues with special characters in secrets
    sed -e "s|YOUR_BASE64_KEY|${INTERVALS_ICU_API_KEY}|g" \
        -e "s|YOUR_TELEGRAM_CHAT_ID|${TELEGRAM_CHAT_ID}|g" \
        -e "s|YOUR_TELEGRAM_CREDENTIAL_ID|${TELEGRAM_CREDENTIAL_ID}|g" \
        "$file" > "/tmp/n8n-workflows/$filename"

    count=$((count + 1))
done

echo "✅ Successfully processed $count workflows."
echo "Substituted files are available in /tmp/n8n-workflows/"
