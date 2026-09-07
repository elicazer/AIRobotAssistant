#!/bin/bash
# Refresh AWS STS credentials via ada and update .env
#
# Usage: ./scripts/refresh_aws_creds.sh [account_id] [role_name]
#
# Defaults to the account/role configured below.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# Default account and role — change these to match your setup
AWS_ACCOUNT="${1:-183531038514}"
AWS_ROLE="${2:-Admin}"

echo "🔑 Refreshing AWS credentials..."
echo "   Account: $AWS_ACCOUNT"
echo "   Role: $AWS_ROLE"

# Get credentials via ada
CREDS=$(ada credentials update --account "$AWS_ACCOUNT" --role "$AWS_ROLE" --output json 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$CREDS" ]; then
    echo "❌ Failed to get credentials via ada. Trying with --once flag..."
    CREDS=$(ada credentials update --account "$AWS_ACCOUNT" --role "$AWS_ROLE" --once --output json 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$CREDS" ]; then
        echo "❌ ada credentials failed. Make sure you're authenticated:"
        echo "   ada credentials update --account $AWS_ACCOUNT --role $AWS_ROLE"
        exit 1
    fi
fi

# Parse credentials
ACCESS_KEY=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKeyId'])" 2>/dev/null)
SECRET_KEY=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['SecretAccessKey'])" 2>/dev/null)
SESSION_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['SessionToken'])" 2>/dev/null)

if [ -z "$ACCESS_KEY" ] || [ -z "$SECRET_KEY" ] || [ -z "$SESSION_TOKEN" ]; then
    # Try alternate format (ada might output differently)
    ACCESS_KEY=$(echo "$CREDS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Credentials',d).get('AccessKeyId',''))" 2>/dev/null)
    SECRET_KEY=$(echo "$CREDS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Credentials',d).get('SecretAccessKey',''))" 2>/dev/null)
    SESSION_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Credentials',d).get('SessionToken',''))" 2>/dev/null)
fi

if [ -z "$ACCESS_KEY" ]; then
    echo "❌ Could not parse credentials from ada output"
    echo "   Raw output: $CREDS"
    exit 1
fi

# Read existing .env and update AWS credentials
if [ -f "$ENV_FILE" ]; then
    # Remove old AWS credential lines
    grep -v '^export AWS_ACCESS_KEY_ID=' "$ENV_FILE" | \
    grep -v '^export AWS_SECRET_ACCESS_KEY=' | \
    grep -v '^export AWS_SESSION_TOKEN=' > "$ENV_FILE.tmp"
else
    touch "$ENV_FILE.tmp"
fi

# Append fresh credentials
echo "export AWS_ACCESS_KEY_ID=$ACCESS_KEY" >> "$ENV_FILE.tmp"
echo "export AWS_SECRET_ACCESS_KEY=$SECRET_KEY" >> "$ENV_FILE.tmp"
echo "export AWS_SESSION_TOKEN=$SESSION_TOKEN" >> "$ENV_FILE.tmp"

mv "$ENV_FILE.tmp" "$ENV_FILE"

echo "✅ Credentials updated in .env"
echo "   Access Key: ${ACCESS_KEY:0:8}..."
echo "   Expires: ~12 hours from now"
echo ""
echo "   Run: source .env && ./start.sh"
