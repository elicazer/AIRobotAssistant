#!/bin/bash
# Start InMoov Robot Voice Assistant

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.robot.pid"

# Kill any previous zombie process on port 8080
cleanup_port() {
    local pid=$(lsof -ti :8080 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "⚠️  Killing stale process on port 8080 (PID: $pid)"
        kill $pid 2>/dev/null
        sleep 1
        # Force kill if still alive
        kill -9 $pid 2>/dev/null
    fi
}

# Clean shutdown function
shutdown() {
    echo ""
    echo "🛑 Shutting down..."
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            # Send SIGINT first (triggers Python's KeyboardInterrupt)
            kill -INT "$pid" 2>/dev/null
            # Wait up to 5 seconds for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                echo "⚠️  Force killing process..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    # Final cleanup of port
    cleanup_port
    echo "✅ Shutdown complete"
    exit 0
}

# Trap signals
trap shutdown SIGINT SIGTERM EXIT

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please run: python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Fix Python SSL certificates (macOS Python framework issue)
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null)

# Source .env if it exists (for OPENAI_API_KEY and other non-AWS vars)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Use AWS SSO profile if configured (no need for manual STS tokens)
# Set this to your SSO profile name from ~/.aws/config
export AWS_PROFILE="${AWS_PROFILE:-robot}"

# Check AWS SSO session is valid, login if expired
echo "🔐 Checking AWS credentials..."
if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
    echo "⚠️  AWS SSO session expired. Logging in..."
    aws sso login --profile "$AWS_PROFILE"
    if [ $? -ne 0 ]; then
        echo "❌ AWS SSO login failed. Some features (memory, face recognition) may not work."
    fi
fi

# Fetch OpenAI API key from Secrets Manager (if not already set)
SECRET_NAME="robot/openai-api-key"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "🔑 Fetching OpenAI API key from Secrets Manager..."
    OPENAI_API_KEY=$(aws secretsmanager get-secret-value \
        --secret-id "$SECRET_NAME" \
        --region us-east-1 \
        --profile "$AWS_PROFILE" \
        --query 'SecretString' \
        --output text 2>/dev/null)
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "None" ]; then
        echo "⚠️  OpenAI API key not found in Secrets Manager ($SECRET_NAME)"
        echo ""
        echo "Options:"
        echo "  1) Enter your OpenAI API key now (will be stored in Secrets Manager)"
        echo "  2) Skip (use Nova Sonic voice model instead)"
        echo ""
        read -p "Enter OpenAI API key (or press Enter to skip): " USER_KEY
        
        if [ -n "$USER_KEY" ]; then
            # Store in Secrets Manager
            echo "💾 Storing key in Secrets Manager..."
            aws secretsmanager create-secret \
                --name "$SECRET_NAME" \
                --secret-string "$USER_KEY" \
                --region us-east-1 \
                --profile "$AWS_PROFILE" 2>/dev/null || \
            aws secretsmanager put-secret-value \
                --secret-id "$SECRET_NAME" \
                --secret-string "$USER_KEY" \
                --region us-east-1 \
                --profile "$AWS_PROFILE" 2>/dev/null
            
            if [ $? -eq 0 ]; then
                echo "✅ Key stored in Secrets Manager"
                export OPENAI_API_KEY="$USER_KEY"
            else
                echo "⚠️  Failed to store in Secrets Manager, using key for this session only"
                export OPENAI_API_KEY="$USER_KEY"
            fi
        else
            echo "⏭️  Skipping OpenAI — will use Nova Sonic voice model"
        fi
    else
        export OPENAI_API_KEY
        echo "✅ OpenAI API key loaded from Secrets Manager"
    fi
fi

# Clean up any stale processes from previous runs
cleanup_port

# Remove stale PID file
rm -f "$PID_FILE"

# Start the application
echo "Starting InMoov Robot Voice Assistant..."
python run.py &
APP_PID=$!
echo $APP_PID > "$PID_FILE"

# Wait for the process to finish
wait $APP_PID
EXIT_CODE=$?

# Cleanup
rm -f "$PID_FILE"
cleanup_port
exit $EXIT_CODE
