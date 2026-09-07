#!/bin/bash
# Stop InMoov Robot Voice Assistant

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.robot.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Stopping voice assistant (PID: $PID)..."
        kill -INT "$PID"
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "✅ Stopped"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 0.5
        done
        echo "⚠️  Force killing..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$PID_FILE"
    else
        echo "Process $PID not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Checking port 8080..."
    PID=$(lsof -ti :8080 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "🛑 Killing process on port 8080 (PID: $PID)..."
        kill -INT $PID 2>/dev/null
        sleep 2
        kill -9 $PID 2>/dev/null
        echo "✅ Stopped"
    else
        echo "Nothing running."
    fi
fi
