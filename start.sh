#!/bin/bash
# Start InMoov Robot Voice Assistant

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please run: python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Start the application
echo "Starting InMoov Robot Voice Assistant..."
python run.py
