#!/bin/bash
# Activate the Python 3.12 virtual environment

echo "🐍 Activating Python 3.12 virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "Python version: $(python --version)"
echo ""
echo "To run the voice assistant server:"
echo "  sudo $(which python) voice_assistant_server.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
