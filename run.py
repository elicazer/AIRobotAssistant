#!/usr/bin/env python3
"""
Main entry point for InMoov Robot Voice Assistant
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main server
from voice_assistant_server import main

if __name__ == '__main__':
    main()
