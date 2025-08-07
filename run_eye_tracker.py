#!/usr/bin/env python3
"""
Eye Tracker Launcher
===================

Simple launcher for the modern eye tracking application.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
from face_eye_tracker.main import main

if __name__ == "__main__":
    sys.exit(main()) 