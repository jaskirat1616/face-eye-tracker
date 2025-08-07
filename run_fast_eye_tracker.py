#!/usr/bin/env python3
"""
Fast Eye Tracker Pro Launcher
=============================

Optimized launcher for the professional eye tracking application.
"""

import sys
import os
import warnings
import logging

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Suppress MediaPipe and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '3'

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
import sys; sys.path.append("face-eye-tracker"); from main import main

if __name__ == "__main__":
    print("🚀 Fast Eye Tracker Pro")
    print("⚡ Optimized for maximum performance")
    print("📊 Real-time fatigue detection")
    print("=" * 50)
    
    sys.exit(main()) 