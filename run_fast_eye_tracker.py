#!/usr/bin/env python3
"""
Eye Tracking & Cognitive Fatigue Detection System Launcher
==========================================================

Optimized launcher for the eye tracking and fatigue detection application.
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
    print("👁️ Eye Tracking & Cognitive Fatigue Detection System")
    print("🔬 Real-time eye tracking and fatigue analysis")
    print("📊 Professional monitoring and data collection")
    print("=" * 60)
    
    sys.exit(main()) 