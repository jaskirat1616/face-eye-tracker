#!/usr/bin/env python3
"""
Advanced Eye Tracking & Cognitive Load Detection System Launcher
===============================================================

Professional research-grade launcher for the advanced eye tracking and 
cognitive load detection application with enhanced accuracy and features.
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

# Add the parent directory to the path to allow importing face_eye_tracker module
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import and run the main application
try:
    from face_eye_tracker.main import main
except ImportError:
    # Fallback to changing directory to the nested folder and importing
    import subprocess
    os.chdir(os.path.join(parent_dir, "face-eye-tracker"))
    sys.path.insert(0, os.path.join(parent_dir, "face-eye-tracker"))
    from main import main

if __name__ == "__main__":
    print("🔬 Advanced Eye Tracking & Cognitive Load Detection System")
    print("=" * 70)
    print("📊 Professional Research-Grade Eye Tracking")
    print("🧠 Cognitive Load Assessment & Fatigue Detection")
    print("📈 Real-time Analytics & Export Capabilities")
    print("🎯 Advanced Calibration & Quality Monitoring")
    print("=" * 70)
    print()
    
    # Check for UI argument to customize the startup message
    import sys
    ui_mode = "research"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--ui" and i + 1 < len(sys.argv):
            ui_mode = sys.argv[i + 1]
            break
    
    if ui_mode == "research":
        print("🚀 Starting in Research Mode (recommended for professional use)")
        print("💡 Use --ui simple for basic mode or --help for all options")
    else:
        print(f"🚀 Starting in {ui_mode.title()} Mode")
        print("💡 Use --ui research for full research features or --help for all options")
    
    print("=" * 70)
    
    sys.exit(main()) 