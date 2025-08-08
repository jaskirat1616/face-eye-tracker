#!/usr/bin/env python3
"""
Advanced Eye Tracking & Cognitive Load Detection Application
===========================================================

A professional, research-grade application for real-time eye tracking, 
fatigue detection, and cognitive load assessment with advanced features.
"""

import sys
import os
import argparse
import warnings
import logging

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Suppress MediaPipe and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '3'

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.core.advanced_tracker import AdvancedEyeTracker
from utils.data_logger import DataLogger
from ui.research_ui import ResearchEyeTrackerUI
from ui.comprehensive_ui import ComprehensiveEyeTrackerUI
from ui.simple_ui import SimpleEyeTrackerUI
from ui.modern_ui import ModernEyeTrackerUI
from ui.headless_ui import HeadlessUI

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('tkinter', 'tkinter'),
        ('scipy', 'scipy'),
        ('pillow', 'PIL')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_file():
    """Check if the MediaPipe model file exists"""
    model_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_landmarker.task')
    
    if not os.path.exists(model_file):
        print("❌ MediaPipe model file not found!")
        print(f"   Missing: {model_file}")
        print("\n📥 Download the model file from:")
        print("   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("\n💡 Place the downloaded file in the project directory")
        return False
    
    print("✅ MediaPipe model file found")
    return True

def get_ui_class(ui_name):
    """Return the UI class based on the name"""
    ui_map = {
        "research": ResearchEyeTrackerUI,
        "comprehensive": ComprehensiveEyeTrackerUI,
        "simple": SimpleEyeTrackerUI,
        "modern": ModernEyeTrackerUI,
        "headless": HeadlessUI,
    }
    return ui_map.get(ui_name.lower(), ResearchEyeTrackerUI)

def get_tracker_class(ui_name):
    """Return the tracker class based on the UI"""
    if ui_name.lower() == "research":
        return AdvancedEyeTracker
    else:
        # Import the regular tracker for other UIs
        from utils.core.tracker import FaceEyeTracker
        return FaceEyeTracker

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is available")
                return True
            else:
                print("❌ Camera is not working properly")
                return False
        else:
            print("❌ Camera is not accessible")
            return False
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def main():
    """Main application function"""
    parser = argparse.ArgumentParser(description="Advanced Eye Tracking & Cognitive Load Detection Application")
    parser.add_argument("--ui", type=str, default="research", 
                        choices=["research", "comprehensive", "simple", "modern", "headless"],
                        help="The user interface to use for the application.")
    parser.add_argument("--camera", type=int, default=0, help="The camera index to use.")
    parser.add_argument("--research-mode", action="store_true", default=True,
                        help="Enable research mode with advanced features.")
    args = parser.parse_args()

    print("🔬 Advanced Eye Tracking & Cognitive Load Detection System")
    print("=" * 60)
    print("📊 Professional Research-Grade Eye Tracking")
    print("🧠 Cognitive Load Assessment")
    print("😴 Advanced Fatigue Detection")
    print("📈 Real-time Analytics & Export")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Check model file
    print("\n🔍 Checking MediaPipe model...")
    if not check_model_file():
        return 1
    
    # Check camera
    print("\n🔍 Checking camera...")
    if not check_camera():
        print("\n💡 Camera issues can be resolved by:")
        print("   - Granting camera permissions to Terminal/Python")
        print("   - Ensuring no other application is using the camera")
        print("   - Checking System Settings > Privacy & Security > Camera")
        return 1
    
    print(f"\n🚀 Starting Advanced Eye Tracking & Cognitive Load Detection System...")
    if args.ui != "headless":
        print("   Application will open in a new window")
    
    if args.ui == "research":
        print("\n🔬 Research Mode Features:")
        print("   • Advanced calibration with 9-point system")
        print("   • High-precision pupil tracking")
        print("   • Cognitive load assessment")
        print("   • Research-grade data logging")
        print("   • Real-time quality monitoring")
        print("   • Export capabilities for analysis")
    
    print("\n📋 Instructions:")
    if args.ui == "research":
        print("   1. Complete calibration procedure for accurate tracking")
        print("   2. Start research session")
        print("   3. Monitor real-time research metrics")
        print("   4. Add annotations for significant events")
        print("   5. Export research data for analysis")
    else:
        print("   1. Click 'Start' to begin")
        print("   2. Allow camera access when prompted")
        print("   3. Position yourself in front of the camera")
        print("   4. Wait for calibration to complete")
        print("   5. Monitor real-time metrics and charts")
    
    print("\n⚡ Optimized for maximum performance and accuracy")
    print("\n🛑 Close window or press 'q' to stop")
    
    try:
        # Get appropriate tracker and UI classes
        tracker_class = get_tracker_class(args.ui)
        ui_class = get_ui_class(args.ui)
        
        # Initialize components with research mode if applicable
        if args.ui == "research":
            tracker = tracker_class(camera_index=args.camera, research_mode=args.research_mode)
        else:
            tracker = tracker_class(camera_index=args.camera)
        
        # Disable CSV logging for research mode - research UI handles its own data logging
        data_logger = DataLogger(enable_logging=args.ui != "research")
        
        # Initialize UI
        ui = ui_class(tracker, data_logger)
        
        # Run the application
        ui.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("\n💡 Common solutions:")
        print("   - Make sure all dependencies are installed")
        print("   - Check if you have proper permissions")
        print("   - Ensure the MediaPipe model file is present")
        print("   - Try running with --ui simple for basic mode")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 