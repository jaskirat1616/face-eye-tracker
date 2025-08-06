#!/usr/bin/env python3
"""
Eye Tracking & Fatigue Detection Application
============================================

A modern, professional application for real-time eye tracking and fatigue detection.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_eye_tracker.core.tracker import FaceEyeTracker
from face_eye_tracker.utils.data_logger import DataLogger
from face_eye_tracker.ui.comprehensive_ui import ComprehensiveEyeTrackerUI

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('tkinter', 'tkinter')
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
    model_file = 'face_landmarker.task'
    
    if not os.path.exists(model_file):
        print("❌ MediaPipe model file not found!")
        print(f"   Missing: {model_file}")
        print("\n📥 Download the model file from:")
        print("   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("\n💡 Place the downloaded file in the project directory")
        return False
    
    print("✅ MediaPipe model file found")
    return True

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
    print("👁️ Eye Tracking & Fatigue Detection")
    print("=" * 50)
    
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
    
    print("\n🚀 Starting Eye Tracking Application...")
    print("   The application will open in a new window")
    print("\n📋 Usage Instructions:")
    print("   1. Click 'Start Tracking' to begin")
    print("   2. Allow camera access when prompted")
    print("   3. Position yourself in front of the camera")
    print("   4. Wait for calibration to complete")
    print("   5. Monitor real-time metrics and charts")
    print("\n🛑 Close the window to stop the application")
    
    try:
        # Initialize components
        tracker = FaceEyeTracker(camera_index=0)
        data_logger = DataLogger()
        ui = ComprehensiveEyeTrackerUI(tracker, data_logger)
        
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
        return 1

if __name__ == "__main__":
    sys.exit(main()) 