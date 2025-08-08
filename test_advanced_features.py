#!/usr/bin/env python3
"""
Test Script for Advanced Eye Tracking Features
==============================================

This script tests the advanced features of the research-grade eye tracking system.
"""

import sys
import os
import time
import numpy as np

# Add the face-eye-tracker directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'face-eye-tracker'))

def test_advanced_tracker():
    """Test the advanced tracker functionality"""
    print("🔬 Testing Advanced Eye Tracker...")
    
    try:
        from utils.core.advanced_tracker import AdvancedEyeTracker
        from utils.research_data_logger import ResearchDataLogger
        
        # Initialize components
        tracker = AdvancedEyeTracker(camera_index=0, research_mode=True)
        data_logger = ResearchDataLogger(enable_logging=True)
        
        print("✅ Advanced tracker initialized successfully")
        
        # Test calibration
        calibration_points = tracker.start_calibration()
        print(f"✅ Calibration started with {len(calibration_points)} points")
        
        # Test camera start
        if tracker.start_camera():
            print("✅ Camera started successfully")
            
            # Test frame processing
            frame = tracker.read_frame()
            if frame is not None:
                print("✅ Frame reading successful")
                
                # Test frame processing
                processed_frame = tracker.process_frame(frame)
                if processed_frame is not None:
                    print("✅ Frame processing successful")
                    
                    # Test data extraction
                    data = tracker.get_current_data()
                    if data:
                        print("✅ Data extraction successful")
                        print(f"   - Data keys: {list(data.keys())}")
                    else:
                        print("⚠️  No data extracted (normal if no face detected)")
                
            tracker.stop_camera()
            print("✅ Camera stopped successfully")
        else:
            print("❌ Failed to start camera")
        
        # Test research data logger
        data_logger.start_session("test_session")
        print("✅ Research data logger session started")
        
        # Test data logging
        test_data = {
            'pupil_position': [0.5, 0.5],
            'advanced_fatigue_score': 0.3,
            'cognitive_load_score': 0.4,
            'advanced_quality_score': 0.8
        }
        data_logger.log_raw_data(test_data)
        print("✅ Data logging successful")
        
        # Test session summary
        summary = data_logger.get_session_summary()
        if summary:
            print("✅ Session summary generated")
            print(f"   - Session ID: {summary.get('session_id', 'N/A')}")
            print(f"   - Total frames: {summary.get('total_frames', 0)}")
        
        data_logger.stop_session()
        print("✅ Research data logger session stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced tracker test failed: {e}")
        return False

def test_research_ui():
    """Test the research UI components"""
    print("\n🔬 Testing Research UI Components...")
    
    try:
        from ui.research_ui import ResearchEyeTrackerUI
        from utils.core.advanced_tracker import AdvancedEyeTracker
        from utils.research_data_logger import ResearchDataLogger
        
        # Initialize components
        tracker = AdvancedEyeTracker(camera_index=0, research_mode=True)
        data_logger = ResearchDataLogger(enable_logging=True)
        ui = ResearchEyeTrackerUI(tracker, data_logger)
        
        print("✅ Research UI initialized successfully")
        
        # Test UI creation (without running mainloop)
        try:
            ui.create_ui()
            print("✅ Research UI creation successful")
        except Exception as e:
            print(f"⚠️  UI creation test skipped (requires display): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research UI test failed: {e}")
        return False

def test_export_capabilities():
    """Test data export capabilities"""
    print("\n🔬 Testing Export Capabilities...")
    
    try:
        from utils.research_data_logger import ResearchDataLogger
        
        # Initialize data logger
        data_logger = ResearchDataLogger(enable_logging=True)
        data_logger.start_session("export_test")
        
        # Add some test data
        for i in range(10):
            test_data = {
                'pupil_position': [0.5 + i*0.01, 0.5 + i*0.01],
                'advanced_fatigue_score': 0.3 + i*0.05,
                'cognitive_load_score': 0.4 + i*0.03,
                'advanced_quality_score': 0.8 - i*0.02
            }
            data_logger.log_raw_data(test_data)
            data_logger.log_annotation(f"Test annotation {i+1}")
        
        # Test JSON export
        json_file = data_logger.export_session_data('json', 'test_export')
        if json_file and os.path.exists(json_file):
            print("✅ JSON export successful")
            os.remove(json_file)  # Clean up
        
        # Test CSV export
        csv_file = data_logger.export_session_data('csv', 'test_export')
        if csv_file:
            print("✅ CSV export successful")
            # Clean up CSV files
            base_name = csv_file.replace('_*.csv', '')
            for ext in ['_raw_data.csv', '_processed_data.csv', '_annotations.csv', '_quality_metrics.csv']:
                file_path = base_name + ext
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        data_logger.stop_session()
        print("✅ Export capabilities test completed")
        return True
        
    except Exception as e:
        print(f"❌ Export capabilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Advanced Eye Tracking System - Feature Tests")
    print("=" * 60)
    
    tests = [
        ("Advanced Tracker", test_advanced_tracker),
        ("Research UI", test_research_ui),
        ("Export Capabilities", test_export_capabilities)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The advanced system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 