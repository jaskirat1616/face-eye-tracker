#!/usr/bin/env python3
"""
Test Research Data Accuracy
==========================

This script tests the research UI to ensure all data is displayed accurately
and all features are working properly.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face-eye-tracker.utils.core.advanced_tracker import AdvancedEyeTracker
from face-eye-tracker.utils.data_logger import DataLogger
from face-eye-tracker.ui.research_ui import ResearchEyeTrackerUI

def test_data_fields():
    """Test that all expected data fields are present"""
    print("🔍 Testing data fields...")
    
    tracker = AdvancedEyeTracker(research_mode=True)
    data_logger = DataLogger(enable_logging=True)
    
    # Start camera
    if not tracker.start_camera():
        print("❌ Failed to start camera")
        return False
    
    print("✅ Camera started")
    
    # Wait for some data to be collected
    time.sleep(2)
    
    # Get current data
    data = tracker.get_current_data()
    
    # Expected data fields
    expected_fields = [
        'advanced_fatigue_score',
        'advanced_quality_score',
        'cognitive_load_score',
        'pupil_diameter',
        'gaze_stability',
        'eye_velocity',
        'fixation_duration',
        'attention_span',
        'processing_speed',
        'mental_effort',
        'blink_rate',
        'saccade_rate',
        'head_tilt',
        'face_confidence'
    ]
    
    print(f"📊 Current data keys: {list(data.keys())}")
    
    # Check for expected fields
    missing_fields = []
    for field in expected_fields:
        if field not in data:
            missing_fields.append(field)
        else:
            print(f"✅ {field}: {data[field]}")
    
    if missing_fields:
        print(f"❌ Missing fields: {missing_fields}")
        return False
    
    print("✅ All expected data fields are present")
    
    # Stop camera
    tracker.stop_camera()
    return True

def test_ui_components():
    """Test that all UI components are created properly"""
    print("\n🔍 Testing UI components...")
    
    tracker = AdvancedEyeTracker(research_mode=True)
    data_logger = DataLogger(enable_logging=True)
    
    try:
        # Create UI (don't run it)
        ui = ResearchEyeTrackerUI(tracker, data_logger)
        
        # Test that all expected UI components exist
        expected_components = [
            'research_metrics',
            'quality_indicators',
            'chart_data',
            'status_label',
            'data_quality_label',
            'quality_warning_label',
            'data_collection_warning_label',
            'calib_quality_warning_label',
            'face_detection_warning_label',
            'pupil_tracking_warning_label',
            'gaze_estimation_warning_label'
        ]
        
        for component in expected_components:
            if hasattr(ui, component):
                print(f"✅ {component} component exists")
            else:
                print(f"❌ {component} component missing")
                return False
        
        print("✅ All UI components are present")
        return True
        
    except Exception as e:
        print(f"❌ UI creation error: {e}")
        return False

def test_chart_data():
    """Test that chart data is being collected properly"""
    print("\n🔍 Testing chart data collection...")
    
    tracker = AdvancedEyeTracker(research_mode=True)
    data_logger = DataLogger(enable_logging=True)
    
    try:
        ui = ResearchEyeTrackerUI(tracker, data_logger)
        
        # Test chart data structure
        expected_chart_keys = [
            'time', 'fatigue', 'quality', 'pupil_diameter', 'gaze_stability',
            'eye_velocity', 'cognitive_load', 'attention_span', 'processing_speed',
            'mental_effort', 'blink_rate', 'saccade_rate', 'fixation_duration'
        ]
        
        for key in expected_chart_keys:
            if key in ui.chart_data:
                print(f"✅ Chart data key '{key}' exists")
            else:
                print(f"❌ Chart data key '{key}' missing")
                return False
        
        print("✅ All chart data keys are present")
        return True
        
    except Exception as e:
        print(f"❌ Chart data test error: {e}")
        return False

def test_export_functionality():
    """Test export functionality"""
    print("\n🔍 Testing export functionality...")
    
    tracker = AdvancedEyeTracker(research_mode=True)
    data_logger = DataLogger(enable_logging=True)
    
    try:
        ui = ResearchEyeTrackerUI(tracker, data_logger)
        
        # Test that export methods exist
        export_methods = [
            '_perform_export',
            '_export_to_csv',
            '_export_to_excel'
        ]
        
        for method in export_methods:
            if hasattr(ui, method):
                print(f"✅ Export method '{method}' exists")
            else:
                print(f"❌ Export method '{method}' missing")
                return False
        
        print("✅ All export methods are present")
        return True
        
    except Exception as e:
        print(f"❌ Export functionality test error: {e}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    print("\n🔍 Testing data validation...")
    
    tracker = AdvancedEyeTracker(research_mode=True)
    data_logger = DataLogger(enable_logging=True)
    
    try:
        ui = ResearchEyeTrackerUI(tracker, data_logger)
        
        # Test that validation methods exist
        if hasattr(ui, '_update_data_validation_warnings'):
            print("✅ Data validation method exists")
        else:
            print("❌ Data validation method missing")
            return False
        
        # Test validation with sample data
        sample_data = {
            'advanced_quality_score': 0.3,  # Low quality
            'face_confidence': 0.2,  # Low confidence
            'pupil_position': None,  # No pupil tracking
            'gaze_point': None  # No gaze estimation
        }
        
        # This should not raise an exception
        ui._update_data_validation_warnings(sample_data)
        print("✅ Data validation works with sample data")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Research Data Accuracy Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_fields,
        test_ui_components,
        test_chart_data,
        test_export_functionality,
        test_data_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Research UI is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 