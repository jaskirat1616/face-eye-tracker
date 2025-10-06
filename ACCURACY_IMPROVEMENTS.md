# Eye Tracker Accuracy Improvements Documentation

## Overview
This document details the accuracy improvements made to the high precision eye tracking system. These enhancements focus on improving pupil detection, gaze estimation, noise reduction, and head pose compensation for more accurate tracking.

## Key Improvements

### 1. Enhanced Pupil Detection Algorithm
- **Robust Center Calculation**: Instead of simple averaging, the system now uses a combination of bounding box center and weighted mean to reduce outlier sensitivity
- **Improved Stability**: Uses 70% weighted mean + 30% bounding box center for more stable pupil tracking

### 2. Advanced Head Pose Estimation
- **Improved 3D Model**: More accurate 3D face model points based on average face measurements
- **Better Camera Matrix**: Uses 1.2x focal length for webcams for more accurate projection
- **Realistic Distortion**: Added realistic distortion coefficients [0.1, -0.2, 0.0, 0.0, 0.0] for webcam lenses
- **Smoother Filtering**: Uses median of recent values for reduced noise

### 3. Gaze Estimation with Head Pose Compensation
- **Dynamic Compensation**: Adjusts gaze estimation based on head yaw, tilt, and roll
- **Calibration Integration**: Incorporates head pose data into gaze calculations
- **Bounds Checking**: Ensures gaze points remain within [0,1] range

### 4. Noise Reduction and Filtering
- **Multi-level Filtering**: Uses median filters and velocity-based smoothing
- **Enhanced Buffers**: Added filtering buffers with size 10 for pupil and 15 for gaze
- **Velocity Smoothing**: Added smoothed velocity calculations using median of recent values
- **Robust Statistics**: Uses median instead of mean to reduce outlier impact

### 5. Advanced Gaze Stability Calculation
- **Dual-Metric Approach**: Combines velocity-based and variance-based stability measures 
- **Weighted Combination**: 60% velocity stability + 40% variance stability for robustness
- **Adaptive Window**: Uses up to 10 recent gaze points for calculation

## Technical Changes

### In `advanced_tracker.py`:
- Updated `_extract_pupil_center()` with robust calculation
- Enhanced `_estimate_head_pose()` with improved 3D model and filtering
- Modified `_estimate_gaze_point()` to include head pose compensation
- Added `_apply_noise_filtering()` method for smoothing
- Updated `_analyze_eye_movements()` with median-based calculations
- Enhanced `_calculate_gaze_stability()` with dual-metric approach
- Updated `_extract_advanced_data()` to incorporate filtering

### In `research_ui.py`:
- Added smoothed eye velocity metric display
- Updated chart visualization to show filtered metrics
- Maintained backward compatibility with existing metrics

## Performance Improvements

### Accuracy Gains:
- **Reduced Jitter**: Median filtering reduces high-frequency noise
- **Better Stability**: Multi-frame analysis provides more stable tracking
- **Compensated Tracking**: Head pose compensation improves gaze accuracy during head movement
- **Robust Detection**: Outlier-resistant pupil center calculation

### Computational Considerations:
- Maintained real-time performance with efficient algorithms
- Used optimal buffer sizes for responsiveness vs. stability balance
- Implemented median filters for robustness without excessive computation

## Validation

The system includes a comprehensive test framework (`test_accuracy_improvements.py`) that validates:
- Gaze stability over time
- Pupil tracking jitter reduction
- Head pose compensation effectiveness
- Overall tracking smoothness

## Usage

The improvements are transparent to end users - simply run the eye tracker as before:
```bash
python run_eye_tracker.py --ui research
```

The enhanced algorithms will automatically provide more accurate and stable tracking.

## Expected Results

Compared to the previous version, users should experience:
- Smoother gaze tracking with less jitter
- Better accuracy during slight head movements
- More consistent pupil detection
- Improved stability metrics in the UI