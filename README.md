# 👁️ Eye Tracking & Cognitive Fatigue Detection System

A desktop application for real-time eye tracking and cognitive fatigue detection using advanced computer vision and machine learning techniques.

## 🎯 What This Application Does

### Primary Purpose
This application monitors your eyes in real-time to detect signs of cognitive fatigue and mental workload. It's designed for:

- **Researchers** studying cognitive load and fatigue
- **Students** monitoring study sessions and mental fatigue
- **Professionals** tracking work-related mental strain
- **Drivers** monitoring alertness (safety applications)
- **Gamers** tracking gaming session fatigue

### How It Works
1. **Real-time Eye Tracking**: Uses MediaPipe's advanced face landmark detection to track 468 facial points
2. **Eye Analysis**: Monitors eye openness, blink patterns, and pupil movements
3. **Fatigue Detection**: Analyzes multiple indicators to calculate cognitive fatigue scores
4. **Visual Feedback**: Provides real-time charts and metrics for immediate feedback

## ✨ Key Features

### 🔬 Advanced Eye Tracking
- **468-Point Face Mesh**: High-precision facial landmark detection
- **Dual Eye Monitoring**: Independent tracking of left and right eyes
- **Iris Detection**: Precise pupil position tracking for gaze analysis
- **Real-time Processing**: 30+ FPS processing with minimal latency

### 🧠 Cognitive Fatigue Detection
- **Multi-Indicator Analysis**: Combines blink rate, eye openness, and saccade patterns
- **Baseline Calibration**: Establishes personal baseline for accurate measurements
- **Fatigue Scoring**: 0-1 scale with color-coded severity levels
- **Trend Analysis**: Tracks fatigue progression over time

### 📊 Professional Analytics
- **Eye Openness Tracking**: Precise measurements of eyelid position
- **Blink Detection**: Accurate blink counting with duration analysis
- **Saccade Detection**: Large eye movements and microsaccades
- **Quality Monitoring**: Real-time tracking quality assessment

### ⚡ Performance Optimizations
- **Hardware Acceleration**: GPU-optimized processing when available
- **Frame Skipping**: Intelligent frame processing for smooth performance
- **Memory Efficiency**: Optimized data structures with rolling buffers
- **Minimal UI Updates**: Efficient rendering for consistent performance

## 🚀 Quick Start Guide

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: Built-in or external webcam
- **Storage**: 100MB free space

### Step-by-Step Installation

#### 1. Clone or Download the Repository
```bash
git clone https://github.com/jaskirat1616/face-eye-tracker.git
cd face_eye-tracker
```

#### 2. Install Python Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or install manually if needed:
pip install opencv-python mediapipe numpy matplotlib tkinter
```

#### 3. Verify the Model File
The `face_landmarker.task` file should already be included in the repository. If missing:
```bash
# Download the MediaPipe model (if needed)
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

#### 4. Run the Application
```bash
# Optimized version (recommended for most users)
python run_fast_eye_tracker.py

# Standard launcher
python run_eye_tracker.py
```

## 🎮 How to Use the Application

### Initial Setup
1. **Launch the Application**: Run the script and wait for system checks
2. **Camera Permission**: Allow camera access when prompted
3. **Position Yourself**: Sit 2-3 feet from the camera, face well-lit
4. **Calibration**: Wait 10-15 seconds for the system to calibrate

### Understanding the Interface

#### 🎛️ Control Panel (Left Sidebar)
- **Start/Stop Button**: Begin or pause tracking
- **Status Indicator**: Shows current tracking status (Ready/Active/Error)
- **Real-time Metrics**: Live updates of key measurements
- **Fatigue Indicators**: Progress bars for each fatigue component

#### 📹 Main Display Area
- **Live Video Feed**: Camera view with face mesh overlay
- **Face Mesh**: Green lines showing detected facial landmarks
- **Eye Tracking**: Highlighted eye regions and measurements

#### 📊 Real-time Charts (4 Panels)
1. **Eye Openness Chart**: Left and right eye openness over time
2. **Blink Rate Chart**: Blinks per minute trend
3. **Fatigue Score Chart**: Overall cognitive fatigue progression
4. **Quality Score Chart**: Detection quality and reliability

### Interpreting the Results

#### 🟢 Fatigue Levels (Color-Coded)
- **Green (0.0-0.3)**: Normal alertness, optimal performance
- **Yellow (0.3-0.5)**: Mild fatigue, consider taking a break
- **Orange (0.5-0.7)**: Moderate fatigue, rest recommended
- **Red (0.7-0.9)**: High fatigue, immediate rest advised
- **Dark Red (0.9-1.0)**: Severe fatigue, stop current activity

#### 📈 Key Metrics Explained
- **Eye Openness (0.0-1.0)**: 
  - Normal: 0.2-0.4
  - Fatigue: < 0.2 (drooping eyelids)
- **Blink Rate (per minute)**:
  - Normal: 15-20 blinks/minute
  - Fatigue: < 10 or > 30 blinks/minute
- **Quality Score (0.0-1.0)**:
  - Good: > 0.7
  - Poor: < 0.5 (check lighting/positioning)

## 🔬 Technical Architecture

### Core Components
```
face_eye_tracker/
├── face-eye-tracker/
│   ├── main.py                 # Application entry point
│   ├── utils/
│   │   ├── core/
│   │   │   └── tracker.py      # Core tracking engine
│   │   └── data_logger.py      # Data logging utilities
│   └── ui/
│       ├── modern_ui.py        # Modern desktop interface
│       ├── simple_ui.py        # Simple interface option
│       └── headless_ui.py      # Command-line interface
├── run_fast_eye_tracker.py     # Optimized launcher
├── run_eye_tracker.py          # Standard launcher
└── face_landmarker.task        # MediaPipe model file
```

### Technology Stack
- **MediaPipe**: Google's ML framework for face landmark detection
- **OpenCV**: Computer vision and video processing
- **NumPy**: Numerical computing and data analysis
- **Matplotlib**: Real-time chart rendering
- **Tkinter**: Cross-platform GUI framework

## 📊 Data Collection & Analysis

### Automatic Data Logging
The application automatically saves all tracking data to CSV files:
- **Location**: `face-eye-tracker/utils/data/`
- **Format**: Timestamped CSV with all metrics
- **Frequency**: Every frame processed
- **Retention**: Files kept locally, not uploaded

### Data Fields Recorded
- Timestamp (ISO format)
- Eye openness (left/right)
- Blink detection and duration
- Saccade events and velocities
- Fatigue scores (individual and overall)
- Quality indicators
- Head pose and position

### Privacy & Security
- **Local Processing**: All analysis happens on your device
- **No Cloud Upload**: Data never leaves your computer
- **Secure Storage**: CSV files stored locally only
- **No Network Access**: Application works completely offline

## 🎨 Customization Options

### UI Themes
The application supports different interface themes:
```bash
# Modern dark theme (default)
python run_fast_eye_tracker.py --ui modern

# Simplified interface
python run_fast_eye_tracker.py --ui simple

# Comprehensive interface
python run_fast_eye_tracker.py --ui comprehensive

# Headless mode (no GUI)
python run_fast_eye_tracker.py --ui headless
```

### Performance Settings
Adjust performance in `face-eye-tracker/utils/core/tracker.py`:
```python
# Frame skipping for performance
self.frame_skip = 1  # Process every 2nd frame

# Quality vs speed trade-off
self.min_face_detection_confidence = 0.5  # Lower = faster
```

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### Camera Problems
**Issue**: "Camera not found" or "Camera not accessible"
**Solutions**:
- Check camera permissions in system settings
- Close other applications using the camera
- Try different camera index: `--camera 1`
- Restart the application

#### Performance Issues
**Issue**: Low FPS or laggy interface
**Solutions**:
- Close other applications to free up resources
- Reduce camera resolution in system settings
- Ensure good lighting conditions
- Check if hardware acceleration is enabled

#### Model File Errors
**Issue**: "MediaPipe model file not found"
**Solutions**:
- Verify `face_landmarker.task` is in the project directory
- Download the model file from the provided URL
- Check file permissions

#### Import Errors
**Issue**: "ModuleNotFoundError" or missing dependencies
**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install manually
pip install opencv-python mediapipe numpy matplotlib
```

### Getting Better Results

#### Optimal Setup
1. **Lighting**: Ensure even, bright lighting on your face
2. **Positioning**: Sit 2-3 feet from camera, face centered
3. **Background**: Use a plain, uncluttered background
4. **Glasses**: Remove reflective glasses if possible
5. **Calibration**: Wait 10-15 seconds for initial calibration

#### Performance Tips
- Use the optimized version for better performance
- Close unnecessary applications
- Ensure good internet connection (for initial model download)
- Regular breaks every 20-30 minutes of use

## 🔬 Research Applications

### Academic Use Cases
- **Cognitive Load Studies**: Measure mental workload during tasks
- **Fatigue Research**: Study patterns of mental fatigue
- **Attention Studies**: Track focus and attention spans
- **Ergonomics Research**: Evaluate work environment impact

### Professional Applications
- **Driver Safety**: Monitor alertness during long drives
- **Workplace Wellness**: Track employee fatigue levels
- **Gaming Research**: Study gaming session fatigue
- **Medical Applications**: Assist in fatigue-related assessments

## 📚 Scientific Background

### Eye Tracking Metrics
- **Eye Openness**: Correlates with alertness and fatigue
- **Blink Rate**: Changes with cognitive load and fatigue
- **Saccades**: Eye movements indicate attention and processing
- **Microsaccades**: Small movements related to visual processing

### Fatigue Detection Algorithm
The application uses a multi-indicator approach:
1. **Baseline Establishment**: Personal normal ranges
2. **Real-time Monitoring**: Continuous measurement
3. **Pattern Analysis**: Trend detection and analysis
4. **Scoring Algorithm**: Weighted combination of indicators

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone https://github.com/jaskirat1616/face-eye-tracker.git
cd face-eye-tracker

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

### Code Structure
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints where appropriate
- Test changes before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team**: For the excellent face landmark detection
- **OpenCV Community**: For computer vision capabilities
- **Research Community**: For eye tracking and fatigue detection research

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this README and code comments
- **Community**: Join discussions in the repository

---

**⚠️ Important Note**: This application is for research and educational purposes. It should not be used as a medical device or for critical safety applications without proper validation and certification.

**🎯 Best Results**: For optimal performance, ensure good lighting, proper positioning, and allow time for calibration. The application works best in controlled environments with consistent lighting conditions.
