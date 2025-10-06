# High Precision Eye Tracking System

🔬 **Eye Tracking System with Research Features**

A high precision eye tracking system designed for research, academic studies, and human-computer interaction applications.

## 🌟 Key Features

### 🔬 Research Accuracy
- **High-precision pupil tracking** with sub-pixel accuracy
- **Calibration system** with 9-point calibration
- **Real-time quality assessment** and monitoring
- **Multi-modal sensor fusion** for enhanced accuracy

### 🧠 Eye Tracking Analytics
- **Attention span analysis** and processing speed assessment

### 📊 Research Tools
- **Data logging** with multiple export formats
- **Real-time analytics dashboard** with charts
- **Session management** and annotation capabilities
- **Quality monitoring** and statistical analysis

### 🎯 Calibration
- **9-point calibration system** for high accuracy
- **Real-time calibration quality assessment**
- **Adaptive calibration** based on user performance
- **Calibration validation** and feedback

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd face_eye_tracker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the MediaPipe model:**
```bash
# Download face_landmarker.task from:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Place it in the face_eye_tracker directory
```

### Running the Application

#### Research Mode
```bash
python run_eye_tracker.py --ui research
```

#### Other UI Options
```bash
# Modern UI
python run_eye_tracker.py --ui modern

# Simple UI
python run_eye_tracker.py --ui simple

# Full Feature UI
python run_eye_tracker.py --ui comprehensive

# Headless mode
python run_eye_tracker.py --ui headless
```

## 🔬 Research Mode Features

### Calibration
1. **Start Calibration**: Click "Start Calibration" in the research interface
2. **Follow Points**: Look at each of the 9 calibration points as prompted
3. **Quality Assessment**: Monitor calibration quality in real-time
4. **Complete**: Finish calibration when all points are calibrated

### Research Session
1. **Start Session**: Begin data collection with "Start Research Session"
2. **Monitor Metrics**: Watch real-time research metrics and quality indicators
3. **Add Annotations**: Mark significant events or conditions during the session
4. **Export Data**: Export research data for analysis

### Data Export
The system supports multiple export formats:
- **JSON**: Complete session data with metadata
- **CSV**: Tabular data for statistical analysis
- **Excel**: Multi-sheet workbook with data
- **Pickle**: Python-compatible data format

## 📊 Research Metrics

### Eye Tracking Metrics
- **Pupil Position**: High-precision pupil center coordinates
- **Gaze Point**: Estimated gaze position on screen
- **Pupil Diameter**: Pupil size measurements
- **Eye Velocity**: Movement speed and patterns
- **Fixation Duration**: Time spent looking at specific areas

- **Blink Pattern Analysis**: Blink rate, duration, and patterns
- **Eye Openness**: Continuous monitoring of eye openness
- **Head Pose**: Head position and movement analysis


### Quality Metrics
- **Tracking Quality**: Overall system performance
- **Calibration Quality**: Calibration accuracy assessment
- **Face Detection**: Face detection confidence
- **Pupil Tracking**: Pupil detection reliability

## 🏗️ System Architecture

### Core Components

#### AdvancedEyeTracker
- High-precision pupil tracking
- Real-time quality monitoring

#### ResearchEyeTrackerUI
- Research interface
- Calibration system
- Real-time analytics dashboard
- Session management tools

#### ResearchDataLogger
- Data collection
- Multiple export formats
- Real-time analysis
- Quality assessment

### File Structure
```
face_eye_tracker/
├── face-eye-tracker/
│   ├── utils/
│   │   ├── core/
│   │   │   ├── advanced_tracker.py    # Tracking engine
│   │   │   └── tracker.py             # Standard tracking engine
│   │   ├── research_data_logger.py    # Research data logging
│   │   └── data_logger.py             # Standard data logging
│   ├── ui/
│   │   ├── research_ui.py             # Research interface
│   │   ├── modern_ui.py               # Modern interface
│   │   ├── comprehensive_ui.py        # Full feature interface
│   │   ├── simple_ui.py               # Simple interface
│   │   └── headless_ui.py             # Headless interface
│   └── main.py                        # Main application
├── face_landmarker.task               # MediaPipe model
├── requirements.txt                   # Dependencies
└── run_eye_tracker.py                 # Launcher script
```

## 📈 Performance Optimization

### Real-time Processing
- **Optimized frame processing** for minimal latency
- **Efficient data structures** for high-frequency updates
- **Background analysis threads** for non-blocking operation
- **Smart buffering** for smooth UI updates

### Quality Assurance
- **Real-time quality monitoring** with automatic adjustments
- **Adaptive thresholds** based on environmental conditions
- **Error handling** and recovery mechanisms
- **Performance metrics** and optimization feedback

## 🔧 Configuration

### Camera Settings
- **Resolution**: 1280x720 (research mode), 640x480 (standard mode)
- **Frame Rate**: 60 FPS (research mode), 30 FPS (standard mode)
- **Auto-focus**: Enabled for stability
- **Auto-exposure**: Optimized for eye tracking

### Processing Parameters
- **Pupil Detection Confidence**: 0.8 (high accuracy)
- **Gaze Estimation Confidence**: 0.7 (balanced accuracy)
- **Quality Threshold**: 0.7 (minimum acceptable quality)

## 📚 Research Applications

### Academic Research
- **Cognitive science studies**
- **Human-computer interaction research**
- **Attention and focus studies**

### Applications
- **Driver monitoring systems**
- **Workplace safety assessment**
- **Educational technology research**
- **Healthcare monitoring**

### User Experience Research
- **Interface usability studies**
- **Attention pattern analysis**
- **User behavior research**

## 🛠️ Development

### Adding New Features
1. **Extend AdvancedEyeTracker** for new tracking capabilities
2. **Update ResearchEyeTrackerUI** for new interface elements
3. **Enhance ResearchDataLogger** for new data types
4. **Add new export formats** as needed

### Customization
- **Modify tracking parameters** in `advanced_tracker.py`
- **Customize UI elements** in `research_ui.py`
- **Add new metrics** in the data logging system
- **Implement custom analysis** algorithms

## 📊 Data Analysis

### Export Formats
- **JSON**: Complete session data with metadata
- **CSV**: Tabular data for statistical analysis
- **Excel**: Multi-sheet workbook with data
- **Pickle**: Python-compatible data format

### Analysis Tools
- **Real-time analytics** in the research interface
- **Statistical summaries** and trend analysis
- **Quality assessment** and validation
- **Custom analysis** capabilities

## 🤝 Contributing

We welcome contributions to improve the system:

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement improvements**
4. **Add tests and documentation**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** for face landmark detection
- **OpenCV** for computer vision capabilities
- **Research community** for feedback and improvements

## 📞 Support

For questions, issues, or feature requests:
- **Create an issue** on GitHub
- **Check the documentation** for common solutions
- **Review the examples** for usage patterns

---

**🔬 Eye Tracking System**  
*Accurate and Reliable*

# Research Eye Tracking System


## Quick Start: Running the Research UI

To run the research UI, execute the following command from the `face_eye_tracker` directory:

```bash
python3 face-eye-tracker/main.py --ui research
```

This will launch the research interface, which includes:
-   Real-time data visualization
-   Head pose (yaw and roll) display

## Head Pose Estimation

The system now displays the head's yaw and roll angles in the research UI. These values are estimated from the face mesh and can be used to monitor the subject's head orientation during the session.

## Note on Glasses

The accuracy of the eye tracking may be affected if the user is wearing glasses. The system may not perform optimally with all types of eyewear.

## Original README Content

... (The original README content can be appended here if needed) ...
