# 👁️ Modern Eye Tracking & Fatigue Detection

A clean, professional desktop application for real-time eye tracking and fatigue detection with beautiful charts and a modern interface.

## ✨ Features

### 🎥 Real-time Processing
- Live camera feed with face mesh overlay
- High-accuracy eye tracking with MediaPipe
- Real-time fatigue analysis and scoring

### 📊 Professional UI
- Modern dark theme interface
- Real-time line charts with smooth animations
- Live metrics display with color-coded indicators
- Progress bars for fatigue scores

### 🏥 Advanced Analytics
- **Eye Openness Tracking**: Precise left/right eye measurements
- **Blink Detection**: Accurate blink detection with rate analysis
- **Saccade Detection**: Large and microsaccade analysis
- **Fatigue Analysis**: Comprehensive multi-indicator scoring

### 📈 Real-time Charts
- Eye openness over time (dual-line chart)
- Blink rate trends
- Fatigue score progression
- Detection quality monitoring

## 🚀 Quick Start

### Prerequisites
1. Python 3.8 or higher
2. Webcam access
3. MediaPipe model file

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download MediaPipe Model**:
   Download `face_landmarker.task` from:
   ```
   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```
   Place it in the project root directory.

3. **Run the Application**:
   ```bash
   python run_eye_tracker.py
   ```

## 🎯 Usage

### Starting the Application
1. Run `python run_eye_tracker.py`
2. The application will perform system checks
3. A modern desktop window will open
4. Click "Start Tracking" to begin

### Understanding the Interface

#### 🎛️ Control Panel (Left Sidebar)
- **Start/Stop Tracking**: Control buttons
- **Status Indicator**: Shows current tracking status
- **Real-time Metrics**: Live eye openness, blink rate, etc.
- **Fatigue Analysis**: Progress bars for each fatigue indicator

#### 📹 Main Area
- **Video Feed**: Camera view with face mesh overlay
- **Real-time Charts**: Four animated charts showing trends

#### 📊 Charts
- **Eye Openness**: Left and right eye openness over time
- **Blink Rate**: Blinks per minute trend
- **Fatigue Score**: Overall fatigue progression
- **Quality Score**: Detection quality over time

### Interpreting Results

#### Fatigue Levels
- **Normal (Green)**: < 0.3 fatigue score
- **Mild Fatigue (Orange)**: 0.3 - 0.6 fatigue score
- **Moderate Fatigue (Red)**: 0.6 - 0.8 fatigue score
- **Severe Fatigue (Dark Red)**: > 0.8 fatigue score

#### Key Metrics
- **Eye Openness**: Normal range 0.2-0.4
- **Blink Rate**: Normal 15-20 blinks/minute
- **Quality Score**: > 0.7 indicates good tracking

## 🏗️ Architecture

### Modular Design
```
face_eye_tracker/
├── core/
│   └── tracker.py          # Core tracking engine
├── ui/
│   └── modern_ui.py        # Modern desktop UI
├── utils/
│   └── data_logger.py      # Data logging utilities
├── data/                   # CSV data storage
└── main.py                 # Application entry point
```

### Key Components
- **FaceEyeTracker**: Core tracking with MediaPipe
- **ModernEyeTrackerUI**: Tkinter-based modern interface
- **DataLogger**: CSV logging and real-time data management

## 🔧 Technical Details

### Performance
- **Frame Rate**: ~30 FPS processing
- **Charts**: Real-time updates with smooth animations
- **Memory**: Efficient data structures with rolling buffers

### Dependencies
- **OpenCV**: Video capture and processing
- **MediaPipe**: Face landmark detection
- **Matplotlib**: Real-time chart rendering
- **Tkinter**: Desktop UI framework

## 📝 Data Logging

The application automatically logs all tracking data to CSV files:
- Timestamped data points
- All eye tracking metrics
- Fatigue analysis scores
- Quality indicators

Files are saved in `face_eye_tracker/data/` with timestamps.

## 🎨 Customization

### UI Themes
The application uses a modern dark theme with:
- Dark backgrounds (#1e1e1e, #2d2d2d)
- White text and grid lines
- Color-coded indicators (green/orange/red)

### Chart Customization
Modify chart appearance in `modern_ui.py`:
```python
# Chart colors
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
```

## 🐛 Troubleshooting

### Common Issues
- **Camera not found**: Check permissions and availability
- **Model file missing**: Download the MediaPipe model
- **Performance issues**: Reduce camera resolution or FPS
- **Import errors**: Install all dependencies with `pip install -r requirements.txt`

### Performance Tips
- Close other applications using the camera
- Ensure good lighting conditions
- Position face clearly in camera view
- Wait for calibration to complete

## 🔒 Privacy & Security

- **Local Processing**: All processing happens locally
- **No Data Transmission**: No data sent to external servers
- **Secure Storage**: Data saved locally in CSV format

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This application is for research and educational purposes. Ensure proper lighting and camera positioning for best results. # face-eye-tracker
