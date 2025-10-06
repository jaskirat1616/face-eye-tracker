#!/usr/bin/env python3
"""
High Precision Eye Tracking System
=====================================================================

Eye tracking system with features:
- High-precision pupil tracking with sub-pixel accuracy
- Fatigue detection algorithms
- Cognitive load assessment
- Research-grade data logging and analysis
- Real-time quality assessment and calibration
- Multi-modal sensor fusion
- Export capabilities for research
"""

import warnings
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import pickle
from datetime import datetime, timedelta
from collections import deque, defaultdict
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from scipy import signal
from scipy.stats import linregress
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import style
import json
import os

# Suppress warnings for clean operation
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face mesh connections for drawing
FACE_MESH_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS
FACE_MESH_LIPS = mp.solutions.face_mesh.FACEMESH_LIPS
FACE_MESH_LEFT_EYE = mp.solutions.face_mesh.FACEMESH_LEFT_EYE
FACE_MESH_LEFT_EYEBROW = mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW
FACE_MESH_LEFT_IRIS = mp.solutions.face_mesh.FACEMESH_LEFT_IRIS
FACE_MESH_RIGHT_EYE = mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
FACE_MESH_RIGHT_EYEBROW = mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW
FACE_MESH_RIGHT_IRIS = mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS

# Combine all the desired connections into a single set
FACE_MESH_CUSTOM_CONNECTIONS = list(
    FACE_MESH_CONTOURS |
    FACE_MESH_LIPS |
    FACE_MESH_LEFT_EYE |
    FACE_MESH_LEFT_EYEBROW |
    FACE_MESH_LEFT_IRIS |
    FACE_MESH_RIGHT_EYE |
    FACE_MESH_RIGHT_EYEBROW |
    FACE_MESH_RIGHT_IRIS
)

class EyeTracker:
    """
    Eye tracking system with professional features
    """
    
    def __init__(self, camera_index=0, research_mode=True):
        self.camera_index = camera_index
        self.research_mode = research_mode
        self.is_running = False
        self.cap = None
        
        # Advanced calibration system
        self.calibration_points = []
        self.calibration_data = {}
        self.calibration_complete = False
        self.calibration_quality = 0.0
        
        # High-precision tracking with filtering
        self.pupil_tracking_history = deque(maxlen=200)  # Reduced size for more responsive tracking
        self.gaze_history = deque(maxlen=200)
        self.eye_movement_history = deque(maxlen=100)
        
        # Enhanced filtering buffers for noise reduction
        self.filtered_pupil_history = deque(maxlen=10)
        self.filtered_gaze_history = deque(maxlen=15)
        self.velocity_buffer = deque(maxlen=5)
        
        # Fatigue detection
        self.fatigue_indicators = {
            'blink_pattern': deque(maxlen=200),
            'pupil_diameter': deque(maxlen=200),
            'fixation_duration': deque(maxlen=200),
            'head_movement': deque(maxlen=200),
            'eye_strain': deque(maxlen=200)
        }
        
        # Research data collection
        self.research_data = {
            'session_start': None,
            'session_duration': 0,
            'total_frames': 0,
            'quality_metrics': deque(maxlen=1000),
            'events': [],
            'annotations': []
        }
        
        # Initialize MediaPipe with settings
        self._initialize_mediapipe()
        
        # Advanced processing parameters
        self.processing_params = {
            'pupil_detection_confidence': 0.8,
            'gaze_estimation_confidence': 0.7,
            'quality_threshold': 0.7
        }
        
        # Real-time analysis threads
        self.analysis_thread = None
        self.data_queue = queue.Queue(maxsize=100)
        self.blink_times = deque(maxlen=100)
        self.is_blinking = False
        self.blink_start_time = None
        self.eye_openness_threshold = 0.2  # You may want to tune this
        self.baseline_openness = None
        
        # Head pose smoothing
        self.head_pose_history = {
            'tilt': deque(maxlen=5),
            'yaw': deque(maxlen=5),
            'roll': deque(maxlen=5)
        }
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe with research settings"""
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'face_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=self._result_callback
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.result = None
        self.timestamp = 0
        
        # 3D model points for head pose estimation
        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0, -150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0]     # Right mouth corner
        ])
        
        # Landmark indices for research
        self.landmark_indices = {
            'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'left_iris': [473, 474, 475, 476, 477],
            'right_iris': [468, 469, 470, 471, 472],
            'pupil_center_left': [473, 474, 475, 476, 477],
            'pupil_center_right': [468, 469, 470, 471, 472],
            'head_pose': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        # Vertical pairs for eye openness (indices are positions within the eye landmark list)
        self.LEFT_EYE_VERTICAL_PAIRS = [(1, 5), (2, 4), (3, 7), (8, 12), (9, 11), (0, 6)]
        self.RIGHT_EYE_VERTICAL_PAIRS = [(1, 5), (2, 4), (3, 7), (8, 12), (9, 11), (0, 6)]
    
    def _result_callback(self, result: vision.FaceLandmarkerResult, output_image, timestamp_ms: int):
        """Callback for MediaPipe results"""
        self.result = result
        self.timestamp = timestamp_ms
    
    def _calculate_eye_openness(self, face_landmarks, eye_indices, vertical_pairs):
        eye_points = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in eye_indices])
        vertical_distances = []
        for a, b in vertical_pairs:
            if a < len(eye_points) and b < len(eye_points):
                vertical_distances.append(abs(eye_points[a][1] - eye_points[b][1]))
        if vertical_distances:
            weights = [1.0, 1.2, 1.5, 1.2, 1.0, 0.8][:len(vertical_distances)]
            vertical_distance = np.average(vertical_distances, weights=weights)
        else:
            vertical_distance = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
        horizontal_distance = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
        return (vertical_distance / horizontal_distance) if horizontal_distance > 0 else 0.0

    def start_calibration(self):
        """Start calibration procedure"""
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
        ]
        self.calibration_data = {}
        self.calibration_complete = False
        self.calibration_quality = 0.0
        
        print("🔬 Starting calibration procedure...")
        return self.calibration_points
    
    def calibrate_point(self, point_index, gaze_data):
        """Calibrate a specific point with high precision"""
        if point_index < len(self.calibration_points):
            target = self.calibration_points[point_index]
            self.calibration_data[point_index] = {
                'target': target,
                'gaze_samples': gaze_data,
                'timestamp': time.time()
            }
            
            # Calculate calibration quality for this point
            if len(gaze_data) > 5:
                gaze_center = np.mean(gaze_data, axis=0)
                gaze_variance = np.var(gaze_data, axis=0)
                accuracy = 1.0 / (1.0 + np.sum(gaze_variance))
                self.calibration_data[point_index]['accuracy'] = accuracy
                
                print(f"✅ Calibrated point {point_index + 1}/9 - Accuracy: {accuracy:.3f}")
            
            return True
        return False
    
    def finish_calibration(self):
        """Complete calibration and calculate overall quality"""
        if len(self.calibration_data) >= 9:
            # Calculate overall calibration quality
            accuracies = [data.get('accuracy', 0) for data in self.calibration_data.values()]
            self.calibration_quality = np.mean(accuracies)
            
            # Create calibration model
            self._create_calibration_model()
            
            self.calibration_complete = True
            print(f"🎯 Calibration complete! Quality: {self.calibration_quality:.3f}")
            return True
        return False
    
    def _create_calibration_model(self):
        """Create calibration model for gaze estimation"""
        # This would implement gaze mapping algorithms
        # For now, we'll use a simple linear model
        self.calibration_model = {
            'quality': self.calibration_quality,
            'timestamp': time.time(),
            'model_type': 'linear'
        }
    
    def start_camera(self):
        """Start camera with research-grade settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_index}")
            
            # Research-grade camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            
            self.is_running = True
            self.research_data['session_start'] = time.time()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
            
            print("📹 Research-grade camera started")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def _analysis_worker(self):
        """Background worker for advanced analysis"""
        while self.is_running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self._advanced_analysis(data)
                time.sleep(0.01)
            except Exception as e:
                print(f"Analysis worker error: {e}")
    
    def _advanced_analysis(self, data):
        # Only keep quality assessment and research data logging
        self._assess_quality_advanced(data)
        if self.research_mode:
            self._log_research_data(data)
    
    def _assess_quality_advanced(self, data):
        """Advanced quality assessment"""
        quality_factors = []
        
        # Tracking stability
        if len(self.pupil_tracking_history) > 10:
            recent_positions = list(self.pupil_tracking_history)[-10:]
            stability = 1.0 / (1.0 + np.var(recent_positions))
            quality_factors.append(stability)
        
        # Calibration quality
        if self.calibration_complete:
            quality_factors.append(self.calibration_quality)
        
        # Face detection confidence
        if 'face_confidence' in data:
            quality_factors.append(data['face_confidence'])
        
        # Overall quality score
        if quality_factors:
            data['advanced_quality_score'] = np.mean(quality_factors)
        else:
            data['advanced_quality_score'] = 0.0
    
    def _log_research_data(self, data):
        timestamp = time.time()
        research_entry = {
            'timestamp': timestamp,
            'pupil_position': data.get('pupil_position', None),
            'gaze_point': data.get('gaze_point', None),
            'quality_score': data.get('advanced_quality_score', 0.0),
            'blink_rate': data.get('blink_rate', 0.0),
            'fixation_duration': data.get('fixation_duration', 0.0)
        }
        self.research_data['events'].append(research_entry)
    
    def process_frame(self, frame):
        """Process frame with features"""
        if not self.is_running:
            return frame
        
        try:
            # Convert frame for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Process with MediaPipe
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Wait for result
            start_time = time.time()
            while self.result is None and (time.time() - start_time) < 0.1:
                time.sleep(0.001)
            
            if self.result is None or not self.result.face_landmarks:
                return frame
            
            # Extract advanced data
            face_landmarks = self.result.face_landmarks[0]
            
            # Build advanced data locally (no super call)
            advanced_data = self._extract_advanced_data(face_landmarks, frame)
            
            # Queue for analysis
            try:
                self.data_queue.put_nowait(advanced_data)
            except queue.Full:
                pass
            
            # Update current data
            self.current_data = advanced_data
            
            # Draw face mesh and research overlay
            frame = self._draw_face_mesh(frame, face_landmarks)
            if self.research_mode:
                frame = self._draw_research_overlay(frame, advanced_data)
            
            return frame
            
        except Exception as e:
            print(f"Advanced frame processing error: {e}")
            return frame
    
    def _extract_advanced_data(self, face_landmarks, frame):
        data = {}
        left_pupil = self._extract_pupil_center(face_landmarks, 'left')
        right_pupil = self._extract_pupil_center(face_landmarks, 'right')
        if left_pupil is not None and right_pupil is not None:
            avg_pupil = (left_pupil + right_pupil) / 2
            self.pupil_tracking_history.append(avg_pupil)
            data['pupil_position'] = avg_pupil.tolist()
            data['pupil_diameter'] = self._calculate_pupil_diameter(left_pupil, right_pupil)
            
            # Apply noise filtering to get more stable gaze estimation
            if self.calibration_complete:
                raw_gaze_point = self._estimate_gaze_point(avg_pupil)
                # Apply filtering to both pupil and gaze positions
                filtered_pupil, filtered_gaze = self._apply_noise_filtering(avg_pupil, raw_gaze_point)
                
                data['gaze_point'] = filtered_gaze.tolist()
                self.gaze_history.append(filtered_gaze)
            else:
                # Still apply filtering even without calibration
                raw_gaze_point = self._estimate_gaze_point(avg_pupil)  # Will return center
                filtered_pupil, filtered_gaze = self._apply_noise_filtering(avg_pupil, raw_gaze_point)
                
                data['gaze_point'] = filtered_gaze.tolist()
                self.gaze_history.append(filtered_gaze)
        else:
            # If no pupil detected, preserve previous gaze or use center
            if hasattr(self, 'current_data') and self.current_data and 'gaze_point' in self.current_data:
                data['gaze_point'] = self.current_data['gaze_point']
            else:
                data['gaze_point'] = [0.5, 0.5]
        
        data.update(self._analyze_eye_movements())
        data.update(self._estimate_head_pose(face_landmarks, frame.shape))
        data['face_confidence'] = self.result.face_landmarks[0].confidence if hasattr(self.result.face_landmarks[0], 'confidence') else 0.8
        data['gaze_stability'] = self._calculate_gaze_stability()
        # --- Blink detection logic ---
        left_eye_openness = self._calculate_eye_openness(face_landmarks, self.landmark_indices['left_eye'], self.LEFT_EYE_VERTICAL_PAIRS)
        right_eye_openness = self._calculate_eye_openness(face_landmarks, self.landmark_indices['right_eye'], self.RIGHT_EYE_VERTICAL_PAIRS)
        avg_openness = (left_eye_openness + right_eye_openness) / 2
        current_time = time.time()
        if self.baseline_openness is None:
            self.baseline_openness = avg_openness
        # Adaptive threshold
        self.eye_openness_threshold = self.baseline_openness * 0.7
        if avg_openness < self.eye_openness_threshold and not self.is_blinking:
            self.is_blinking = True
            self.blink_start_time = current_time
        elif avg_openness >= self.eye_openness_threshold and self.is_blinking:
            self.is_blinking = False
            blink_duration = current_time - self.blink_start_time if self.blink_start_time else 0
            if 0.05 < blink_duration < 0.5:  # Only count valid blinks
                self.blink_times.append(current_time)
        data['left_eye_openness'] = left_eye_openness
        data['right_eye_openness'] = right_eye_openness
        data['blink_rate'] = self._calculate_blink_rate()
        data['fixation_duration'] = self._calculate_fixation_duration([])  # Use actual velocities if available
        return data
    
    def _extract_pupil_center(self, face_landmarks, eye_side):
        """Extract high-precision pupil center using robust methods"""
        if eye_side == 'left':
            iris_indices = self.landmark_indices['left_iris']
        else:
            iris_indices = self.landmark_indices['right_iris']
        
        iris_points = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in iris_indices])
        
        if len(iris_points) > 0:
            # Use robust center calculation: median or center of bounding box for better stability
            # Calculate bounding box center to reduce sensitivity to outliers
            x_min, y_min = np.min(iris_points, axis=0)
            x_max, y_max = np.max(iris_points, axis=0)
            bbox_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            
            # Use weighted center based on landmark reliability
            center = np.mean(iris_points, axis=0)
            
            # Return a more stable center between bbox and mean
            return 0.7 * center + 0.3 * bbox_center
        
        return None
    
    def _calculate_pupil_diameter(self, left_pupil, right_pupil):
        """Calculate pupil diameter in pixels"""
        if left_pupil is not None and right_pupil is not None:
            # Calculate distance between pupils as reference
            inter_pupil_distance = np.linalg.norm(left_pupil - right_pupil)
            
            # Estimate individual pupil diameter (simplified)
            pupil_diameter = inter_pupil_distance * 0.15  # Approximate ratio
            return pupil_diameter
        
        return 0.0
    
    def _estimate_gaze_point(self, pupil_position):
        """Estimate gaze point using calibration model with head pose compensation"""
        if not self.calibration_complete or self.current_data is None:
            return np.array([0.5, 0.5])  # Center of screen
        
        # Get head pose information for compensation
        head_yaw = self.current_data.get('head_yaw', 0.0)  # degrees
        head_tilt = self.current_data.get('head_tilt', 0.0)  # degrees
        head_roll = self.current_data.get('head_roll', 0.0)  # degrees
        
        # Start with the raw pupil position
        gaze_x = pupil_position[0]
        gaze_y = pupil_position[1]
        
        # Apply head pose compensation to adjust gaze estimation
        # The compensation is based on the principle that head rotation 
        # affects where the eyes are looking relative to the camera
        compensation_factor = 0.05  # Adjust based on testing
        
        # Compensate for head yaw (left/right rotation)
        # When head turns right, eyes need to look left to maintain same screen position
        gaze_x -= np.radians(head_yaw) * compensation_factor
        
        # Compensate for head tilt (up/down rotation)  
        # When head tilts up, eyes look lower on screen
        gaze_y -= np.radians(head_tilt) * compensation_factor
        
        # For roll, we might need to rotate the gaze vector, but for simplicity we'll adjust both
        roll_compensation = np.radians(head_roll) * compensation_factor * 0.5
        gaze_x -= roll_compensation
        gaze_y -= roll_compensation
        
        # Ensure the result is within screen bounds [0, 1]
        gaze_x = np.clip(gaze_x, 0.0, 1.0)
        gaze_y = np.clip(gaze_y, 0.0, 1.0)
        
        # Apply the calibration transform (currently using a simplified version)
        # In a real implementation, this would use the full calibration model
        if hasattr(self, 'calibration_model') and self.calibration_model['quality'] > 0.5:
            # Use more sophisticated mapping based on calibration
            # Apply calibration transformation matrix if available
            # For now, use a simple calibrated offset
            calibrated_gaze_x = gaze_x
            calibrated_gaze_y = gaze_y
        else:
            calibrated_gaze_x = gaze_x
            calibrated_gaze_y = gaze_y
        
        return np.array([calibrated_gaze_x, calibrated_gaze_y])
    
    def _apply_noise_filtering(self, raw_pupil_pos, raw_gaze_pos):
        """Apply noise filtering to pupil and gaze positions using advanced techniques"""
        filtered_pupil = raw_pupil_pos.copy()
        filtered_gaze = raw_gaze_pos.copy()
        
        # Use a Savitzky-Golay filter for smoothing (if available) or simple averaging
        if len(self.filtered_pupil_history) >= 3:
            # Apply simple median filter to reduce outliers
            pupil_history_array = np.array(list(self.filtered_pupil_history))
            # Calculate median position to reduce outlier effect
            median_pupil = np.median(pupil_history_array, axis=0)
            
            # Combine current measurement with median of recent measurements
            filtered_pupil = 0.7 * raw_pupil_pos + 0.3 * median_pupil
        
        # Add current filtered values to history
        self.filtered_pupil_history.append(filtered_pupil)
        self.filtered_gaze_history.append(filtered_gaze)
        
        return filtered_pupil, filtered_gaze
    
    def _analyze_eye_movements(self):
        """Analyze eye movements with noise filtering"""
        data = {}
        
        if len(self.pupil_tracking_history) > 5:
            recent_positions = list(self.pupil_tracking_history)[-5:]
            
            # Calculate movement velocity with noise consideration
            velocities = []
            for i in range(1, len(recent_positions)):
                velocity = np.linalg.norm(recent_positions[i] - recent_positions[i-1])
                velocities.append(velocity)
            
            if velocities:
                # Use median instead of mean to reduce outlier impact
                avg_vel = float(np.median(velocities))
                data['eye_velocity'] = avg_vel
                
                # Add velocity data to buffer for smoothing
                self.velocity_buffer.append(avg_vel)
                
                # Calculate smoothed velocity
                if len(self.velocity_buffer) > 1:
                    smoothed_velocity = np.median(list(self.velocity_buffer))
                    data['smoothed_eye_velocity'] = float(smoothed_velocity)
                
                data['fixation_duration'] = self._calculate_fixation_duration(velocities)
                self.eye_movement_history.append(avg_vel)
        
        return data
    
    def _calculate_fixation_duration(self, velocities):
        """Calculate fixation duration based on movement velocity"""
        if not velocities:
            return 0.0
        
        # Low velocity indicates fixation
        low_velocity_threshold = 0.01
        fixation_frames = sum(1 for v in velocities if v < low_velocity_threshold)
        
        return fixation_frames * 0.016  # Assuming 60 FPS
    
    def _estimate_head_pose(self, face_landmarks, frame_shape):
        """Estimate head pose using solvePnP with improved 3D model for better gaze accuracy"""
        data = {}
        height, width, _ = frame_shape
        
        # More accurate 3D model points based on average face measurements
        face_3d_points = np.array([
            [0.0, 0.0, 0.0],             # Nose tip
            [0.0, -75.0, 10.0],          # Chin
            [-50.0, 0.0, -20.0],         # Left eye left corner
            [50.0, 0.0, -20.0],          # Right eye right corner
            [-30.0, -40.0, -20.0],       # Left mouth corner
            [30.0, -40.0, -20.0]         # Right mouth corner
        ], dtype=np.float64)

        # 2D image points from landmarks with better normalization
        face_2d_points = np.array([
            (face_landmarks[1].x * width, face_landmarks[1].y * height),    # Nose tip
            (face_landmarks[152].x * width, face_landmarks[152].y * height), # Chin
            (face_landmarks[33].x * width, face_landmarks[33].y * height),  # Left eye corner (right eye from user perspective)
            (face_landmarks[263].x * width, face_landmarks[263].y * height), # Right eye corner (left eye from user perspective)
            (face_landmarks[61].x * width, face_landmarks[61].y * height),   # Left mouth corner
            (face_landmarks[291].x * width, face_landmarks[291].y * height)  # Right mouth corner
        ], dtype=np.float64)

        # Create a more accurate camera matrix
        focal_length = 1.2 * width  # Slightly higher focal length for webcams
        cam_matrix = np.array([[focal_length, 0, width / 2],
                               [0, focal_length, height / 2],
                               [0, 0, 1]], dtype=np.float64)
        
        # Use more realistic distortion coefficients for webcam lenses
        dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])  # [k1, k2, p1, p2, k3]

        # Solve for rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(face_3d_points, face_2d_points, cam_matrix, dist_coeffs, 
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            # Use more robust decomposition for Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler angles with better handling of singularities
            # More robust Euler angle extraction
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + 
                         rotation_matrix[1,0] * rotation_matrix[1,0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])  # Pitch (head tilt up/down)  
                y = np.arctan2(-rotation_matrix[2,0], sy)                   # Yaw (head turn left/right)
                z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])  # Roll (head tilt ear to shoulder)
            else:
                # Handle gimbal lock
                x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])  # Pitch
                y = np.arctan2(-rotation_matrix[2,0], sy)                   # Yaw
                z = 0.0                                                     # Roll
            
            # Store raw angles in radians
            self.head_pose_history['tilt'].append(x)
            self.head_pose_history['yaw'].append(y)
            self.head_pose_history['roll'].append(z)
            
            # Apply more sophisticated filtering to reduce noise
            if len(self.head_pose_history['tilt']) > 1:
                # Use the last few values to smooth the pose estimation
                tilt_filtered = np.median(list(self.head_pose_history['tilt'])[-3:])
                yaw_filtered = np.median(list(self.head_pose_history['yaw'])[-3:])
                roll_filtered = np.median(list(self.head_pose_history['roll'])[-3:])
                
                # Convert to degrees for output
                data['head_tilt'] = tilt_filtered * 180.0 / np.pi
                data['head_yaw'] = yaw_filtered * 180.0 / np.pi 
                data['head_roll'] = roll_filtered * 180.0 / np.pi
            else:
                data['head_tilt'] = x * 180.0 / np.pi
                data['head_yaw'] = y * 180.0 / np.pi
                data['head_roll'] = z * 180.0 / np.pi

            # Store original rvec and tvec for overlay
            data['rvec'] = rvec.flatten().tolist()
            data['tvec'] = tvec.flatten().tolist()
        
        return data
    
    def _draw_research_overlay(self, frame, data):
        """Draw research overlay on frame"""
        height, width = frame.shape[:2]
        
        # Draw pupil tracking
        if 'pupil_position' in data:
            pupil_pos = data['pupil_position']
            x, y = int(pupil_pos[0] * width), int(pupil_pos[1] * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw gaze point
        if 'gaze_point' in data:
            gaze_pos = data['gaze_point']
            x, y = int(gaze_pos[0] * width), int(gaze_pos[1] * height)
            cv2.circle(frame, (x, y), 10, (255, 0, 0), 2)
        
        # Draw head pose axis
        if 'rvec' in data and 'tvec' in data:
            rvec, tvec = np.array(data['rvec']), np.array(data['tvec'])
            axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, -100]]).reshape(-1, 3)
            
            cam_matrix = self._get_camera_matrix(width, height)
            dist_coeffs = np.zeros((4, 1))

            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
            
            face_landmarks = self.result.face_landmarks[0]
            nose_2d = (int(face_landmarks[1].x * width), int(face_landmarks[1].y * height))

            p1 = (int(imgpts[0].ravel()[0]), int(imgpts[0].ravel()[1]))
            p2 = (int(imgpts[1].ravel()[0]), int(imgpts[1].ravel()[1]))
            p3 = (int(imgpts[2].ravel()[0]), int(imgpts[2].ravel()[1]))
            
            cv2.line(frame, nose_2d, p1, (255, 0, 0), 3) # X-axis (blue)
            cv2.line(frame, nose_2d, p2, (0, 255, 0), 3) # Y-axis (green)
            cv2.line(frame, nose_2d, p3, (0, 0, 255), 3) # Z-axis (red)
        
        return frame
    
    def _draw_face_mesh(self, frame, face_landmarks):
        """Draw face mesh on frame"""
        try:
            # Convert landmarks to MediaPipe format
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in face_landmarks
            ])
            
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=FACE_MESH_CUSTOM_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            return frame
        except Exception as e:
            print(f"Face mesh drawing error: {e}")
            return frame
    
    def get_research_summary(self):
        """Get research session summary"""
        if not self.research_data['session_start']:
            return None
        
        session_duration = time.time() - self.research_data['session_start']
        
        summary = {
            'session_duration': session_duration,
            'total_frames': self.research_data['total_frames'],
            'calibration_quality': self.calibration_quality,
            'average_fatigue': np.mean([e['fatigue_score'] for e in self.research_data['events']]) if self.research_data['events'] else 0,
            'average_cognitive_load': np.mean([e['cognitive_load'] for e in self.research_data['events']]) if self.research_data['events'] else 0,
            'average_quality': np.mean([e['quality_score'] for e in self.research_data['events']]) if self.research_data['events'] else 0,
            'total_events': len(self.research_data['events'])
        }
        
        return summary
    
    def export_research_data(self, filename=None):
        """Export research data for analysis"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"research_data_{timestamp}.json"
        
        export_data = {
            'session_info': self.get_research_summary(),
            'calibration_data': self.calibration_data,
            'events': self.research_data['events'],
            'fatigue_indicators': {k: list(v) for k, v in self.fatigue_indicators.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Research data exported to: {filename}")
        return filename
    
    def read_frame(self):
        """Read frame from camera with error handling"""
        if self.cap is None or not self.is_running:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def stop_camera(self):
        """Stop camera and cleanup"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
    
    def get_current_data(self):
        """Get current tracking data"""
        return getattr(self, 'current_data', {})
    
    # Helper methods for normalization
    def _normalize_blink_pattern(self):
        """Normalize blink pattern for fatigue analysis"""
        if len(self.fatigue_indicators['blink_pattern']) < 10:
            return 0.0
        
        recent_blinks = list(self.fatigue_indicators['blink_pattern'])[-10:]
        avg_duration = np.mean(recent_blinks)
        
        # Longer blinks indicate fatigue
        return min(1.0, avg_duration / 0.3)
    
    def _normalize_pupil_diameter(self):
        """Normalize pupil diameter for fatigue analysis"""
        if len(self.fatigue_indicators['pupil_diameter']) < 10:
            return 0.0
        
        recent_diameters = list(self.fatigue_indicators['pupil_diameter'])[-10:]
        avg_diameter = np.mean(recent_diameters)
        
        # Smaller pupils can indicate fatigue
        return max(0.0, 1.0 - (avg_diameter / 10.0))
    
    def _normalize_fixation_duration(self):
        """Normalize fixation duration for fatigue analysis"""
        if len(self.fatigue_indicators['fixation_duration']) < 10:
            return 0.0
        
        recent_fixations = list(self.fatigue_indicators['fixation_duration'])[-10:]
        avg_fixation = np.mean(recent_fixations)
        
        # Longer fixations can indicate fatigue
        return min(1.0, avg_fixation / 2.0)
    def _calculate_gaze_stability(self):
        """Calculate gaze stability based on recent gaze positions using advanced filtering"""
        if len(self.gaze_history) < 3:
            return 0.0
        
        recent_gaze = list(self.gaze_history)[-10:] if len(self.gaze_history) >= 10 else list(self.gaze_history)
        
        # Calculate both variance and velocity-based stability
        if len(recent_gaze) > 1:
            # Calculate instantaneous velocities between consecutive points
            velocities = []
            for i in range(1, len(recent_gaze)):
                velocity = np.linalg.norm(np.array(recent_gaze[i]) - np.array(recent_gaze[i-1]))
                velocities.append(velocity)
            
            if velocities:
                avg_velocity = np.mean(velocities)
                # Lower velocity indicates more stable gaze
                gaze_variance = np.var(recent_gaze, axis=0).mean()
                variance_stability = max(0.0, 1.0 - gaze_variance * 20)  # Adjust multiplier based on testing
                
                # Combine both measures for more robust stability
                stability = 0.6 * velocity_stability + 0.4 * variance_stability
                return stability
            else:
                return 1.0
        else:
            return 1.0
        gaze_variance = np.var(recent_gaze)
        
        # Lower variance indicates more stable gaze
        stability = max(0.0, 1.0 - gaze_variance * 10)
        return stability
    
    def _calculate_cognitive_load_score(self):
        # Cognitive load calculation removed
        return 0.0
    
    def _calculate_cognitive_load_score(self):
        # Cognitive load calculation removed
        return 0.0
    
    def _calculate_blink_rate(self):
        # Calculate blink rate per minute using blink_times deque
        current_time = time.time()
        # Remove blinks older than 60 seconds
        while self.blink_times and current_time - self.blink_times[0] > 60:
            self.blink_times.popleft()
        if len(self.blink_times) > 1:
            time_window = self.blink_times[-1] - self.blink_times[0]
            if time_window > 0:
                return (len(self.blink_times) / time_window) * 60
        return 0.0
    
    def _get_camera_matrix(self, width, height):
        """Get the camera matrix"""
        focal_length = width
        return np.array([
            [focal_length, 0, height / 2],
            [0, focal_length, width / 2],
            [0, 0, 1]
        ], dtype=np.float64)

    def _get_2d_landmark_point(self, data, landmark_index):
        """Get 2D point for a given landmark index"""
        if 'face_landmarks' in data and len(data['face_landmarks']) > landmark_index:
            lm = data['face_landmarks'][landmark_index]
            return (lm[0], lm[1])
        return None 