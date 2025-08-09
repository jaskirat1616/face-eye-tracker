import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from datetime import datetime
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
FACE_MESH_TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION

# More efficient drawing by selecting specific feature connections
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

class FaceEyeTracker:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.is_running = False
        self.cap = None
        
        # Performance optimizations
        self.frame_skip = 0  # Skip every nth frame for performance
        self.frame_count = 0
        self.enable_drawing = True  # Can be disabled for performance
        self.enable_data_logging = False  # Disabled by default for performance
        self.stability_mode = True  # Enable stability mode by default
        
        # MediaPipe Face Landmarker with optimized settings
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'face_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self.result_callback
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.result = None
        self.timestamp = 0

        # Eye indices (MediaPipe 468 landmarks)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris indices (MediaPipe 478 landmarks)
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]

        # Key vertical pairs for eye openness
        self.LEFT_EYE_VERTICAL_PAIRS = [(1, 5), (2, 4), (3, 7), (8, 12), (9, 11), (0, 6)]
        self.RIGHT_EYE_VERTICAL_PAIRS = [(1, 5), (2, 4), (3, 7), (8, 12), (9, 11), (0, 6)]

        # Blink detection
        self.blink_counter = 0
        self.blink_times = deque(maxlen=100)
        self.blink_durations = deque(maxlen=10)
        self.last_blink_time = time.time()
        self.blink_start_time = None
        self.is_blinking = False
        self.eye_openness_threshold = 0.25
        self.baseline_openness = None
        self.openness_history = deque(maxlen=50)
        
        # Saccade detection
        self.saccade_counter = 0
        self.saccade_times = deque(maxlen=100)
        self.last_pupil_pos = None
        self.last_frame_time = time.time()
        self.saccade_threshold = 0.02
        self.saccade_cooldown = 0.3
        self.last_saccade_time = 0
        self.saccade_amplitudes = deque(maxlen=10)
        self.saccade_velocities = deque(maxlen=10)

        # Microsaccade detection
        self.microsaccade_counter = 0
        self.microsaccade_times = deque(maxlen=100)
        self.last_pupil_pos_local = None
        self.microsaccade_threshold = 0.2
        self.last_microsaccade_time = 0

        # Blink validation parameters
        self.min_blink_duration = 0.04
        self.max_blink_duration = 0.4
        self.blink_cooldown = 0.2
        self.last_valid_blink_time = 0
        
        # Quality monitoring
        self.quality_score = 1.0
        self.stable_frames = 0
        self.last_landmarks = None
        
        # Stability improvements
        self.landmark_stability_threshold = 0.02  # Increased threshold for stability
        
        # Temporal smoothing
        self.smoothing_buffer = deque(maxlen=10)  # Increased for more stability
        
        # Blink detection state
        self.calibration_frames = 0
        self.calibration_complete = False
        self.eye_openness_buffer = deque(maxlen=10)
        
        # Remove fatigue detection variables and scores
        # Only keep blink, saccade, and fixation related variables
        self.blink_counter = 0
        self.blink_times = deque(maxlen=100)
        self.blink_durations = deque(maxlen=10)
        self.last_blink_time = time.time()
        self.blink_start_time = None
        self.is_blinking = False
        self.eye_openness_threshold = 0.25
        self.baseline_openness = None
        self.openness_history = deque(maxlen=50)
        self.saccade_counter = 0
        self.saccade_times = deque(maxlen=100)
        self.last_pupil_pos = None
        self.last_frame_time = time.time()
        self.saccade_threshold = 0.02
        self.saccade_cooldown = 0.3
        self.last_saccade_time = 0
        self.saccade_amplitudes = deque(maxlen=10)
        self.saccade_velocities = deque(maxlen=10)
        self.microsaccade_counter = 0
        self.microsaccade_times = deque(maxlen=100)
        self.last_pupil_pos_local = None
        self.microsaccade_threshold = 0.2
        self.last_microsaccade_time = 0
        self.current_data = {}

    def result_callback(self, result: vision.FaceLandmarkerResult, output_image, timestamp_ms: int):
        self.result = result
        self.timestamp = timestamp_ms

    def get_eye_bounding_box(self, face_landmarks, eye_indices):
        """Calculates the bounding box for a given eye."""
        eye_points = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in eye_indices])
        x_min, y_min = eye_points.min(axis=0)
        x_max, y_max = eye_points.max(axis=0)
        return x_min, y_min, x_max - x_min, y_max - y_min

    def calculate_eye_openness_accurate(self, face_landmarks, eye_indices, vertical_pairs):
        """Calculate eye openness using multiple vertical pairs for accuracy"""
        eye_points = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in eye_indices])
        
        # Calculate multiple vertical distances
        vertical_distances = []
        for pair in vertical_pairs:
            if pair[0] < len(eye_points) and pair[1] < len(eye_points):
                dist = np.abs(eye_points[pair[0]][1] - eye_points[pair[1]][1])
                vertical_distances.append(dist)
        
        if not vertical_distances:
            # Fallback to min/max
            vertical_distance = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
        else:
            # Use weighted average (more weight to center pairs)
            weights = [1.0, 1.2, 1.5, 1.2, 1.0, 0.8]
            weights = weights[:len(vertical_distances)]
            vertical_distance = np.average(vertical_distances, weights=weights)
        
        # Horizontal distance (eye width)
        horizontal_distance = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
        
        if horizontal_distance > 0:
            openness_ratio = vertical_distance / horizontal_distance
        else:
            openness_ratio = 0
            
        return openness_ratio

    def update_baseline_openness(self, left_openness, right_openness):
        """Update adaptive baseline for blink detection with better calibration"""
        avg_openness = (left_openness + right_openness) / 2
        self.openness_history.append(avg_openness)
        self.eye_openness_buffer.append(avg_openness)
        
        # Wait for calibration to complete
        if len(self.openness_history) < 20:
            self.calibration_frames += 1
            return
        
        # Calculate baseline as 30th percentile (eyes are usually open)
        baseline = np.percentile(list(self.openness_history), 30)
        
        if self.baseline_openness is None:
            self.baseline_openness = baseline
            self.calibration_complete = True
        else:
            # Smooth baseline update (slower adaptation)
            self.baseline_openness = 0.95 * self.baseline_openness + 0.05 * baseline
        
        # Adaptive threshold: 90% of baseline (much more sensitive)
        self.eye_openness_threshold = self.baseline_openness * 0.9

    def validate_blink_stability(self):
        """Check if eye openness is stable enough for reliable blink detection"""
        if len(self.eye_openness_buffer) < 5:
            return False
        
        # Calculate variance of recent measurements
        recent_values = list(self.eye_openness_buffer)[-5:]
        variance = np.var(recent_values)
        return variance < 0.05

    def detect_blink_accurate(self, left_openness, right_openness):
        """Enhanced blink detection with comprehensive validation"""
        current_time = time.time()
        avg_openness = (left_openness + right_openness) / 2
        
        # Update baseline
        self.update_baseline_openness(left_openness, right_openness)
        
        # Don't detect blinks during calibration
        if not self.calibration_complete:
            return False, 0.0
        
        # Check stability before detecting blinks
        if not self.validate_blink_stability():
            return False, 0.0
        
        blink_detected = False
        blink_duration = 0.0
        
        # Check if eyes are closed (blink start)
        if avg_openness < self.eye_openness_threshold and not self.is_blinking:
            self.is_blinking = True
            self.blink_start_time = current_time
        
        # Check if eyes are open again (blink end)
        elif avg_openness >= self.eye_openness_threshold and self.is_blinking:
            self.is_blinking = False
            blink_duration = current_time - self.blink_start_time
            
            # Comprehensive blink validation
            if (self.min_blink_duration <= blink_duration <= self.max_blink_duration and
                current_time - self.last_valid_blink_time > self.blink_cooldown):
                
                # Additional validation: check if eyes returned to normal openness
                if avg_openness > self.baseline_openness * 0.8:
                    self.blink_counter += 1
                    self.blink_times.append(current_time)
                    self.blink_durations.append(blink_duration)
                    self.last_blink_time = current_time
                    self.last_valid_blink_time = current_time
                    blink_detected = True
        
        return blink_detected, blink_duration

    def calculate_blink_rate(self):
        """Calculate blink rate per minute"""
        current_time = time.time()
        
        # Remove blinks older than 60 seconds
        while self.blink_times and current_time - self.blink_times[0] > 60:
            self.blink_times.popleft()
        
        if len(self.blink_times) > 0:
            time_window = current_time - self.blink_times[0]
            if time_window > 0:
                return (len(self.blink_times) / time_window) * 60
        return 0

    def detect_saccade(self, face_landmarks):
        """Detect eye saccades and measure their amplitude and velocity."""
        current_time = time.time()

        # We need at least 478 landmarks for iris tracking.
        if len(face_landmarks) < 478:
            return False, 0.0, 0.0

        # Calculate pupil centers by averaging the iris landmark positions.
        left_pupil = np.mean([[lm.x, lm.y] for lm in [face_landmarks[i] for i in self.LEFT_IRIS]], axis=0)
        right_pupil = np.mean([[lm.x, lm.y] for lm in [face_landmarks[i] for i in self.RIGHT_IRIS]], axis=0)
        avg_pupil_pos = (left_pupil + right_pupil) / 2

        saccade_detected = False
        amplitude = 0.0
        velocity = 0.0

        if self.last_pupil_pos is not None:
            delta_time = current_time - self.last_frame_time
            if delta_time > 0:
                # Amplitude is the Euclidean distance between pupil positions.
                amplitude = np.linalg.norm(avg_pupil_pos - self.last_pupil_pos)
                velocity = amplitude / delta_time

                # A saccade is detected if the velocity exceeds the threshold and is not on cooldown.
                if (velocity > self.saccade_threshold and
                    current_time - self.last_saccade_time > self.saccade_cooldown):
                    
                    saccade_detected = True
                    self.saccade_counter += 1
                    self.saccade_times.append(current_time)
                    self.last_saccade_time = current_time
                    self.saccade_amplitudes.append(amplitude)
                    self.saccade_velocities.append(velocity)

        self.last_pupil_pos = avg_pupil_pos
        self.last_frame_time = current_time
        
        return saccade_detected, amplitude, velocity

    def detect_microsaccade(self, face_landmarks):
        """Detects fine-grained saccades using local-to-eye coordinates."""
        current_time = time.time()
        microsaccade_detected = False

        if len(face_landmarks) < 478:
            return False

        # Get eye bounding boxes
        left_bbox = self.get_eye_bounding_box(face_landmarks, self.LEFT_EYE)
        right_bbox = self.get_eye_bounding_box(face_landmarks, self.RIGHT_EYE)
        
        # Get global pupil positions
        left_pupil = np.mean([[lm.x, lm.y] for lm in [face_landmarks[i] for i in self.LEFT_IRIS]], axis=0)
        right_pupil = np.mean([[lm.x, lm.y] for lm in [face_landmarks[i] for i in self.RIGHT_IRIS]], axis=0)

        # Re-normalize pupil positions to be local to their eye's bounding box
        if left_bbox[2] > 0 and right_bbox[2] > 0:
            local_left_pupil = np.array([(left_pupil[0] - left_bbox[0]) / left_bbox[2], (left_pupil[1] - left_bbox[1]) / left_bbox[3]])
            local_right_pupil = np.array([(right_pupil[0] - right_bbox[0]) / right_bbox[2], (right_pupil[1] - right_bbox[1]) / right_bbox[3]])
            avg_pupil_pos_local = (local_left_pupil + local_right_pupil) / 2.0
            
            if self.last_pupil_pos_local is not None:
                delta_time = current_time - self.last_frame_time
                if delta_time > 0:
                    local_velocity = np.linalg.norm(avg_pupil_pos_local - self.last_pupil_pos_local) / delta_time
                    
                    # Detect based on local velocity
                    if (local_velocity > self.microsaccade_threshold and
                        current_time - self.last_microsaccade_time > self.saccade_cooldown and
                        current_time - self.last_saccade_time > self.saccade_cooldown):
                        
                        microsaccade_detected = True
                        self.microsaccade_counter += 1
                        self.microsaccade_times.append(current_time)
                        self.last_microsaccade_time = current_time
            
            self.last_pupil_pos_local = avg_pupil_pos_local

        return microsaccade_detected

    def calculate_saccade_rate(self):
        """Calculate the saccade rate per minute."""
        current_time = time.time()
        
        # Remove saccades older than 60 seconds to keep the window current.
        while self.saccade_times and current_time - self.saccade_times[0] > 60:
            self.saccade_times.popleft()
        
        if len(self.saccade_times) > 1:
            time_window = current_time - self.saccade_times[0]
            if time_window > 0:
                return (len(self.saccade_times) / time_window) * 60
        return 0

    def calculate_microsaccade_rate(self):
        """Calculate the microsaccade rate per minute."""
        current_time = time.time()
        
        while self.microsaccade_times and current_time - self.microsaccade_times[0] > 60:
            self.microsaccade_times.popleft()
        
        if len(self.microsaccade_times) > 1:
            time_window = current_time - self.microsaccade_times[0]
            if time_window > 0:
                return (len(self.microsaccade_times) / time_window) * 60
        return 0

    def assess_quality(self, face_landmarks):
        """Assess detection quality and stability"""
        if self.last_landmarks is None:
            self.last_landmarks = face_landmarks
            return 1.0
        
        # Calculate landmark stability
        current_points = np.array([[lm.x, lm.y] for lm in face_landmarks])
        last_points = np.array([[lm.x, lm.y] for lm in self.last_landmarks])
        
        # Mean displacement
        displacement = np.mean(np.linalg.norm(current_points - last_points, axis=1))
        
        # Update quality score (lower displacement = higher quality)
        if displacement < self.landmark_stability_threshold:  # Very stable
            self.quality_score = min(1.0, self.quality_score + 0.03)  # Slower increase
            self.stable_frames += 1
        elif displacement > 0.08:  # Unstable (increased threshold)
            self.quality_score = max(0.1, self.quality_score - 0.05)  # Slower decrease
            self.stable_frames = 0
        else:
            self.quality_score = max(0.1, self.quality_score - 0.01)  # Very slow decrease
        
        self.last_landmarks = face_landmarks
        return self.quality_score

    def smooth_measurements(self, left_openness, right_openness):
        """Apply temporal smoothing to measurements"""
        self.smoothing_buffer.append((left_openness, right_openness))
        
        if len(self.smoothing_buffer) < 5:
            return left_openness, right_openness
        
        # Use stronger smoothing for stability
        alpha = 0.5  # Reduced from 0.7 for more stability
        smoothed_left = alpha * left_openness + (1 - alpha) * np.mean([x[0] for x in list(self.smoothing_buffer)[:-1]])
        smoothed_right = alpha * right_openness + (1 - alpha) * np.mean([x[1] for x in list(self.smoothing_buffer)[:-1]])
        
        return smoothed_left, smoothed_right

    def get_current_data(self):
        """Get current tracking data for UI display"""
        return self.current_data
    
    def set_stability_mode(self, enabled=True):
        """Enable or disable stability mode"""
        self.stability_mode = enabled
        if enabled:
            # Increase smoothing and thresholds for stability
            self.smoothing_buffer = deque(maxlen=15)
            self.landmark_stability_threshold = 0.03
        else:
            # Use default settings for performance
            self.smoothing_buffer = deque(maxlen=5)
            self.landmark_stability_threshold = 0.01

    def process_frame(self, frame):
        """Optimized frame processing with performance improvements"""
        self.frame_count += 1
        
        # Process every frame for stability (reduced frame skipping)
        # if self.frame_count % 2 == 0:
        #     return frame
        
        try:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Process with MediaPipe
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Wait for result (with timeout)
            start_time = time.time()
            while self.result is None and (time.time() - start_time) < 0.1:  # Increased timeout for stability
                time.sleep(0.001)
            
            if self.result is None or not self.result.face_landmarks:
                # No face detected
                self.current_data = {
                    'left_eye_openness': 0.0,
                    'right_eye_openness': 0.0,
                    'blink_rate': 0.0,
                    'saccade_rate': 0.0,
                    'microsaccade_rate': 0.0
                }
                return frame
            
            face_landmarks = self.result.face_landmarks[0]
            
            # Assess quality
            quality = self.assess_quality(face_landmarks)
            
            # Calculate accurate eye openness
            left_eye_openness = self.calculate_eye_openness_accurate(face_landmarks, self.LEFT_EYE, self.LEFT_EYE_VERTICAL_PAIRS)
            right_eye_openness = self.calculate_eye_openness_accurate(face_landmarks, self.RIGHT_EYE, self.RIGHT_EYE_VERTICAL_PAIRS)
            
            # Apply smoothing
            left_eye_openness, right_eye_openness = self.smooth_measurements(left_eye_openness, right_eye_openness)
            
            # Detect blink
            blink_detected, blink_duration = self.detect_blink_accurate(left_eye_openness, right_eye_openness)
            blink_rate = self.calculate_blink_rate()
            
            # Detect saccade
            saccade_detected, saccade_amplitude, saccade_velocity = self.detect_saccade(face_landmarks)
            saccade_rate = self.calculate_saccade_rate()
            
            # Detect microsaccade
            microsaccade_detected = self.detect_microsaccade(face_landmarks)
            microsaccade_rate = self.calculate_microsaccade_rate()
            
            # Update current data for UI
            self.current_data = {
                'left_eye_openness': left_eye_openness,
                'right_eye_openness': right_eye_openness,
                'blink_rate': blink_rate,
                'saccade_rate': saccade_rate,
                'microsaccade_rate': microsaccade_rate,
                'blink_duration': blink_duration,
                'saccade_amplitude': saccade_amplitude,
                'saccade_velocity': saccade_velocity,
                'blink_counter': self.blink_counter,
                'saccade_counter': self.saccade_counter,
                'blink_durations': list(self.blink_durations)
            }
            
            # Draw face mesh if enabled
            if self.enable_drawing:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in face_landmarks
                ])
                
                # Use optimized drawing (only essential connections)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks_proto,
                    connections=FACE_MESH_CUSTOM_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

    def start_camera(self):
        """Start camera with optimized settings for maximum performance"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera with index {self.camera_index}")
            
            # Optimize camera settings for maximum performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            
            # Optimize camera settings for stability
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus for stability
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure for better lighting
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)  # Auto brightness
            self.cap.set(cv2.CAP_PROP_CONTRAST, -1)  # Auto contrast
            
            # Use stable codec
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

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