import csv
import os
from datetime import datetime
from collections import deque

class DataLogger:
    def __init__(self, data_dir="face_eye_tracker/data"):
        self.data_dir = data_dir
        self.csv_file = None
        self.writer = None
        self.csv_handle = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data buffers for real-time analysis
        self.time_data = deque(maxlen=100)
        self.left_eye_data = deque(maxlen=100)
        self.right_eye_data = deque(maxlen=100)
        self.blink_rate_data = deque(maxlen=100)
        self.fatigue_data = deque(maxlen=100)
        self.quality_data = deque(maxlen=100)
        
    def start_logging(self):
        """Start logging to a new CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"eye_tracking_data_{timestamp}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        self.csv_file = filepath
        self.csv_handle = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.csv_handle)
        
        # Write header
        self.writer.writerow([
            'timestamp', 'left_eye_openness', 'right_eye_openness', 'blink_detected', 
            'blink_rate', 'quality_score', 'baseline_openness', 'confidence', 'blink_duration',
            'saccade_detected', 'saccade_rate', 'saccade_amplitude', 'saccade_velocity',
            'microsaccade_detected', 'microsaccade_rate',
            'fatigue_blink_rate_score', 'fatigue_blink_duration_score', 'fatigue_eye_openness_score',
            'fatigue_yawn_score', 'fatigue_gaze_drift_score', 'fatigue_head_slouch_score',
            'overall_fatigue_score', 'fatigue_level', 'yawn_detected'
        ])
        
        print(f"Started logging to: {filepath}")
        
    def log_data(self, data):
        """Log a single data point"""
        if not self.writer:
            return
            
        timestamp = datetime.now().isoformat()
        
        # Write to CSV
        self.writer.writerow([
            timestamp, data['left_eye_openness'], data['right_eye_openness'], data['blink_detected'], 
            data['blink_rate'], data['quality_score'], data['baseline_openness'], 0.9, data['blink_duration'],
            data['saccade_detected'], data['saccade_rate'], data['saccade_amplitude'], data['saccade_velocity'],
            data['microsaccade_detected'], data['microsaccade_rate'],
            data['fatigue_blink_rate_score'], data['fatigue_blink_duration_score'], data['fatigue_eye_openness_score'],
            data['fatigue_yawn_score'], data['fatigue_gaze_drift_score'], data['fatigue_head_slouch_score'],
            data['overall_fatigue_score'], data['fatigue_level'], data['yawn_detected']
        ])
        
        # Update real-time data buffers
        current_time = datetime.now().timestamp()
        self.time_data.append(current_time)
        self.left_eye_data.append(data['left_eye_openness'])
        self.right_eye_data.append(data['right_eye_openness'])
        self.blink_rate_data.append(data['blink_rate'])
        self.fatigue_data.append(data['overall_fatigue_score'])
        self.quality_data.append(data['quality_score'])
        
    def get_chart_data(self):
        """Get data for real-time charts"""
        return {
            'time': list(self.time_data),
            'left_eye': list(self.left_eye_data),
            'right_eye': list(self.right_eye_data),
            'blink_rate': list(self.blink_rate_data),
            'fatigue': list(self.fatigue_data),
            'quality': list(self.quality_data)
        }
        
    def stop_logging(self):
        """Stop logging and close the CSV file"""
        if self.csv_handle:
            self.csv_handle.close()
            print(f"Stopped logging. Data saved to: {self.csv_file}")
            self.csv_handle = None
            self.writer = None 