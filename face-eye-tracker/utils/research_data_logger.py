#!/usr/bin/env python3
"""
Advanced Research Data Logger for Eye Tracking & Cognitive Load Detection
========================================================================

Professional research-grade data logging system with advanced features:
- Comprehensive data collection
- Real-time analysis
- Export capabilities
- Quality assessment
- Session management
"""

import csv
import json
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import time
import pandas as pd

class ResearchDataLogger:
    """
    Advanced research-grade data logger for comprehensive data collection
    """
    
    def __init__(self, data_dir="research_data", enable_logging=True):
        self.data_dir = data_dir
        self.enable_logging = enable_logging
        self.session_id = None
        self.session_start_time = None
        
        # Data storage
        self.raw_data = []
        self.processed_data = []
        self.annotations = []
        self.quality_metrics = []
        self.session_events = []
        
        # Real-time analysis buffers
        self.analysis_buffers = {
            'fatigue_scores': deque(maxlen=1000),
            'quality_scores': deque(maxlen=1000),
            'pupil_diameters': deque(maxlen=1000),
            'gaze_positions': deque(maxlen=1000),
            'eye_velocities': deque(maxlen=1000),
            'blink_events': deque(maxlen=1000),
            'saccade_events': deque(maxlen=1000)
        }
        
        # Statistical tracking
        self.statistics = {
            'session_duration': 0,
            'total_frames': 0,
            'average_fatigue': 0,
            'average_quality': 0,
            'total_blinks': 0,
            'total_saccades': 0,
            'calibration_quality': 0
        }
        
        # Export formats
        self.export_formats = ['json', 'csv', 'pickle', 'excel']
        
        # Create data directory
        if self.enable_logging:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'sessions'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'exports'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'analysis'), exist_ok=True)
    
    def start_session(self, session_name=None):
        """Start a new research session"""
        if not self.enable_logging:
            return
        
        self.session_start_time = time.time()
        self.session_id = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize session data
        self.raw_data = []
        self.processed_data = []
        self.annotations = []
        self.quality_metrics = []
        self.session_events = []
        
        # Reset analysis buffers
        for buffer_name in self.analysis_buffers:
            self.analysis_buffers[buffer_name].clear()
        
        # Reset statistics
        for stat_name in self.statistics:
            self.statistics[stat_name] = 0
        
        # Log session start event
        self.log_session_event("session_start", {
            "session_id": self.session_id,
            "timestamp": self.session_start_time,
            "description": "Research session started"
        })
        
        print(f"📊 Research session started: {self.session_id}")
    
    def log_raw_data(self, data):
        """Log raw tracking data"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        timestamp = time.time()
        session_time = timestamp - self.session_start_time
        
        # Add metadata
        raw_entry = {
            'timestamp': timestamp,
            'session_time': session_time,
            'data': data.copy()
        }
        
        self.raw_data.append(raw_entry)
        self.statistics['total_frames'] += 1
        
        # Update analysis buffers
        self._update_analysis_buffers(data)
        
        # Update statistics
        self._update_statistics(data)
    
    def log_processed_data(self, data):
        """Log processed/analyzed data"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        timestamp = time.time()
        session_time = timestamp - self.session_start_time
        
        processed_entry = {
            'timestamp': timestamp,
            'session_time': session_time,
            'data': data.copy()
        }
        
        self.processed_data.append(processed_entry)
    
    def log_annotation(self, annotation_text, annotation_type="general"):
        """Log research annotation"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        timestamp = time.time()
        session_time = timestamp - self.session_start_time
        
        annotation = {
            'timestamp': timestamp,
            'session_time': session_time,
            'text': annotation_text,
            'type': annotation_type
        }
        
        self.annotations.append(annotation)
        
        # Log as session event
        self.log_session_event("annotation", annotation)
        
        print(f"📝 Annotation logged: {annotation_text}")
    
    def log_quality_metric(self, metric_name, value, threshold=None):
        """Log quality metric"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        timestamp = time.time()
        session_time = timestamp - self.session_start_time
        
        quality_entry = {
            'timestamp': timestamp,
            'session_time': session_time,
            'metric_name': metric_name,
            'value': value,
            'threshold': threshold,
            'status': 'good' if threshold is None or value >= threshold else 'poor'
        }
        
        self.quality_metrics.append(quality_entry)
    
    def log_session_event(self, event_type, event_data):
        """Log session event"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        timestamp = time.time()
        session_time = timestamp - self.session_start_time
        
        event = {
            'timestamp': timestamp,
            'session_time': session_time,
            'event_type': event_type,
            'event_data': event_data
        }
        
        self.session_events.append(event)
    
    def _update_analysis_buffers(self, data):
        """Update real-time analysis buffers"""
        # Fatigue scores
        if 'advanced_fatigue_score' in data:
            self.analysis_buffers['fatigue_scores'].append(data['advanced_fatigue_score'])
        
        # Quality scores
        if 'advanced_quality_score' in data:
            self.analysis_buffers['quality_scores'].append(data['advanced_quality_score'])
        
        # Pupil diameters
        if 'pupil_diameter' in data:
            self.analysis_buffers['pupil_diameters'].append(data['pupil_diameter'])
        
        # Gaze positions
        if 'gaze_point' in data:
            self.analysis_buffers['gaze_positions'].append(data['gaze_point'])
        
        # Eye velocities
        if 'eye_velocity' in data:
            self.analysis_buffers['eye_velocities'].append(data['eye_velocity'])
    
    def _update_statistics(self, data):
        """Update session statistics"""
        # Update averages
        if self.analysis_buffers['fatigue_scores']:
            self.statistics['average_fatigue'] = np.mean(list(self.analysis_buffers['fatigue_scores']))
        
        if self.analysis_buffers['quality_scores']:
            self.statistics['average_quality'] = np.mean(list(self.analysis_buffers['quality_scores']))
        
        # Update session duration
        if self.session_start_time:
            self.statistics['session_duration'] = time.time() - self.session_start_time
    
    def get_real_time_analysis(self):
        """Get real-time analysis results"""
        analysis = {}
        
        # Calculate trends
        for buffer_name, buffer_data in self.analysis_buffers.items():
            if len(buffer_data) > 10:
                recent_data = list(buffer_data)[-10:]
                analysis[f'{buffer_name}_trend'] = np.mean(recent_data)
                analysis[f'{buffer_name}_variance'] = np.var(recent_data)
        
        # Calculate session statistics
        analysis.update(self.statistics)
        
        return analysis
    
    def get_session_summary(self):
        """Get comprehensive session summary"""
        if not self.session_start_time:
            return None
        
        session_duration = time.time() - self.session_start_time
        
        # Calculate averages from analysis buffers
        avg_fatigue = np.mean(list(self.analysis_buffers['fatigue_scores'])) if self.analysis_buffers['fatigue_scores'] else 0
        avg_quality = np.mean(list(self.analysis_buffers['quality_scores'])) if self.analysis_buffers['quality_scores'] else 0
        
        summary = {
            'session_id': self.session_id,
            'session_start': datetime.fromtimestamp(self.session_start_time).isoformat(),
            'session_duration': session_duration,
            'total_frames': self.statistics['total_frames'],
            'total_annotations': len(self.annotations),
            'total_events': len(self.session_events),
            'statistics': self.statistics.copy(),
            'quality_metrics': {
                'average_quality': avg_quality,
                'quality_trend': np.mean(list(self.analysis_buffers['quality_scores'][-50:])) if len(self.analysis_buffers['quality_scores']) >= 50 else avg_quality
            },
            'fatigue_analysis': {
                'average_fatigue': avg_fatigue,
                'fatigue_trend': np.mean(list(self.analysis_buffers['fatigue_scores'][-50:])) if len(self.analysis_buffers['fatigue_scores']) >= 50 else avg_fatigue
            }
        }
        
        return summary
    
    def export_session_data(self, format='json', filename=None):
        """Export session data in various formats"""
        if not self.enable_logging or not self.session_id:
            return None
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.session_id}_{timestamp}"
        
        export_path = os.path.join(self.data_dir, 'exports', filename)
        
        try:
            if format == 'json':
                return self._export_json(export_path)
            elif format == 'csv':
                return self._export_csv(export_path)
            elif format == 'pickle':
                return self._export_pickle(export_path)
            elif format == 'excel':
                return self._export_excel(export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None
    
    def _export_json(self, base_path):
        """Export data as JSON"""
        export_data = {
            'session_summary': self.get_session_summary(),
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'annotations': self.annotations,
            'quality_metrics': self.quality_metrics,
            'session_events': self.session_events,
            'analysis_buffers': {k: list(v) for k, v in self.analysis_buffers.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        filepath = f"{base_path}.json"
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Session data exported to JSON: {filepath}")
        return filepath
    
    def _export_csv(self, base_path):
        """Export data as CSV files"""
        # Export raw data
        if self.raw_data:
            raw_filepath = f"{base_path}_raw_data.csv"
            self._write_csv(raw_filepath, self.raw_data, 'raw_data')
        
        # Export processed data
        if self.processed_data:
            processed_filepath = f"{base_path}_processed_data.csv"
            self._write_csv(processed_filepath, self.processed_data, 'processed_data')
        
        # Export annotations
        if self.annotations:
            annotations_filepath = f"{base_path}_annotations.csv"
            self._write_csv(annotations_filepath, self.annotations, 'annotations')
        
        # Export quality metrics
        if self.quality_metrics:
            quality_filepath = f"{base_path}_quality_metrics.csv"
            self._write_csv(quality_filepath, self.quality_metrics, 'quality_metrics')
        
        print(f"📊 Session data exported to CSV files: {base_path}_*.csv")
        return f"{base_path}_*.csv"
    
    def _write_csv(self, filepath, data, data_type):
        """Write data to CSV file"""
        if not data:
            return
        
        with open(filepath, 'w', newline='') as f:
            if data_type == 'raw_data':
                # Get all possible keys from all data entries
                all_keys = set()
                for entry in data:
                    if 'data' in entry and isinstance(entry['data'], dict):
                        all_keys.update(entry['data'].keys())
                
                fieldnames = ['timestamp', 'session_time'] + list(all_keys)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    row = {
                        'timestamp': entry['timestamp'],
                        'session_time': entry['session_time']
                    }
                    if 'data' in entry and isinstance(entry['data'], dict):
                        row.update(entry['data'])
                    writer.writerow(row)
            
            elif data_type == 'processed_data':
                # Get all possible keys from all data entries
                all_keys = set()
                for entry in data:
                    if 'data' in entry and isinstance(entry['data'], dict):
                        all_keys.update(entry['data'].keys())
                
                fieldnames = ['timestamp', 'session_time'] + list(all_keys)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    row = {
                        'timestamp': entry['timestamp'],
                        'session_time': entry['session_time']
                    }
                    if 'data' in entry and isinstance(entry['data'], dict):
                        row.update(entry['data'])
                    writer.writerow(row)
            
            elif data_type == 'annotations':
                fieldnames = ['timestamp', 'session_time', 'text', 'type']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    writer.writerow(entry)
            
            elif data_type == 'quality_metrics':
                fieldnames = ['timestamp', 'session_time', 'metric_name', 'value', 'threshold', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    writer.writerow(entry)
    
    def _export_pickle(self, base_path):
        """Export data as pickle file"""
        export_data = {
            'session_summary': self.get_session_summary(),
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'annotations': self.annotations,
            'quality_metrics': self.quality_metrics,
            'session_events': self.session_events,
            'analysis_buffers': {k: list(v) for k, v in self.analysis_buffers.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        filepath = f"{base_path}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"📊 Session data exported to pickle: {filepath}")
        return filepath
    
    def _export_excel(self, base_path):
        """Export data as Excel file"""
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas not available for Excel export")
            return None
        
        filepath = f"{base_path}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Session summary
            summary = self.get_session_summary()
            if summary:
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Session_Summary', index=False)
            
            # Raw data
            if self.raw_data:
                raw_df = self._data_to_dataframe(self.raw_data, 'raw_data')
                raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Processed data
            if self.processed_data:
                processed_df = self._data_to_dataframe(self.processed_data, 'processed_data')
                processed_df.to_excel(writer, sheet_name='Processed_Data', index=False)
            
            # Annotations
            if self.annotations:
                annotations_df = pd.DataFrame(self.annotations)
                annotations_df.to_excel(writer, sheet_name='Annotations', index=False)
            
            # Quality metrics
            if self.quality_metrics:
                quality_df = pd.DataFrame(self.quality_metrics)
                quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
        
        print(f"📊 Session data exported to Excel: {filepath}")
        return filepath
    
    def _data_to_dataframe(self, data, data_type):
        """Convert data to pandas DataFrame"""
        if data_type in ['raw_data', 'processed_data']:
            rows = []
            for entry in data:
                row = {
                    'timestamp': entry['timestamp'],
                    'session_time': entry['session_time']
                }
                row.update(entry['data'])
                rows.append(row)
            return pd.DataFrame(rows)
        
        return pd.DataFrame(data)
    
    def stop_session(self):
        """Stop the current session"""
        if not self.enable_logging or not self.session_start_time:
            return
        
        # Log session end event
        self.log_session_event("session_end", {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "description": "Research session ended"
        })
        
        # Update final statistics
        self._update_statistics({})
        
        print(f"📊 Research session ended: {self.session_id}")
        print(f"   Duration: {self.statistics['session_duration']:.1f} seconds")
        print(f"   Total frames: {self.statistics['total_frames']}")
        print(f"   Average quality: {self.statistics['average_quality']:.3f}")
    
    def get_available_exports(self):
        """Get list of available export formats"""
        return self.export_formats
    
    def cleanup_old_sessions(self, days_to_keep=30):
        """Clean up old session data"""
        if not self.enable_logging:
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        # Clean up exports directory
        exports_dir = os.path.join(self.data_dir, 'exports')
        if os.path.exists(exports_dir):
            for filename in os.listdir(exports_dir):
                filepath = os.path.join(exports_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    print(f"🗑️ Cleaned up old export: {filename}")
        
        print(f"🧹 Cleanup completed (kept last {days_to_keep} days)") 