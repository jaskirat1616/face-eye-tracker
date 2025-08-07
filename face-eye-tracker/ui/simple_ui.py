import tkinter as tk
from tkinter import ttk
import threading
import time

class SimpleEyeTrackerUI:
    def __init__(self, tracker, data_logger):
        self.tracker = tracker
        self.data_logger = data_logger
        self.is_running = False
        self.root = None
        
    def create_ui(self):
        """Create the main UI window"""
        self.root = tk.Tk()
        self.root.title("Eye Tracking & Fatigue Detection")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
    def create_sidebar(self):
        """Create the sidebar with controls and metrics"""
        sidebar = tk.Frame(self.root, bg='#2d2d2d', width=300)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        sidebar.grid_propagate(False)
        
        # Title
        title_label = tk.Label(sidebar, text="👁️ Eye Tracker", 
                              font=("Arial", 16, "bold"), 
                              bg='#2d2d2d', fg='white')
        title_label.pack(pady=20)
        
        # Control buttons
        control_frame = tk.Frame(sidebar, bg='#2d2d2d')
        control_frame.pack(pady=20, fill='x', padx=10)
        
        self.start_btn = tk.Button(control_frame, text="Start Tracking", 
                                  command=self.start_tracking,
                                  bg='#4CAF50', fg='white', 
                                  font=("Arial", 12, "bold"),
                                  relief='flat', padx=20, pady=10)
        self.start_btn.pack(fill='x', pady=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Tracking", 
                                 command=self.stop_tracking,
                                 bg='#f44336', fg='white', 
                                 font=("Arial", 12, "bold"),
                                 relief='flat', padx=20, pady=10,
                                 state='disabled')
        self.stop_btn.pack(fill='x', pady=5)
        
        # Status indicator
        self.status_label = tk.Label(sidebar, text="Status: Ready", 
                                    font=("Arial", 12), 
                                    bg='#2d2d2d', fg='#4CAF50')
        self.status_label.pack(pady=10)
        
        # Metrics section
        metrics_frame = tk.Frame(sidebar, bg='#2d2d2d')
        metrics_frame.pack(pady=20, fill='x', padx=10)
        
        tk.Label(metrics_frame, text="Real-time Metrics", 
                font=("Arial", 14, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=10)
        
        # Create metric displays
        self.create_metric_display(metrics_frame, "Left Eye Openness", "left_eye_label")
        self.create_metric_display(metrics_frame, "Right Eye Openness", "right_eye_label")
        self.create_metric_display(metrics_frame, "Blink Rate (/min)", "blink_rate_label")
        self.create_metric_display(metrics_frame, "Saccade Rate (/min)", "saccade_rate_label")
        self.create_metric_display(metrics_frame, "Quality Score", "quality_label")
        self.create_metric_display(metrics_frame, "Fatigue Level", "fatigue_label")
        
        # Fatigue scores section
        fatigue_frame = tk.Frame(sidebar, bg='#2d2d2d')
        fatigue_frame.pack(pady=20, fill='x', padx=10)
        
        tk.Label(fatigue_frame, text="Fatigue Analysis", 
                font=("Arial", 14, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=10)
        
        self.create_fatigue_progress(fatigue_frame, "Blink Rate", "blink_rate_progress")
        self.create_fatigue_progress(fatigue_frame, "Blink Duration", "blink_duration_progress")
        self.create_fatigue_progress(fatigue_frame, "Eye Openness", "eye_openness_progress")
        self.create_fatigue_progress(fatigue_frame, "Yawn Detection", "yawn_progress")
        self.create_fatigue_progress(fatigue_frame, "Gaze Drift", "gaze_drift_progress")
        self.create_fatigue_progress(fatigue_frame, "Head Slouch", "head_slouch_progress")
        
    def create_metric_display(self, parent, label_text, attr_name):
        """Create a metric display with label and value"""
        frame = tk.Frame(parent, bg='#2d2d2d')
        frame.pack(fill='x', pady=5)
        
        tk.Label(frame, text=label_text, 
                font=("Arial", 10), 
                bg='#2d2d2d', fg='#cccccc').pack(anchor='w')
        
        label = tk.Label(frame, text="0.000", 
                        font=("Arial", 12, "bold"), 
                        bg='#2d2d2d', fg='#4CAF50')
        label.pack(anchor='w')
        
        setattr(self, attr_name, label)
        
    def create_fatigue_progress(self, parent, label_text, attr_name):
        """Create a fatigue progress bar"""
        frame = tk.Frame(parent, bg='#2d2d2d')
        frame.pack(fill='x', pady=5)
        
        tk.Label(frame, text=label_text, 
                font=("Arial", 10), 
                bg='#2d2d2d', fg='#cccccc').pack(anchor='w')
        
        progress = ttk.Progressbar(frame, length=200, mode='determinate')
        progress.pack(fill='x', pady=2)
        
        value_label = tk.Label(frame, text="0.00", 
                              font=("Arial", 9), 
                              bg='#2d2d2d', fg='#cccccc')
        value_label.pack(anchor='w')
        
        setattr(self, attr_name, (progress, value_label))
        
    def create_main_content(self):
        """Create the main content area"""
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Info panel
        info_frame = tk.Frame(main_frame, bg='#2d2d2d', height=400)
        info_frame.pack(fill='both', expand=True, pady=10)
        info_frame.pack_propagate(False)
        
        info_text = """
        Eye Tracking & Fatigue Detection
        
        This application provides real-time monitoring of:
        
        • Eye openness and blink detection
        • Saccade and microsaccade analysis
        • Fatigue scoring based on multiple indicators
        • Real-time data logging to CSV files
        
        Click 'Start Tracking' to begin monitoring.
        
        Data is automatically saved to CSV files in the data directory.
        """
        
        info_label = tk.Label(info_frame, text=info_text, 
                             font=("Arial", 12), 
                             bg='#2d2d2d', fg='white',
                             justify='left', anchor='nw')
        info_label.pack(expand=True, fill='both', padx=20, pady=20)
        
    def start_tracking(self):
        """Start the eye tracking"""
        if self.is_running:
            return
            
        # Start the tracker
        if not self.tracker.start_camera():
            self.status_label.config(text="Status: Camera Error", fg='#f44336')
            return
            
        # Start data logging
        self.data_logger.start_logging()
        
        # Update UI state
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Status: Tracking Active", fg='#4CAF50')
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_tracking(self):
        """Stop the eye tracking"""
        if not self.is_running:
            return
            
        # Stop the tracker
        self.tracker.stop_camera()
        self.data_logger.stop_logging()
        
        # Update UI state
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Stopped", fg='#f44336')
        
    def process_frames(self):
        """Process frames in a separate thread"""
        while self.is_running:
            try:
                frame = self.tracker.read_frame()
                if frame is None:
                    continue
                    
                # Process the frame
                result = self.tracker.process_frame(frame)
                if result is None:
                    continue
                    
                # Log the data
                self.data_logger.log_data(result)
                
                # Update UI (in main thread) - use after_idle for better thread safety
                self.root.after_idle(self.update_ui, result)
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.1)
            
    def update_ui(self, data):
        """Update the UI with new data"""
        # Update metric labels
        self.left_eye_label.config(text=f"{data['left_eye_openness']:.3f}")
        self.right_eye_label.config(text=f"{data['right_eye_openness']:.3f}")
        self.blink_rate_label.config(text=f"{data['blink_rate']:.1f}")
        self.saccade_rate_label.config(text=f"{data['saccade_rate']:.1f}")
        self.quality_label.config(text=f"{data['quality_score']:.2f}")
        self.fatigue_label.config(text=data['fatigue_level'])
        
        # Update fatigue progress bars
        self.update_progress_bar(self.blink_rate_progress, data['fatigue_blink_rate_score'])
        self.update_progress_bar(self.blink_duration_progress, data['fatigue_blink_duration_score'])
        self.update_progress_bar(self.eye_openness_progress, data['fatigue_eye_openness_score'])
        self.update_progress_bar(self.yawn_progress, data['fatigue_yawn_score'])
        self.update_progress_bar(self.gaze_drift_progress, data['fatigue_gaze_drift_score'])
        self.update_progress_bar(self.head_slouch_progress, data['fatigue_head_slouch_score'])
        
    def update_progress_bar(self, progress_tuple, value):
        """Update a progress bar and its label"""
        progress_bar, label = progress_tuple
        progress_bar['value'] = value * 100
        label.config(text=f"{value:.2f}")
        
        # Color coding based on value
        if value < 0.3:
            label.config(fg='#4CAF50')  # Green
        elif value < 0.6:
            label.config(fg='#FF9800')  # Orange
        else:
            label.config(fg='#F44336')  # Red
            
    def run(self):
        """Run the UI"""
        self.create_ui()
        self.root.mainloop() 