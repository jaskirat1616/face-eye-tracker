import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from datetime import datetime
import threading
import time
import queue
import cv2
from PIL import Image, ImageTk
import csv

# Set matplotlib style for a modern look
style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

class ComprehensiveEyeTrackerUI:
    def __init__(self, tracker, data_logger):
        self.tracker = tracker
        self.data_logger = data_logger
        self.is_running = False
        self.root = None
        self.fig = None
        self.canvas = None
        self.ani = None
        self.video_label = None
        
        # Data queue for thread-safe communication
        self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent lag
        
        # Chart data with limited history for performance
        self.chart_data = {
            'time': [],
            'left_eye': [],
            'right_eye': [],
            'blink_rate': [],
            'fatigue': [],
            'quality': []
        }
        
        # Performance optimization flags
        self.last_update_time = 0
        self.update_interval = 0.3  # Update UI every 300ms (reduced frequency)
        self.chart_update_interval = 2.0  # Update charts every 2 seconds (much less frequent)
        self.last_chart_update = 0
        self.video_update_interval = 0.3  # Update video every 300ms (reduced frequency)
        self.last_video_update = 0
        self.frame_skip_counter = 0
        self.frame_skip_rate = 6  # Process every 6th frame for video (more aggressive)
        self.ui_update_counter = 0
        self.ui_update_rate = 5  # Update UI every 5th data point
        
        # Session data
        self.session_start_time = None
        self.session_data = {
            'total_blinks': 0,
            'total_saccades': 0,
            'avg_fatigue': 0.0,
            'peak_fatigue': 0.0,
            'attention_score': 0.0
        }
        
    def create_ui(self):
        """Create the main UI window"""
        self.root = tk.Tk()
        self.root.title("Eye Tracking & Fatigue Detection - Comprehensive")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#1e1e1e')
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
        # Create charts
        self.create_charts()
        
        # Create video display
        self.create_video_display()
        
        # Create live data metrics area
        self.create_live_data_area()
        
        # Start data processing
        self.start_data_processing()
        
    def create_sidebar(self):
        """Create the sidebar with controls and metrics"""
        sidebar = tk.Frame(self.root, bg='#2d2d2d', width=350)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        sidebar.grid_propagate(False)
        
        # Title
        title_label = tk.Label(sidebar, text="👁️ Eye Tracker Pro", 
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
        
        # Data logging toggle
        self.logging_var = tk.BooleanVar(value=True)
        self.logging_check = tk.Checkbutton(control_frame, 
                                           text="Enable Data Logging", 
                                           variable=self.logging_var,
                                           bg='#2d2d2d', fg='white',
                                           selectcolor='#2d2d2d',
                                           font=("Arial", 10))
        self.logging_check.pack(fill='x', pady=5)
        
        # Export and alerts
        export_frame = tk.Frame(sidebar, bg='#2d2d2d')
        export_frame.pack(pady=10, fill='x', padx=10)
        
        tk.Label(export_frame, text="Export & Alerts", 
                font=("Arial", 12, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=5)
        
        # Export button
        self.export_btn = tk.Button(export_frame, text="Export Session Data", 
                                   command=self.export_session_data,
                                   bg='#2196F3', fg='white', 
                                   font=("Arial", 10, "bold"),
                                   relief='flat', padx=10, pady=5)
        self.export_btn.pack(fill='x', pady=2)
        
        # Fatigue alert toggle
        self.alert_var = tk.BooleanVar(value=True)
        self.alert_check = tk.Checkbutton(export_frame, 
                                         text="Enable Fatigue Alerts", 
                                         variable=self.alert_var,
                                         bg='#2d2d2d', fg='white',
                                         selectcolor='#2d2d2d',
                                         font=("Arial", 10))
        self.alert_check.pack(fill='x', pady=2)
        
        # Charts toggle
        self.charts_enabled = tk.BooleanVar(value=False)  # Disabled by default for performance
        self.charts_check = tk.Checkbutton(export_frame, 
                                          text="Enable Real-time Charts", 
                                          variable=self.charts_enabled,
                                          bg='#2d2d2d', fg='white',
                                          selectcolor='#2d2d2d',
                                          font=("Arial", 10))
        self.charts_check.pack(fill='x', pady=2)
        
        # Performance settings
        perf_frame = tk.Frame(sidebar, bg='#2d2d2d')
        perf_frame.pack(pady=10, fill='x', padx=10)
        
        tk.Label(perf_frame, text="Performance Settings", 
                font=("Arial", 12, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=5)
        
        # Chart update rate
        tk.Label(perf_frame, text="Chart Update Rate (ms):", 
                font=("Arial", 9), 
                bg='#2d2d2d', fg='#cccccc').pack(anchor='w')
        
        self.chart_rate_var = tk.StringVar(value="500")
        chart_rate_entry = tk.Entry(perf_frame, textvariable=self.chart_rate_var, 
                                   bg='#1e1e1e', fg='white', 
                                   font=("Arial", 9), width=8)
        chart_rate_entry.pack(anchor='w', pady=2)
        
        # Performance mode
        tk.Label(perf_frame, text="Performance Mode:", 
                font=("Arial", 9), 
                bg='#2d2d2d', fg='#cccccc').pack(anchor='w', pady=(10,0))
        
        self.performance_mode_var = tk.StringVar(value="Balanced")
        perf_menu = ttk.Combobox(perf_frame, textvariable=self.performance_mode_var,
                                values=["Ultra Fast", "Fast", "Balanced"],
                                state="readonly", width=8)
        perf_menu.pack(anchor='w', pady=2)
        
        # Video quality
        tk.Label(perf_frame, text="Video Quality:", 
                font=("Arial", 9), 
                bg='#2d2d2d', fg='#cccccc').pack(anchor='w', pady=(10,0))
        
        self.video_quality_var = tk.StringVar(value="Low")  # Default to low for performance
        quality_menu = ttk.Combobox(perf_frame, textvariable=self.video_quality_var,
                                   values=["Low", "Medium", "High"],
                                   state="readonly", width=8)
        quality_menu.pack(anchor='w', pady=2)
        
        # Apply performance settings button
        apply_btn = tk.Button(perf_frame, text="Apply Settings", 
                             command=self.apply_performance_settings,
                             bg='#2196F3', fg='white', 
                             font=("Arial", 9, "bold"),
                             relief='flat', padx=10, pady=3)
        apply_btn.pack(fill='x', pady=5)
        
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
        self.create_metric_display(metrics_frame, "Attention Score", "attention_label")
        self.create_metric_display(metrics_frame, "Session Time", "session_time_label")
        
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
        
        # Session summary
        session_frame = tk.Frame(sidebar, bg='#2d2d2d')
        session_frame.pack(pady=20, fill='x', padx=10)
        
        tk.Label(session_frame, text="Session Summary", 
                font=("Arial", 14, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=10)
        
        self.create_metric_display(session_frame, "Total Blinks", "total_blinks_label")
        self.create_metric_display(session_frame, "Total Saccades", "total_saccades_label")
        self.create_metric_display(session_frame, "Avg Fatigue", "avg_fatigue_label")
        self.create_metric_display(session_frame, "Peak Fatigue", "peak_fatigue_label")
        
        # Performance monitoring
        perf_monitor_frame = tk.Frame(sidebar, bg='#2d2d2d')
        perf_monitor_frame.pack(pady=10, fill='x', padx=10)
        
        tk.Label(perf_monitor_frame, text="Performance Monitor", 
                font=("Arial", 12, "bold"), 
                bg='#2d2d2d', fg='white').pack(pady=5)
        
        self.create_metric_display(perf_monitor_frame, "FPS", "fps_label")
        self.create_metric_display(perf_monitor_frame, "Queue Size", "queue_size_label")
        self.create_metric_display(perf_monitor_frame, "Frame Skip", "frame_skip_label")
        
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
        
        # Charts title
        charts_title = tk.Label(main_frame, text="Real-time Charts", 
                               font=("Arial", 16, "bold"), 
                               bg='#1e1e1e', fg='white')
        charts_title.pack(pady=10)
        
    def create_charts(self):
        """Create the charts area"""
        charts_frame = tk.Frame(self.root, bg='#1e1e1e')
        charts_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 6), facecolor='#2b2b2b')
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Eye openness
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Blink rate
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Fatigue score
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Quality score
        
        # Configure subplots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2b2b2b')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
            
        # Set titles
        self.ax1.set_title('Eye Openness Over Time', color='white', fontsize=12)
        self.ax2.set_title('Blink Rate Over Time', color='white', fontsize=12)
        self.ax3.set_title('Fatigue Score Over Time', color='white', fontsize=12)
        self.ax4.set_title('Detection Quality Over Time', color='white', fontsize=12)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_video_display(self):
        """Create the video display area"""
        video_frame = tk.Frame(self.root, bg='#1e1e1e')
        video_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Title
        video_title = tk.Label(video_frame, text="Live Camera Feed", 
                              font=("Arial", 16, "bold"), 
                              bg='#1e1e1e', fg='white')
        video_title.pack(pady=10)
        
        # Create video display label
        self.video_label = tk.Label(video_frame, 
                                   bg='#2d2d2d', 
                                   text="Camera feed will appear here",
                                   font=("Arial", 12),
                                   fg='white')
        self.video_label.pack(expand=True, fill='both', padx=5, pady=5)
        
    def create_live_data_area(self):
        """Create the live data metrics display area"""
        data_frame = tk.Frame(self.root, bg='#1e1e1e')
        data_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Title
        data_title = tk.Label(data_frame, text="Live Data Metrics", 
                             font=("Arial", 16, "bold"), 
                             bg='#1e1e1e', fg='white')
        data_title.pack(pady=10)
        
        # Create main data display frame
        self.data_display_frame = tk.Frame(data_frame, bg='#2d2d2d')
        self.data_display_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create data labels
        self.create_data_labels()
        
    def create_data_labels(self):
        """Create all the data display labels"""
        # Eye tracking metrics
        self.create_metric_section("Eye Tracking", [
            ("Left Eye Openness", "left_eye_data"),
            ("Right Eye Openness", "right_eye_data"),
            ("Blink Rate (/min)", "blink_rate_data"),
            ("Blink Duration (s)", "blink_duration_data"),
            ("Saccade Rate (/min)", "saccade_rate_data"),
            ("Saccade Amplitude", "saccade_amplitude_data"),
            ("Saccade Velocity", "saccade_velocity_data"),
            ("Microsaccade Rate (/min)", "microsaccade_rate_data"),
            ("Quality Score", "quality_data"),
            ("Calibration Status", "calibration_data")
        ])
        
        # Fatigue analysis
        self.create_metric_section("Fatigue Analysis", [
            ("Blink Rate Fatigue", "fatigue_blink_rate_data"),
            ("Blink Duration Fatigue", "fatigue_blink_duration_data"),
            ("Eye Openness Fatigue", "fatigue_eye_openness_data"),
            ("Yawn Detection", "fatigue_yawn_data"),
            ("Gaze Drift Fatigue", "fatigue_gaze_drift_data"),
            ("Head Slouch Fatigue", "fatigue_head_slouch_data"),
            ("Overall Fatigue Score", "overall_fatigue_data"),
            ("Fatigue Level", "fatigue_level_data")
        ])
        
        # Head pose data
        self.create_metric_section("Head Pose Data", [
            ("Head Tilt", "head_tilt_data"),
            ("Forward Tilt", "forward_tilt_data"),
            ("Head Roll", "head_roll_data"),
            ("Forward Position", "forward_position_data"),
            ("Tilt Change", "tilt_change_data"),
            ("Forward Change", "forward_change_data"),
            ("Roll Change", "roll_change_data"),
            ("Slouch Change", "slouch_change_data")
        ])
        
    def create_metric_section(self, title, metrics):
        """Create a section of metrics"""
        section_frame = tk.Frame(self.data_display_frame, bg='#2d2d2d')
        section_frame.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        
        # Section title
        title_label = tk.Label(section_frame, text=title, 
                              font=("Arial", 12, "bold"), 
                              bg='#2d2d2d', fg='#4CAF50')
        title_label.pack(pady=5)
        
        # Create metrics
        for label_text, attr_name in metrics:
            frame = tk.Frame(section_frame, bg='#2d2d2d')
            frame.pack(fill='x', pady=2)
            
            label = tk.Label(frame, text=label_text, 
                           font=("Arial", 9), 
                           bg='#2d2d2d', fg='#cccccc')
            label.pack(anchor='w')
            
            value_label = tk.Label(frame, text="0.000", 
                                 font=("Arial", 10, "bold"), 
                                 bg='#2d2d2d', fg='#4CAF50')
            value_label.pack(anchor='w')
            
            setattr(self, attr_name, value_label)
        
    def start_data_processing(self):
        """Start the data processing loop"""
        self.root.after(100, self.process_data_queue)
        
    def process_data_queue(self):
        """Process data from the queue with aggressive performance optimization"""
        current_time = time.time()
        
        # Only update if enough time has passed
        if current_time - self.last_update_time < self.update_interval:
            self.root.after(50, self.process_data_queue)  # Increased delay
            return
            
        try:
            # Process all available data but limit updates
            data = None
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
            
            if data is not None:
                # Only update UI every few data points
                self.ui_update_counter += 1
                if self.ui_update_counter % self.ui_update_rate == 0:
                    self.update_ui(data)
                    self.update_live_data(data)
                
                # Update video much less frequently
                if current_time - self.last_video_update >= self.video_update_interval:
                    self.update_video(data)
                    self.last_video_update = current_time
                
                # Update charts very infrequently (only if enabled)
                if (self.charts_enabled.get() and 
                    current_time - self.last_chart_update >= self.chart_update_interval):
                    self.update_charts_manual()
                    self.last_chart_update = current_time
                    
        except queue.Empty:
            pass
        
        self.last_update_time = current_time
        # Schedule next processing with longer delay
        self.root.after(50, self.process_data_queue)
        
    def start_animation(self):
        """Start the chart animation with configurable interval"""
        try:
            interval = int(self.chart_rate_var.get())
        except ValueError:
            interval = 2000  # Much longer default interval
            
        # Only start animation if charts are enabled
        if hasattr(self, 'charts_enabled') and self.charts_enabled:
            self.ani = animation.FuncAnimation(
                self.fig, 
                self.update_charts, 
                interval=interval,  # Configurable update rate
                blit=False, 
                cache_frame_data=False
            )
        
    def update_charts_manual(self):
        """Manual chart update for performance optimization"""
        try:
            if not self.is_running:
                return
                
            # Get latest data
            chart_data = self.data_logger.get_chart_data()
            
            # Limit data points for performance
            max_points = 100
            if len(chart_data['time']) > max_points:
                for key in chart_data:
                    chart_data[key] = chart_data[key][-max_points:]
            
            # Update charts
            self.update_charts(None)
            
        except Exception as e:
            print(f"Error in manual chart update: {e}")
        
    def update_charts(self, frame):
        """Update the charts with new data"""
        try:
            if not self.is_running:
                return
                
            # Get latest data
            chart_data = self.data_logger.get_chart_data()
            
            # Limit data points for performance
            max_points = 100
            if len(chart_data['time']) > max_points:
                for key in chart_data:
                    chart_data[key] = chart_data[key][-max_points:]
            
            # Update eye openness chart
            self.ax1.clear()
            self.ax1.set_facecolor('#2b2b2b')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.tick_params(colors='white')
            self.ax1.set_title('Eye Openness Over Time', color='white', fontsize=12)
            
            if len(chart_data['time']) > 1:
                times = [t - chart_data['time'][0] for t in chart_data['time']]
                self.ax1.plot(times, chart_data['left_eye'], label='Left Eye', color='#4CAF50', linewidth=2)
                self.ax1.plot(times, chart_data['right_eye'], label='Right Eye', color='#2196F3', linewidth=2)
                self.ax1.legend()
                
            # Update blink rate chart
            self.ax2.clear()
            self.ax2.set_facecolor('#2b2b2b')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(colors='white')
            self.ax2.set_title('Blink Rate Over Time', color='white', fontsize=12)
            
            if len(chart_data['time']) > 1:
                times = [t - chart_data['time'][0] for t in chart_data['time']]
                self.ax2.plot(times, chart_data['blink_rate'], color='#FF9800', linewidth=2)
                
            # Update fatigue chart
            self.ax3.clear()
            self.ax3.set_facecolor('#2b2b2b')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.tick_params(colors='white')
            self.ax3.set_title('Fatigue Score Over Time', color='white', fontsize=12)
            
            if len(chart_data['time']) > 1:
                times = [t - chart_data['time'][0] for t in chart_data['time']]
                self.ax3.plot(times, chart_data['fatigue'], color='#F44336', linewidth=2)
                
            # Update quality chart
            self.ax4.clear()
            self.ax4.set_facecolor('#2b2b2b')
            self.ax4.grid(True, alpha=0.3)
            self.ax4.tick_params(colors='white')
            self.ax4.set_title('Detection Quality Over Time', color='white', fontsize=12)
            
            if len(chart_data['time']) > 1:
                times = [t - chart_data['time'][0] for t in chart_data['time']]
                self.ax4.plot(times, chart_data['quality'], color='#9C27B0', linewidth=2)
        except Exception as e:
            print(f"Error updating charts: {e}")
            
    def update_video(self, data):
        """Update the video display with optimized performance"""
        try:
            if 'frame' not in data or data['frame'] is None:
                return
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
            
            # Flip horizontally to fix mirroring
            frame_rgb = cv2.flip(frame_rgb, 1)
            
            # Much more aggressive quality settings for performance
            quality = self.video_quality_var.get()
            if quality == "Low":
                max_size = 240  # Much smaller for better performance
                interpolation = cv2.INTER_NEAREST
            elif quality == "High":
                max_size = 320  # Much reduced for performance
                interpolation = cv2.INTER_NEAREST  # Fastest interpolation
            else:  # Medium
                max_size = 280  # Much reduced for performance
                interpolation = cv2.INTER_NEAREST
            
            # Resize frame to fit display
            height, width = frame_rgb.shape[:2]
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=interpolation)
            
            # Convert to PIL Image with optimization
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating video: {e}")
            
    def update_live_data(self, data):
        """Update the live data metrics display"""
        try:
            # Update eye tracking metrics
            self.left_eye_data.config(text=f"{data['left_eye_openness']:.3f}")
            self.right_eye_data.config(text=f"{data['right_eye_openness']:.3f}")
            self.blink_rate_data.config(text=f"{data['blink_rate']:.1f}")
            self.blink_duration_data.config(text=f"{data['blink_duration']:.3f}")
            self.saccade_rate_data.config(text=f"{data['saccade_rate']:.1f}")
            self.saccade_amplitude_data.config(text=f"{data['saccade_amplitude']:.3f}")
            self.saccade_velocity_data.config(text=f"{data['saccade_velocity']:.3f}")
            self.microsaccade_rate_data.config(text=f"{data['microsaccade_rate']:.1f}")
            self.quality_data.config(text=f"{data['quality_score']:.2f}")
            
            # Update calibration status
            calib_status = "Calibrating..." if not data.get('calibration_complete', False) else "Ready"
            self.calibration_data.config(text=calib_status)
            
            # Update fatigue analysis
            self.fatigue_blink_rate_data.config(text=f"{data['fatigue_blink_rate_score']:.2f}")
            self.fatigue_blink_duration_data.config(text=f"{data['fatigue_blink_duration_score']:.2f}")
            self.fatigue_eye_openness_data.config(text=f"{data['fatigue_eye_openness_score']:.2f}")
            
            # Yawn detection with special handling
            yawn_text = "DETECTED!" if data.get('yawn_detected', False) else f"{data['fatigue_yawn_score']:.2f}"
            self.fatigue_yawn_data.config(text=yawn_text)
            
            self.fatigue_gaze_drift_data.config(text=f"{data['fatigue_gaze_drift_score']:.2f}")
            self.fatigue_head_slouch_data.config(text=f"{data['fatigue_head_slouch_score']:.2f}")
            self.overall_fatigue_data.config(text=f"{data['overall_fatigue_score']:.2f}")
            self.fatigue_level_data.config(text=data['fatigue_level'])
            
            # Update head pose data (if available)
            if hasattr(self.tracker, 'head_poses') and len(self.tracker.head_poses) > 0:
                current_pose = list(self.tracker.head_poses)[-1]
                self.head_tilt_data.config(text=f"{current_pose[0]:.3f}")
                self.forward_tilt_data.config(text=f"{current_pose[1]:.3f}")
                self.head_roll_data.config(text=f"{current_pose[2]:.3f}")
                self.forward_position_data.config(text=f"{current_pose[3]:.3f}")
                
                # Calculate changes from baseline
                if hasattr(self.tracker, 'baseline_head_pose') and self.tracker.baseline_head_pose is not None:
                    pose_change = current_pose - self.tracker.baseline_head_pose
                    self.tilt_change_data.config(text=f"{pose_change[0]:.3f}")
                    self.forward_change_data.config(text=f"{pose_change[1]:.3f}")
                    self.roll_change_data.config(text=f"{pose_change[2]:.3f}")
                    self.slouch_change_data.config(text=f"{pose_change[3]:.3f}")
                else:
                    self.tilt_change_data.config(text="Calibrating...")
                    self.forward_change_data.config(text="Calibrating...")
                    self.roll_change_data.config(text="Calibrating...")
                    self.slouch_change_data.config(text="Calibrating...")
            else:
                self.head_tilt_data.config(text="Calibrating...")
                self.forward_tilt_data.config(text="Calibrating...")
                self.head_roll_data.config(text="Calibrating...")
                self.forward_position_data.config(text="Calibrating...")
                self.tilt_change_data.config(text="Calibrating...")
                self.forward_change_data.config(text="Calibrating...")
                self.roll_change_data.config(text="Calibrating...")
                self.slouch_change_data.config(text="Calibrating...")
                
        except Exception as e:
            print(f"Error updating live data: {e}")
            
    def start_tracking(self):
        """Start the eye tracking"""
        if self.is_running:
            return
            
        # Start the tracker
        if not self.tracker.start_camera():
            self.status_label.config(text="Status: Camera Error", fg='#f44336')
            return
            
        # Start data logging if enabled
        if self.logging_var.get():
            self.data_logger.start_logging()
        
        # Initialize session data
        self.session_start_time = time.time()
        self.session_data = {
            'total_blinks': 0,
            'total_saccades': 0,
            'avg_fatigue': 0.0,
            'peak_fatigue': 0.0,
            'attention_score': 0.0
        }
        
        # Update UI state
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Status: Tracking Active", fg='#4CAF50')
        
        # Start chart animation
        self.start_animation()
        
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
        """Process frames in a separate thread with frame skipping for performance"""
        while self.is_running:
            try:
                frame = self.tracker.read_frame()
                if frame is None:
                    continue
                
                # Frame skipping for video processing
                self.frame_skip_counter += 1
                process_video = (self.frame_skip_counter % self.frame_skip_rate == 0)
                
                # Process the frame
                result = self.tracker.process_frame(frame)
                if result is None:
                    continue
                
                # Only include frame data if we're processing video
                if not process_video:
                    result.pop('frame', None)
                
                # Log the data if logging is enabled
                if self.logging_var.get():
                    self.data_logger.log_data(result)
                
                # Add to queue for UI update
                try:
                    self.data_queue.put_nowait(result)
                except queue.Full:
                    # If queue is full, remove oldest item and add new one
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                # Much more aggressive frame rate limiting
                if hasattr(self, 'last_frame_time'):
                    elapsed = time.time() - self.last_frame_time
                    target_fps = 10  # Much reduced for better performance
                    target_interval = 1.0 / target_fps
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
                self.last_frame_time = time.time()
                
                # Check for fatigue alerts
                if self.alert_var.get() and result['overall_fatigue_score'] > 0.7:
                    self.show_fatigue_alert(result['overall_fatigue_score'])
                    
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
        
        # Calculate attention score (inverse of fatigue)
        attention_score = max(0, 1.0 - data['overall_fatigue_score'])
        self.attention_label.config(text=f"{attention_score:.2f}")
        
        # Update session time
        if self.session_start_time:
            session_time = time.time() - self.session_start_time
            minutes = int(session_time // 60)
            seconds = int(session_time % 60)
            self.session_time_label.config(text=f"{minutes:02d}:{seconds:02d}")
        
        # Update session data
        if data.get('blink_detected', False):
            self.session_data['total_blinks'] += 1
        if data.get('saccade_detected', False):
            self.session_data['total_saccades'] += 1
            
        self.session_data['peak_fatigue'] = max(self.session_data['peak_fatigue'], data['overall_fatigue_score'])
        
        # Update session summary
        self.total_blinks_label.config(text=str(self.session_data['total_blinks']))
        self.total_saccades_label.config(text=str(self.session_data['total_saccades']))
        self.peak_fatigue_label.config(text=f"{self.session_data['peak_fatigue']:.2f}")
        
        # Update performance metrics
        if hasattr(self, 'last_frame_time'):
            current_time = time.time()
            if hasattr(self, 'fps_start_time'):
                fps_elapsed = current_time - self.fps_start_time
                if fps_elapsed > 0:
                    fps = self.frame_skip_counter / fps_elapsed
                    self.fps_label.config(text=f"{fps:.1f}")
            else:
                self.fps_start_time = current_time
                self.fps_label.config(text="0.0")
        
        self.queue_size_label.config(text=str(self.data_queue.qsize()))
        self.frame_skip_label.config(text=f"1/{self.frame_skip_rate}")
        
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
            
    def export_session_data(self):
        """Export session data to CSV"""
        try:
            if not self.session_start_time:
                return
                
            # Calculate session duration
            session_duration = time.time() - self.session_start_time
            
            # Create export data
            export_data = {
                'session_duration_minutes': session_duration / 60,
                'total_blinks': self.session_data['total_blinks'],
                'total_saccades': self.session_data['total_saccades'],
                'peak_fatigue_score': self.session_data['peak_fatigue'],
                'average_fatigue_score': self.session_data.get('avg_fatigue', 0.0),
                'attention_score': self.session_data.get('attention_score', 0.0),
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Export to CSV
            filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in export_data.items():
                    writer.writerow([key, value])
            
            # Show success message
            self.show_info_dialog("Export Successful", f"Session data exported to {filename}")
            
        except Exception as e:
            self.show_info_dialog("Export Error", f"Failed to export data: {str(e)}")
        
    def apply_performance_settings(self):
        """Apply performance settings based on user selection"""
        mode = self.performance_mode_var.get()
        
        if mode == "Ultra Fast":
            self.update_interval = 0.5
            self.video_update_interval = 0.5
            self.chart_update_interval = 5.0
            self.frame_skip_rate = 8
            self.ui_update_rate = 10
            self.video_quality_var.set("Low")
        elif mode == "Fast":
            self.update_interval = 0.3
            self.video_update_interval = 0.3
            self.chart_update_interval = 2.0
            self.frame_skip_rate = 6
            self.ui_update_rate = 5
            self.video_quality_var.set("Low")
        else:  # Balanced
            self.update_interval = 0.2
            self.video_update_interval = 0.2
            self.chart_update_interval = 1.0
            self.frame_skip_rate = 4
            self.ui_update_rate = 3
            self.video_quality_var.set("Medium")
            
        self.show_info_dialog("Settings Applied", f"Performance mode set to: {mode}")
            
    def show_fatigue_alert(self, fatigue_score):
        """Show fatigue alert dialog"""
        if fatigue_score > 0.8:
            level = "SEVERE"
            color = "#F44336"
        elif fatigue_score > 0.7:
            level = "MODERATE"
            color = "#FF9800"
        else:
            level = "MILD"
            color = "#FFC107"
            
        message = f"⚠️ FATIGUE ALERT ⚠️\n\nFatigue Level: {level}\nScore: {fatigue_score:.2f}\n\nConsider taking a break!"
        self.show_alert_dialog("Fatigue Alert", message, color)
        
    def show_alert_dialog(self, title, message, color="#FF9800"):
        """Show alert dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x200")
        dialog.configure(bg='#1e1e1e')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Message
        msg_label = tk.Label(dialog, text=message, 
                           font=("Arial", 12), 
                           bg='#1e1e1e', fg=color,
                           wraplength=350, justify='center')
        msg_label.pack(expand=True, fill='both', padx=20, pady=20)
        
        # OK button
        ok_btn = tk.Button(dialog, text="OK", 
                          command=dialog.destroy,
                          bg=color, fg='white',
                          font=("Arial", 12, "bold"),
                          relief='flat', padx=20, pady=10)
        ok_btn.pack(pady=10)
        
    def show_info_dialog(self, title, message):
        """Show info dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.configure(bg='#1e1e1e')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Message
        msg_label = tk.Label(dialog, text=message, 
                           font=("Arial", 11), 
                           bg='#1e1e1e', fg='white',
                           wraplength=350, justify='center')
        msg_label.pack(expand=True, fill='both', padx=20, pady=20)
        
        # OK button
        ok_btn = tk.Button(dialog, text="OK", 
                          command=dialog.destroy,
                          bg='#4CAF50', fg='white',
                          font=("Arial", 12, "bold"),
                          relief='flat', padx=20, pady=10)
        ok_btn.pack(pady=10)
            
    def run(self):
        """Run the UI"""
        self.create_ui()
        self.root.mainloop() 