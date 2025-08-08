#!/usr/bin/env python3
"""
Advanced Research UI for Eye Tracking & Cognitive Load Detection
================================================================

Professional research interface with advanced features:
- Real-time research metrics display
- Advanced calibration interface
- Research data visualization
- Export capabilities
- Session management
- Quality monitoring
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import threading
import time
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import style
import json
import os

# Set matplotlib style for professional research look
style.use('default')
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['text.color'] = '#212529'
plt.rcParams['axes.labelcolor'] = '#6c757d'
plt.rcParams['xtick.color'] = '#6c757d'
plt.rcParams['ytick.color'] = '#6c757d'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#e9ecef'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'

class ResearchEyeTrackerUI:
    """
    Advanced research-grade UI for eye tracking and cognitive load detection
    """
    
    def __init__(self, tracker, data_logger):
        self.tracker = tracker
        self.data_logger = data_logger
        self.is_running = False
        self.root = None
        
        # Research session management
        self.session_start_time = None
        self.session_data = []
        self.calibration_mode = False
        self.current_calibration_point = 0
        
        # Performance optimization
        self.frame_queue = queue.Queue(maxsize=1)
        self.ui_update_interval = 16  # 60 FPS
        self.chart_update_interval = 200  # 5 FPS for charts
        self.last_ui_update = 0
        self.last_chart_update = 0
        
        # Chart management
        self.chart_lines = {}
        self.chart_initialized = False
        self.fig = None
        self.canvas = None
        self.ani = None
        
        # UI elements
        self.video_label = None
        self.metrics_frame = None
        self.status_label = None
        self.control_frame = None
        
        # Research metrics
        self.research_metrics = {}
        self.quality_indicators = {}
        
        # Chart data
        self.chart_data = {
            'time': [],
            'left_eye_openness': [],
            'right_eye_openness': [],
            'fatigue': [],
            'quality': [],
            'pupil_diameter': [],
            'gaze_stability': [],
            'eye_velocity': [],
            'cognitive_load': [],
            'attention_span': [],
            'processing_speed': [],
            'mental_effort': [],
            'blink_rate': [],
            'saccade_rate': [],
            'fixation_duration': [],
            'head_tilt': [],
            'head_yaw': [],
            'head_roll': []
        }
        
    def create_ui(self):
        """Create the advanced research UI"""
        self.root = tk.Tk()
        self.root.title("Advanced Eye Tracking & Cognitive Load Research System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#ffffff')
        
        # Set window properties
        self.root.resizable(True, True)
        self.root.minsize(1400, 900)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=5)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_columnconfigure(2, weight=1)
        
        # Create UI components
        self.create_research_sidebar()
        self.create_main_content()
        self.create_research_metrics()
        self.create_advanced_charts()
        self.create_calibration_interface()
        
    def create_research_sidebar(self):
        """Create a scrollable research sidebar with advanced controls"""
        sidebar_container = tk.Frame(self.root, bg='#f8f9fa', width=350)
        sidebar_container.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=2, pady=2)
        sidebar_container.grid_propagate(False)
        
        canvas = tk.Canvas(sidebar_container, bg='#f8f9fa', highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f8f9fa')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # All sidebar content is now packed into the scrollable_frame
        
        # Research title
        title_frame = tk.Frame(scrollable_frame, bg='#f8f9fa')
        title_frame.pack(fill='x', pady=(20, 25), padx=10)
        
        title_label = tk.Label(title_frame, text="Research Eye Tracking System", 
                              font=("SF Pro Display", 16, "bold"), 
                              bg='#f8f9fa', fg='#212529')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Cognitive Load & Fatigue Detection", 
                                 font=("SF Pro Text", 12), 
                                 bg='#f8f9fa', fg='#6c757d')
        subtitle_label.pack()
        
        # Session controls
        self.create_session_controls(scrollable_frame)
        
        # Calibration controls
        self.create_calibration_controls(scrollable_frame)
        
        # Research controls
        self.create_research_controls(scrollable_frame)
        
        # Status and quality indicators
        self.create_status_indicators(scrollable_frame)
        
        # Data validation panel
        self.create_data_validation_panel(scrollable_frame)
    
    def create_session_controls(self, parent):
        """Create session management controls"""
        session_frame = tk.LabelFrame(parent, text="Session Management", 
                                     font=("SF Pro Text", 12, "bold"),
                                     bg='#f8f9fa', fg='#212529', bd=1)
        session_frame.pack(fill='x', padx=15, pady=10)
        
        # Session buttons
        button_style = {
            'font': ("SF Pro Text", 10, "bold"),
            'relief': 'flat',
            'padx': 15,
            'pady': 8,
            'cursor': 'hand2',
            'borderwidth': 0
        }
        
        self.start_btn = tk.Button(session_frame, text="Start Research Session", 
                                  command=self.start_research_session,
                                  bg='#28a745', fg='white', 
                                  activebackground='#218838',
                                  **button_style)
        self.start_btn.pack(fill='x', padx=10, pady=5)
        
        self.stop_btn = tk.Button(session_frame, text="Stop Session", 
                                 command=self.stop_research_session,
                                 bg='#dc3545', fg='white', 
                                 activebackground='#c82333',
                                 state='disabled',
                                 **button_style)
        self.stop_btn.pack(fill='x', padx=10, pady=5)
        
        self.export_btn = tk.Button(session_frame, text="Export Research Data", 
                                   command=self.export_research_data,
                                   bg='#17a2b8', fg='white', 
                                   activebackground='#138496',
                                   **button_style)
        self.export_btn.pack(fill='x', padx=10, pady=5)
        
        # Session info
        self.session_info_frame = tk.Frame(session_frame, bg='#f8f9fa')
        self.session_info_frame.pack(fill='x', padx=10, pady=5)
        
        self.session_duration_label = tk.Label(self.session_info_frame, text="Duration: 00:00:00",
                                              font=("SF Pro Text", 10), 
                                              bg='#f8f9fa', fg='#6c757d')
        self.session_duration_label.pack(anchor='w')
        
        self.session_quality_label = tk.Label(self.session_info_frame, text="Quality: --",
                                             font=("SF Pro Text", 10), 
                                             bg='#f8f9fa', fg='#6c757d')
        self.session_quality_label.pack(anchor='w')
        
    def create_calibration_controls(self, parent):
        """Create calibration controls"""
        calib_frame = tk.LabelFrame(parent, text="Advanced Calibration", 
                                   font=("SF Pro Text", 12, "bold"),
                                   bg='#f8f9fa', fg='#212529', bd=1)
        calib_frame.pack(fill='x', padx=15, pady=10)
        
        button_style = {
            'font': ("SF Pro Text", 10, "bold"),
            'relief': 'flat',
            'padx': 15,
            'pady': 8,
            'cursor': 'hand2',
            'borderwidth': 0
        }
        
        self.calib_start_btn = tk.Button(calib_frame, text="Start Calibration", 
                                        command=self.start_calibration,
                                        bg='#ffc107', fg='#212529', 
                                        activebackground='#e0a800',
                                        **button_style)
        self.calib_start_btn.pack(fill='x', padx=10, pady=5)
        
        self.calib_next_btn = tk.Button(calib_frame, text="Next Calibration Point", 
                                       command=self.next_calibration_point,
                                       bg='#6f42c1', fg='white', 
                                       activebackground='#5a32a3',
                                       state='disabled',
                                       **button_style)
        self.calib_next_btn.pack(fill='x', padx=10, pady=5)
        
        self.calib_finish_btn = tk.Button(calib_frame, text="Finish Calibration", 
                                         command=self.finish_calibration,
                                         bg='#20c997', fg='white', 
                                         activebackground='#1ea085',
                                         state='disabled',
                                         **button_style)
        self.calib_finish_btn.pack(fill='x', padx=10, pady=5)
        
        # Calibration info
        self.calib_info_frame = tk.Frame(calib_frame, bg='#f8f9fa')
        self.calib_info_frame.pack(fill='x', padx=10, pady=5)
        
        self.calib_progress_label = tk.Label(self.calib_info_frame, text="Progress: 0/9 points",
                                            font=("SF Pro Text", 10), 
                                            bg='#f8f9fa', fg='#6c757d')
        self.calib_progress_label.pack(anchor='w')
        
        self.calib_quality_label = tk.Label(self.calib_info_frame, text="Quality: --",
                                           font=("SF Pro Text", 10), 
                                           bg='#f8f9fa', fg='#6c757d')
        self.calib_quality_label.pack(anchor='w')
        
    def create_research_controls(self, parent):
        """Create research-specific controls"""
        research_frame = tk.LabelFrame(parent, text="Research Controls", 
                                      font=("SF Pro Text", 12, "bold"),
                                      bg='#f8f9fa', fg='#212529', bd=1)
        research_frame.pack(fill='x', padx=15, pady=10)
        
        # Research mode toggle
        self.research_mode_var = tk.BooleanVar(value=True)
        research_mode_check = tk.Checkbutton(research_frame, text="Research Mode", 
                                            variable=self.research_mode_var,
                                            font=("SF Pro Text", 10),
                                            bg='#f8f9fa', fg='#212529',
                                            selectcolor='#e9ecef')
        research_mode_check.pack(anchor='w', padx=10, pady=2)
        
        # Quality threshold slider
        quality_frame = tk.Frame(research_frame, bg='#f8f9fa')
        quality_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(quality_frame, text="Quality Threshold:", 
                font=("SF Pro Text", 10), 
                bg='#f8f9fa', fg='#6c757d').pack(anchor='w')
        
        self.quality_threshold = tk.DoubleVar(value=0.7)
        quality_slider = tk.Scale(quality_frame, from_=0.1, to=1.0, 
                                 variable=self.quality_threshold,
                                 orient='horizontal', resolution=0.1,
                                 bg='#f8f9fa', fg='#212529',
                                 highlightthickness=0)
        quality_slider.pack(fill='x', pady=2)
        
        # Annotation controls
        annotation_frame = tk.Frame(research_frame, bg='#f8f9fa')
        annotation_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(annotation_frame, text="Add Annotation:", 
                font=("SF Pro Text", 10), 
                bg='#f8f9fa', fg='#6c757d').pack(anchor='w')
        
        self.annotation_entry = tk.Entry(annotation_frame, font=("SF Pro Text", 10))
        self.annotation_entry.pack(fill='x', pady=2)
        
        self.add_annotation_btn = tk.Button(annotation_frame, text="Add", 
                                           command=self.add_annotation,
                                           bg='#fd7e14', fg='white',
                                           font=("SF Pro Text", 9, "bold"),
                                           relief='flat', padx=10, pady=3)
        self.add_annotation_btn.pack(anchor='w', pady=2)
        
    def create_status_indicators(self, parent):
        """Create status and quality indicators"""
        status_frame = tk.LabelFrame(parent, text="System Status", 
                                    font=("SF Pro Text", 12, "bold"),
                                    bg='#f8f9fa', fg='#212529', bd=1)
        status_frame.pack(fill='x', padx=15, pady=10)
        
        # Status indicators
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    font=("SF Pro Text", 12, "bold"), 
                                    bg='#f8f9fa', fg='#28a745')
        self.status_label.pack(pady=8)
        
        # Data quality indicator
        quality_frame = tk.Frame(status_frame, bg='#f8f9fa')
        quality_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(quality_frame, text="Data Quality:", 
                font=("SF Pro Text", 10, "bold"), 
                bg='#f8f9fa', fg='#6c757d').pack(anchor='w')
        
        self.data_quality_label = tk.Label(quality_frame, text="--", 
                                         font=("SF Pro Text", 10, "bold"), 
                                         bg='#f8f9fa', fg='#6c757d')
        self.data_quality_label.pack(anchor='w')
        
        # Quality indicators
        quality_indicators = [
            ("Tracking Quality", "tracking_quality"),
            ("Calibration Quality", "calibration_quality"),
            ("Face Detection", "face_detection"),
            ("Pupil Tracking", "pupil_tracking"),
            ("Gaze Estimation", "gaze_estimation"),
            ("Data Collection", "data_collection_status")
        ]
        
        self.quality_indicators = {}
        for label_text, key in quality_indicators:
            indicator_frame = tk.Frame(status_frame, bg='#f8f9fa')
            indicator_frame.pack(fill='x', padx=10, pady=2)
            
            tk.Label(indicator_frame, text=f"{label_text}:", 
                    font=("SF Pro Text", 10), 
                    bg='#f8f9fa', fg='#6c757d').pack(side='left')
            
            quality_label = tk.Label(indicator_frame, text="--", 
                                   font=("SF Pro Text", 10, "bold"), 
                                   bg='#f8f9fa', fg='#6c757d')
            quality_label.pack(side='right')
            
            self.quality_indicators[key] = quality_label
        
        # Performance indicators
        perf_frame = tk.Frame(status_frame, bg='#f8f9fa')
        perf_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(perf_frame, text="Performance:", 
                font=("SF Pro Text", 10, "bold"), 
                bg='#f8f9fa', fg='#6c757d').pack(anchor='w')
        
        self.fps_label = tk.Label(perf_frame, text="FPS: 0", 
                                 font=("SF Pro Text", 10), 
                                 bg='#f8f9fa', fg='#6c757d')
        self.fps_label.pack(anchor='w')
        
        self.latency_label = tk.Label(perf_frame, text="Latency: --", 
                                     font=("SF Pro Text", 10), 
                                     bg='#f8f9fa', fg='#6c757d')
        self.latency_label.pack(anchor='w')
        
    def create_data_validation_panel(self, parent):
        """Create a panel for data validation and warnings"""
        validation_frame = tk.LabelFrame(parent, text="Data Validation & Warnings", 
                                        font=("SF Pro Text", 12, "bold"),
                                        bg='#f8f9fa', fg='#212529', bd=1)
        validation_frame.pack(fill='x', padx=15, pady=10)
        
        # Quality threshold warning
        self.quality_warning_label = tk.Label(validation_frame, text="", 
                                             font=("SF Pro Text", 10, "bold"), 
                                             bg='#f8f9fa', fg='#dc3545')
        self.quality_warning_label.pack(pady=5)
        
        # Data collection status warning
        self.data_collection_warning_label = tk.Label(validation_frame, text="", 
                                                      font=("SF Pro Text", 10, "bold"), 
                                                      bg='#f8f9fa', fg='#fd7e14')
        self.data_collection_warning_label.pack(pady=5)
        
        # Calibration quality warning
        self.calib_quality_warning_label = tk.Label(validation_frame, text="", 
                                                     font=("SF Pro Text", 10, "bold"), 
                                                     bg='#f8f9fa', fg='#ffc107')
        self.calib_quality_warning_label.pack(pady=5)
        
        # Face detection warning
        self.face_detection_warning_label = tk.Label(validation_frame, text="", 
                                                      font=("SF Pro Text", 10, "bold"), 
                                                      bg='#f8f9fa', fg='#dc3545')
        self.face_detection_warning_label.pack(pady=5)
        
        # Pupil tracking warning
        self.pupil_tracking_warning_label = tk.Label(validation_frame, text="", 
                                                      font=("SF Pro Text", 10, "bold"), 
                                                      bg='#f8f9fa', fg='#dc3545')
        self.pupil_tracking_warning_label.pack(pady=5)
        
        # Gaze estimation warning
        self.gaze_estimation_warning_label = tk.Label(validation_frame, text="", 
                                                       font=("SF Pro Text", 10, "bold"), 
                                                       bg='#f8f9fa', fg='#dc3545')
        self.gaze_estimation_warning_label.pack(pady=5)
        
    def create_main_content(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg='#ffffff')
        main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Video display
        video_frame = tk.LabelFrame(main_frame, text="Live Research Feed", 
                                   font=("SF Pro Display", 14, "bold"),
                                   bg='#ffffff', fg='#212529', bd=1)
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video container
        video_container = tk.Frame(video_frame, bg='#000000', relief='flat', bd=2)
        video_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.video_label = tk.Label(video_container, bg='#000000', text="Research camera feed",
                                   font=("SF Pro Text", 12), fg='#ffffff')
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Research instructions
        instructions_frame = tk.Frame(main_frame, bg='#f8f9fa', relief='flat', bd=1)
        instructions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        instructions_title = tk.Label(instructions_frame, text="Research Protocol", 
                                     font=("SF Pro Display", 12, "bold"), 
                                     bg='#f8f9fa', fg='#212529')
        instructions_title.pack(pady=(12, 8))
        
        instructions = [
            "1. Complete calibration procedure for accurate tracking",
            "2. Maintain stable head position during research session",
            "3. Monitor real-time research metrics and quality indicators",
            "4. Add annotations for significant events or conditions",
            "5. Export research data for analysis"
        ]
        
        for instruction in instructions:
            label = tk.Label(instructions_frame, text=instruction, 
                           font=("SF Pro Text", 10), 
                           bg='#f8f9fa', fg='#6c757d')
            label.pack(anchor='w', padx=20, pady=2)
        
        tk.Label(instructions_frame, text="", bg='#f8f9fa').pack(pady=12)
        
    def create_research_metrics(self):
        """Create comprehensive research metrics display"""
        metrics_frame = tk.Frame(self.root, bg='#f8f9fa', relief='flat', bd=1)
        metrics_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Title
        metrics_title = tk.Label(metrics_frame, text="Research Metrics", 
                                font=("SF Pro Display", 16, "bold"), 
                                bg='#f8f9fa', fg='#212529')
        metrics_title.pack(pady=12)
        
        # Create scrollable metrics container
        canvas = tk.Canvas(metrics_frame, bg='#f8f9fa', highlightthickness=0)
        scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f8f9fa')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=15)
        scrollbar.pack(side="right", fill="y")
        
        # Research metrics
        research_metrics = [
            ("Advanced Fatigue Score", "advanced_fatigue", "#dc3545"),
            ("Advanced Quality Score", "advanced_quality", "#6f42c1"),
            ("Cognitive Load Score", "cognitive_load", "#fd7e14"),
            ("Pupil Diameter (px)", "pupil_diameter", "#20c997"),
            ("Gaze Stability", "gaze_stability", "#17a2b8"),
            ("Eye Velocity", "eye_velocity", "#ffc107"),
            ("Fixation Duration", "fixation_duration", "#e83e8c"),
            ("Attention Span", "attention_span", "#6f42c1"),
            ("Processing Speed", "processing_speed", "#28a745"),
            ("Mental Effort", "mental_effort", "#dc3545"),
            ("Blink Rate (/min)", "blink_rate", "#17a2b8"),
            ("Saccade Rate (/min)", "saccade_rate", "#ffc107"),
            ("Head Tilt (rad)", "head_tilt", "#6610f2"),
            ("Head Yaw (rad)", "head_yaw", "#6610f2"),
            ("Head Roll (rad)", "head_roll", "#6610f2"),
            ("Calibration Quality", "calibration_quality", "#28a745"),
            ("Session Duration", "session_duration", "#6c757d"),
            ("Total Events", "total_events", "#495057"),
            ("Average Quality", "avg_quality", "#6c757d"),
            ("Research Mode", "research_mode", "#28a745"),
            ("Data Collection", "data_collection", "#17a2b8"),
            ("Export Status", "export_status", "#6c757d")
        ]
        
        self.research_metrics = {}
        for label_text, key, color in research_metrics:
            metric_row = tk.Frame(scrollable_frame, bg='#ffffff', relief='flat', bd=1)
            metric_row.pack(fill='x', pady=2, padx=5)
            
            title_label = tk.Label(metric_row, text=label_text, 
                                 font=("SF Pro Text", 10, "bold"), 
                                 bg='#ffffff', fg='#6c757d', width=20, anchor='w')
            title_label.pack(side='left', padx=12, pady=8)
            
            value_label = tk.Label(metric_row, text="--", 
                                 font=("SF Pro Display", 11, "bold"), 
                                 bg='#ffffff', fg=color, width=15, anchor='w')
            value_label.pack(side='right', padx=12, pady=8)
            
            self.research_metrics[key] = value_label
    
    def create_advanced_charts(self):
        """Create consolidated and advanced research charts"""
        charts_frame = tk.Frame(self.root, bg='#ffffff', relief='flat', bd=1)
        charts_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        charts_title = tk.Label(charts_frame, text="Research Analytics Dashboard", 
                           font=("SF Pro Display", 18, "bold"), 
                           bg='#ffffff', fg='#212529')
        charts_title.pack(pady=10)

        self.fig = Figure(figsize=(12, 6), facecolor='#ffffff', dpi=100)
        gs = GridSpec(3, 2, figure=self.fig)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax5 = self.fig.add_subplot(gs[2, :])

        axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]
        titles = [
            'Eye State (Openness)', 'Event Rates (per second)', 'Cognitive State',
            'Data Quality', 'Advanced Neurometrics'
        ]
    
        for ax, title in zip(axes, titles):
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(colors='#6c757d', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dee2e6')
            ax.spines['bottom'].set_color('#dee2e6')
            ax.set_title(title, color='#212529', fontsize=10, fontweight='bold', pad=6)

        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True, padx=16, pady=(0, 12))
        canvas_widget.configure(height=360)  # Cap charts pane height

        self.start_chart_animation()
    
    def create_calibration_interface(self):
        """Create calibration interface overlay"""
        # This would create a calibration overlay window
        # For now, we'll handle calibration in the main interface
        pass
    
    def start_chart_animation(self):
        """Start chart animation"""
        self.ani = animation.FuncAnimation(self.fig, self.update_charts, 
                                          interval=200, blit=True, cache_frame_data=False)
    
    def update_charts(self, frame):
        """Update research charts"""
        try:
            if not self.is_running:
                return []
            
            current_time = time.time()
            
            # Get latest research data
            data = self.tracker.get_current_data()
            if not data:
                return []
            
            # Update chart data
            self.chart_data['time'].append(current_time)
            self.chart_data['left_eye_openness'].append(float(data.get('left_eye_openness', 0)))
            self.chart_data['right_eye_openness'].append(float(data.get('right_eye_openness', 0)))
            self.chart_data['fatigue'].append(float(data.get('advanced_fatigue_score', 0)))
            self.chart_data['quality'].append(float(data.get('advanced_quality_score', 0)))
            self.chart_data['pupil_diameter'].append(float(data.get('pupil_diameter', 0)))
            self.chart_data['gaze_stability'].append(float(data.get('gaze_stability', 0)))
            self.chart_data['eye_velocity'].append(float(data.get('eye_velocity', 0)))
            self.chart_data['cognitive_load'].append(float(data.get('cognitive_load_score', 0)))
            self.chart_data['attention_span'].append(float(data.get('attention_span', 0)))
            self.chart_data['processing_speed'].append(float(data.get('processing_speed', 0)))
            self.chart_data['mental_effort'].append(float(data.get('mental_effort', 0)))
            self.chart_data['blink_rate'].append(float(data.get('blink_rate', 0)))
            self.chart_data['saccade_rate'].append(float(data.get('saccade_rate', 0)))
            self.chart_data['fixation_duration'].append(float(data.get('fixation_duration', 0)))
            self.chart_data['head_tilt'].append(float(data.get('head_tilt', 0)))
            self.chart_data['head_yaw'].append(float(data.get('head_yaw', 0)))
            self.chart_data['head_roll'].append(float(data.get('head_roll', 0)))
            
            # Keep only last 50 data points
            max_points = 50
            for key in self.chart_data:
                if len(self.chart_data[key]) > max_points:
                    self.chart_data[key] = self.chart_data[key][-max_points:]
            
            # Update research metrics
            self.update_research_metrics(data)
            
            # Update charts efficiently
            return self.update_chart_plots(data)
            
        except Exception as e:
            print(f"Chart update error: {e}")
            return []
    
    def update_chart_plots(self, data):
        """Update chart plots efficiently"""
        try:
            if len(self.chart_data['time']) < 2:
                return []
            
            times = [t - self.chart_data['time'][0] for t in self.chart_data['time']]
            
            # Initialize charts if needed
            if not self.chart_initialized:
                self.initialize_charts()
                self.chart_initialized = True
            
            # Update line data
            self.chart_lines['left_eye'].set_data(times, self.chart_data['left_eye_openness'])
            self.chart_lines['right_eye'].set_data(times, self.chart_data['right_eye_openness'])
            self.chart_lines['blink_rate'].set_data(times, self.chart_data['blink_rate'])
            self.chart_lines['saccade_rate'].set_data(times, self.chart_data['saccade_rate'])
            self.chart_lines['fixation_duration'].set_data(times, self.chart_data['fixation_duration'])
            self.chart_lines['fatigue'].set_data(times, self.chart_data['fatigue'])
            self.chart_lines['cognitive_load'].set_data(times, self.chart_data['cognitive_load'])
            self.chart_lines['quality'].set_data(times, self.chart_data['quality'])
            self.chart_lines['gaze_stability'].set_data(times, self.chart_data['gaze_stability'])
            self.chart_lines['pupil_diameter'].set_data(times, self.chart_data['pupil_diameter'])
            self.chart_lines['eye_velocity'].set_data(times, self.chart_data['eye_velocity'])
            self.chart_lines['head_tilt'].set_data(times, self.chart_data['head_tilt'])
            self.chart_lines['head_yaw'].set_data(times, self.chart_data['head_yaw'])
            self.chart_lines['head_roll'].set_data(times, self.chart_data['head_roll'])
            
            # Update axis limits
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax5_twin]:
                ax.relim()
                ax.autoscale_view()
            
            return list(self.chart_lines.values())
            
        except Exception as e:
            print(f"Chart plot update error: {e}")
            return []
    
    def initialize_charts(self):
        """Initialize chart lines"""
        try:
            # Ax1: Eye State
            self.chart_lines['left_eye'], = self.ax1.plot([], [], color='#007bff', lw=2, alpha=0.9, label='Left Eye')
            self.chart_lines['right_eye'], = self.ax1.plot([], [], color='#28a745', lw=2, alpha=0.9, label='Right Eye')
            self.ax1.legend(fontsize=8)

            # Ax2: Event Rates
            self.chart_lines['blink_rate'], = self.ax2.plot([], [], color='#ffc107', lw=2, alpha=0.9, label='Blink Rate')
            self.chart_lines['saccade_rate'], = self.ax2.plot([], [], color='#fd7e14', lw=2, alpha=0.9, label='Saccade Rate')
            self.chart_lines['fixation_duration'], = self.ax2.plot([], [], color='#17a2b8', lw=2, alpha=0.9, label='Fixation Duration')
            self.ax2.legend(fontsize=8)

            # Ax3: Cognitive State
            self.chart_lines['fatigue'], = self.ax3.plot([], [], color='#dc3545', lw=2, alpha=0.9, label='Fatigue')
            self.chart_lines['cognitive_load'], = self.ax3.plot([], [], color='#6f42c1', lw=2, alpha=0.9, label='Cognitive Load')
            self.ax3.legend(fontsize=8)

            # Ax4: Data Quality
            self.chart_lines['quality'], = self.ax4.plot([], [], color='#20c997', lw=2, alpha=0.9, label='Detection Quality')
            self.chart_lines['gaze_stability'], = self.ax4.plot([], [], color='#6610f2', lw=2, alpha=0.9, label='Gaze Stability')
            self.chart_lines['head_tilt'], = self.ax4.plot([], [], color='#e83e8c', lw=1, linestyle='--', alpha=0.8, label='Head Tilt')
            self.chart_lines['head_yaw'], = self.ax4.plot([], [], color='#fd7e14', lw=1, linestyle='--', alpha=0.8, label='Head Yaw')
            self.chart_lines['head_roll'], = self.ax4.plot([], [], color='#6f42c1', lw=1, linestyle='--', alpha=0.8, label='Head Roll')
            self.ax4.legend(fontsize=8)

            # Ax5: Advanced Neurometrics (with twin y-axis)
            self.ax5_twin = self.ax5.twinx()
            self.chart_lines['pupil_diameter'], = self.ax5.plot([], [], color='#e83e8c', lw=2, alpha=0.9, label='Pupil Diameter (px)')
            self.chart_lines['eye_velocity'], = self.ax5_twin.plot([], [], color='#17a2b8', lw=2, alpha=0.9, label='Eye Velocity (°/s)')
            
            self.ax5.set_ylabel('Pupil Diameter (px)', color='#e83e8c')
            self.ax5_twin.set_ylabel('Eye Velocity (°/s)', color='#17a2b8')
            self.ax5.tick_params(axis='y', labelcolor='#e83e8c')
            self.ax5_twin.tick_params(axis='y', labelcolor='#17a2b8')
            lines = [self.chart_lines['pupil_diameter'], self.chart_lines['eye_velocity']]
            labels = [l.get_label() for l in lines]
            self.ax5.legend(lines, labels, loc='upper left', fontsize=8)

        except Exception as e:
            print(f"Chart initialization error: {e}")
    
    def update_research_metrics(self, data):
        """Update research metrics display"""
        try:
            # Update all research metrics
            self.research_metrics["advanced_fatigue"].config(
                text=f"{data.get('advanced_fatigue_score', 0):.3f}")
            
            self.research_metrics["advanced_quality"].config(
                text=f"{data.get('advanced_quality_score', 0):.3f}")
            
            self.research_metrics["cognitive_load"].config(
                text=f"{data.get('cognitive_load_score', 0):.3f}")
            
            self.research_metrics["pupil_diameter"].config(
                text=f"{data.get('pupil_diameter', 0):.1f}")
            
            self.research_metrics["gaze_stability"].config(
                text=f"{data.get('gaze_stability', 0):.3f}")
            
            self.research_metrics["eye_velocity"].config(
                text=f"{data.get('eye_velocity', 0):.3f}")
            
            self.research_metrics["fixation_duration"].config(
                text=f"{data.get('fixation_duration', 0):.2f}s")
            
            self.research_metrics["attention_span"].config(
                text=f"{data.get('attention_span', 0):.3f}")
            
            self.research_metrics["processing_speed"].config(
                text=f"{data.get('processing_speed', 0):.3f}")
            
            self.research_metrics["mental_effort"].config(
                text=f"{data.get('mental_effort', 0):.3f}")
            
            self.research_metrics["blink_rate"].config(
                text=f"{data.get('blink_rate', 0):.1f}")
            
            self.research_metrics["saccade_rate"].config(
                text=f"{data.get('saccade_rate', 0):.1f}")
            
            self.research_metrics["head_tilt"].config(
                text=f"{data.get('head_tilt', 0):.3f}")
            
            self.research_metrics["head_yaw"].config(
                text=f"{data.get('head_yaw', 0):.3f}")
            
            self.research_metrics["head_roll"].config(
                text=f"{data.get('head_roll', 0):.3f}")
            
            # Session metrics
            if self.session_start_time:
                session_duration = time.time() - self.session_start_time
                self.research_metrics["session_duration"].config(
                    text=f"{session_duration:.0f}s")
            
            # Research mode status
            research_mode = "Active" if self.tracker.research_mode else "Inactive"
            self.research_metrics["research_mode"].config(text=research_mode)
            
            # Data collection status
            total_events = len(self.tracker.research_data.get('events', []))
            self.research_metrics["total_events"].config(text=f"{total_events}")
            
            # Average quality
            if total_events > 0:
                avg_quality = np.mean([e.get('quality_score', 0) for e in self.tracker.research_data.get('events', [])])
                self.research_metrics["avg_quality"].config(text=f"{avg_quality:.3f}")
            
        except Exception as e:
            print(f"Research metrics update error: {e}")
    
    def start_research_session(self):
        """Start research session"""
        try:
            if not self.tracker.start_camera():
                messagebox.showerror("Error", "Failed to start camera")
                return
            
            self.is_running = True
            self.session_start_time = time.time()
            
            # Update UI state
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Research Session Active", fg='#28a745')
            
            # Start processing threads
            self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.processing_thread.start()
            
            # Start UI updates
            self.update_ui()
            
            print("🔬 Research session started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start research session: {str(e)}")
            self.stop_research_session()
    
    def stop_research_session(self):
        """Stop research session"""
        self.is_running = False
        self.tracker.stop_camera()
        
        # Update UI state
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Session Stopped", fg='#dc3545')
        
        # Clear video display
        if self.video_label:
            self.video_label.config(image='', text="Research session stopped")
        
        print("🛑 Research session stopped")
    
    def start_calibration(self):
        """Start calibration procedure"""
        try:
            calibration_points = self.tracker.start_calibration()
            if calibration_points:
                self.calibration_mode = True
                self.current_calibration_point = 0
                
                # Update UI state
                self.calib_start_btn.config(state='disabled')
                self.calib_next_btn.config(state='normal')
                self.calib_progress_label.config(text=f"Progress: {self.current_calibration_point + 1}/9 points")
                
                messagebox.showinfo("Calibration", f"Look at calibration point {self.current_calibration_point + 1}/9")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start calibration: {str(e)}")
    
    def next_calibration_point(self):
        """Move to next calibration point"""
        if not self.calibration_mode:
            return
        
        # Collect gaze data for current point
        gaze_data = []
        for _ in range(30):  # Collect 30 samples
            data = self.tracker.get_current_data()
            if 'pupil_position' in data:
                gaze_data.append(data['pupil_position'])
            time.sleep(0.033)  # ~30 FPS
        
        # Calibrate current point
        if gaze_data:
            self.tracker.calibrate_point(self.current_calibration_point, gaze_data)
        
        self.current_calibration_point += 1
        
        if self.current_calibration_point < 9:
            # Move to next point
            self.calib_progress_label.config(text=f"Progress: {self.current_calibration_point + 1}/9 points")
            messagebox.showinfo("Calibration", f"Look at calibration point {self.current_calibration_point + 1}/9")
        else:
            # Finish calibration
            self.finish_calibration()
    
    def finish_calibration(self):
        """Finish calibration procedure"""
        if self.tracker.finish_calibration():
            self.calibration_mode = False
            
            # Update UI state
            self.calib_next_btn.config(state='disabled')
            self.calib_finish_btn.config(state='normal')
            self.calib_quality_label.config(text=f"Quality: {self.tracker.calibration_quality:.3f}")
            
            messagebox.showinfo("Calibration Complete", 
                              f"Calibration completed with quality: {self.tracker.calibration_quality:.3f}")
        else:
            messagebox.showerror("Error", "Failed to complete calibration")
    
    def add_annotation(self):
        """Add research annotation"""
        annotation_text = self.annotation_entry.get().strip()
        if annotation_text:
            timestamp = time.time()
            annotation = {
                'timestamp': timestamp,
                'text': annotation_text,
                'session_time': timestamp - self.session_start_time if self.session_start_time else 0
            }
            
            self.tracker.research_data['annotations'].append(annotation)
            self.annotation_entry.delete(0, tk.END)
            
            print(f"📝 Annotation added: {annotation_text}")
    
    def export_research_data(self):
        """Export research data"""
        try:
            # Create comprehensive export data
            export_data = {
                'session_info': {
                    'session_start': self.session_start_time,
                    'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
                    'total_events': len(self.tracker.research_data.get('events', [])),
                    'calibration_quality': self.tracker.calibration_quality,
                    'research_mode': self.tracker.research_mode
                },
                'current_metrics': self.tracker.get_current_data(),
                'chart_data': self.chart_data,
                'research_data': self.tracker.research_data,
                'quality_indicators': {
                    key: label.cget('text') for key, label in self.quality_indicators.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Ask user for export format
            export_format = tk.StringVar(value="json")
            format_dialog = tk.Toplevel(self.root)
            format_dialog.title("Export Format")
            format_dialog.geometry("300x200")
            format_dialog.transient(self.root)
            format_dialog.grab_set()
            
            tk.Label(format_dialog, text="Select Export Format:", 
                    font=("SF Pro Text", 12, "bold")).pack(pady=10)
            
            tk.Radiobutton(format_dialog, text="JSON (Recommended)", 
                          variable=export_format, value="json").pack(anchor='w', padx=20)
            tk.Radiobutton(format_dialog, text="CSV", 
                          variable=export_format, value="csv").pack(anchor='w', padx=20)
            tk.Radiobutton(format_dialog, text="Excel", 
                          variable=export_format, value="excel").pack(anchor='w', padx=20)
            
            def confirm_export():
                format_dialog.destroy()
                self._perform_export(export_data, export_format.get())
            
            tk.Button(format_dialog, text="Export", command=confirm_export,
                     bg='#28a745', fg='white', font=("SF Pro Text", 10, "bold"),
                     relief='flat', padx=20, pady=5).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to prepare export data: {str(e)}")
    
    def _perform_export(self, export_data, export_format):
        """Perform the actual export based on format"""
        try:
            if export_format == "json":
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Export Research Data (JSON)"
                )
                if filename:
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    messagebox.showinfo("Export Complete", f"Research data exported to:\n{filename}")
            
            elif export_format == "csv":
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    title="Export Research Data (CSV)"
                )
                if filename:
                    self._export_to_csv(export_data, filename)
                    messagebox.showinfo("Export Complete", f"Research data exported to:\n{filename}")
            
            elif export_format == "excel":
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    title="Export Research Data (Excel)"
                )
                if filename:
                    self._export_to_excel(export_data, filename)
                    messagebox.showinfo("Export Complete", f"Research data exported to:\n{filename}")
                    
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _export_to_csv(self, export_data, filename):
        """Export data to CSV format"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write session info
            writer.writerow(['Session Information'])
            writer.writerow(['Session Start', export_data['session_info']['session_start']])
            writer.writerow(['Session Duration', export_data['session_info']['session_duration']])
            writer.writerow(['Total Events', export_data['session_info']['total_events']])
            writer.writerow(['Calibration Quality', export_data['session_info']['calibration_quality']])
            writer.writerow([])
            
            # Write current metrics
            writer.writerow(['Current Metrics'])
            for key, value in export_data['current_metrics'].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Write chart data
            writer.writerow(['Chart Data'])
            if export_data['chart_data']['time']:
                writer.writerow(['Time'] + [str(t) for t in export_data['chart_data']['time']])
                for metric, values in export_data['chart_data'].items():
                    if metric != 'time':
                        writer.writerow([metric] + [str(v) for v in values])
    
    def _export_to_excel(self, export_data, filename):
        """Export data to Excel format"""
        try:
            import pandas as pd
            
            # Create Excel writer
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Session info
                session_df = pd.DataFrame(list(export_data['session_info'].items()), 
                                        columns=['Metric', 'Value'])
                session_df.to_excel(writer, sheet_name='Session Info', index=False)
                
                # Current metrics
                current_df = pd.DataFrame(list(export_data['current_metrics'].items()), 
                                        columns=['Metric', 'Value'])
                current_df.to_excel(writer, sheet_name='Current Metrics', index=False)
                
                # Chart data
                if export_data['chart_data']['time']:
                    chart_df = pd.DataFrame(export_data['chart_data'])
                    chart_df.to_excel(writer, sheet_name='Chart Data', index=False)
                
                # Research events
                if export_data['research_data']['events']:
                    events_df = pd.DataFrame(export_data['research_data']['events'])
                    events_df.to_excel(writer, sheet_name='Research Events', index=False)
                    
        except ImportError:
            messagebox.showerror("Export Error", "Excel export requires pandas and openpyxl. Install with: pip install pandas openpyxl")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export to Excel: {str(e)}")
    
    def process_frames(self):
        """Process frames for research"""
        while self.is_running:
            try:
                frame = self.tracker.read_frame()
                if frame is not None:
                    processed_frame = self.tracker.process_frame(frame)
                    
                    # Update frame queue
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(processed_frame)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                time.sleep(0.02)
    
    def update_ui(self):
        """Update UI elements"""
        current_time = time.time()
        
        # Update video display
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                if frame is not None:
                    self.update_video_display(frame)
        except queue.Empty:
            pass
        
        # Update session duration
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.session_duration_label.config(text=f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update quality indicators
        data = self.tracker.get_current_data()
        if data:
            # Update data quality
            quality_score = data.get('advanced_quality_score', 0)
            if quality_score < self.quality_threshold.get():
                self.data_quality_label.config(text="Poor", fg="#dc3545")
                self.quality_warning_label.config(text="Data quality below threshold!", fg="#dc3545")
            else:
                self.data_quality_label.config(text="Good", fg="#28a745")
                self.quality_warning_label.config(text="", fg="#dc3545")
            
            # Update quality indicators
            self.quality_indicators["tracking_quality"].config(
                text=f"{data.get('advanced_quality_score', 0):.2f}")
            self.quality_indicators["calibration_quality"].config(
                text=f"{self.tracker.calibration_quality:.2f}")
            self.quality_indicators["face_detection"].config(
                text=f"{data.get('face_confidence', 0):.2f}")
            self.quality_indicators["pupil_tracking"].config(
                text="Active" if 'pupil_position' in data else "Inactive")
            self.quality_indicators["gaze_estimation"].config(
                text="Active" if 'gaze_point' in data else "Inactive")
            self.quality_indicators["data_collection_status"].config(
                text="Active" if self.is_running else "Inactive")
            
            # Update performance indicators
            if hasattr(self, 'last_frame_time'):
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.latency_label.config(text=f"Latency: {(current_time - self.last_frame_time)*1000:.1f}ms")
                self.last_frame_time = current_time
            else:
                self.last_frame_time = time.time()
            
            # Update data validation warnings
            self._update_data_validation_warnings(data)
        
        # Schedule next update
        if self.is_running:
            self.root.after(self.ui_update_interval, self.update_ui)
    
    def _update_data_validation_warnings(self, data):
        """Update data validation warnings based on current data quality"""
        try:
            # Data collection warning
            if not self.is_running:
                self.data_collection_warning_label.config(text="Data collection stopped", fg="#dc3545")
            else:
                self.data_collection_warning_label.config(text="", fg="#dc3545")
            
            # Calibration quality warning
            if self.tracker.calibration_quality < 0.7:
                self.calib_quality_warning_label.config(text="Low calibration quality", fg="#ffc107")
            else:
                self.calib_quality_warning_label.config(text="", fg="#ffc107")
            
            # Face detection warning
            face_confidence = data.get('face_confidence', 0)
            if face_confidence < 0.5:
                self.face_detection_warning_label.config(text="Face not detected clearly", fg="#dc3545")
            else:
                self.face_detection_warning_label.config(text="", fg="#dc3545")
            
            # Pupil tracking warning
            if 'pupil_position' not in data:
                self.pupil_tracking_warning_label.config(text="Pupil tracking lost", fg="#dc3545")
            else:
                self.pupil_tracking_warning_label.config(text="", fg="#dc3545")
            
            # Gaze estimation warning
            if 'gaze_point' not in data:
                self.gaze_estimation_warning_label.config(text="Gaze estimation unavailable", fg="#dc3545")
            else:
                self.gaze_estimation_warning_label.config(text="", fg="#dc3545")
            
            # Overall data quality warning
            quality_score = data.get('advanced_quality_score', 0)
            if quality_score < self.quality_threshold.get():
                self.quality_warning_label.config(text=f"Quality below threshold ({quality_score:.2f})", fg="#dc3545")
            else:
                self.quality_warning_label.config(text="", fg="#dc3545")
                
        except Exception as e:
            print(f"Data validation warning update error: {e}")
    
    def update_video_display(self, frame):
        """Update video display"""
        try:
            # Flip frame horizontally to create a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get latest data to draw on the flipped frame
            data = self.tracker.get_current_data()
            if data:
                # Prepare metrics text
                metrics = [
                    f"Fatigue: {data.get('advanced_fatigue_score', 0):.2f}",
                    f"Quality: {data.get('advanced_quality_score', 0):.2f}",
                    f"Pupil Diameter: {data.get('pupil_diameter', 0):.1f}px",
                    f"Head Yaw: {data.get('head_yaw', 0):.1f}",
                    f"Head Roll: {data.get('head_roll', 0):.1f}"
                ]
                
                # Draw metrics on the top-left of the frame
                y_offset = 30
                for metric in metrics:
                    # Draw black outline for better visibility
                    cv2.putText(frame, metric, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    # Draw the actual text in green
                    cv2.putText(frame, metric, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 25

            # Resize for display
            height, width = frame.shape[:2]
            target_width = 640
            target_height = 720
            
            scale_x = target_width / width
            scale_y = target_height / height
            scale = min(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
        except Exception as e:
            print(f"Video display error: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_research_session()
        self.root.destroy()
    
    def run(self):
        """Run the research UI"""
        self.create_ui()
        self.root.mainloop()

# Alias for backward compatibility
ResearchEyeTrackerUI = ResearchEyeTrackerUI 