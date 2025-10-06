import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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
from matplotlib import style

# Set matplotlib style for modern, clean look
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

class EyeTrackerUI:
    def __init__(self, tracker, data_logger):
        self.tracker = tracker
        self.data_logger = data_logger
        self.is_running = False
        self.root = None
        
        # Performance optimizations
        self.frame_queue = queue.Queue(maxsize=1)  # Single frame buffer for minimal lag
        self.ui_update_interval = 16  # Update UI every 16ms (60 FPS) for smooth performance
        self.last_ui_update = 0
        self.chart_update_interval = 200  # Update charts every 200ms for better responsiveness
        self.metrics_update_interval = 50  # Update metrics every 50ms for higher accuracy
        self.last_metrics_update = 0
        
        # Chart optimization
        self.chart_lines = {}  # Store line objects for efficient updates
        self.chart_initialized = False
        
        # UI elements
        self.video_label = None
        self.metrics_frame = None
        self.status_label = None
        self.start_btn = None
        self.stop_btn = None
        
        # Metrics display
        self.metric_labels = {}
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Chart data
        self.chart_data = {
            'time': [],
            'left_eye': [],
            'right_eye': [],
            'blink_rate': [],
            'fatigue': [],
            'quality': []
        }
        
        # Chart elements
        self.fig = None
        self.canvas = None
        self.ani = None
        self.last_chart_update = 0
        
    def create_ui(self):
        """Create the main UI window with modern design"""
        self.root = tk.Tk()
        self.root.title("Eye Tracking & Cognitive Fatigue Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#ffffff')  # Light background for modern look
        
        # Set window properties
        self.root.resizable(True, True)
        self.root.minsize(1200, 800)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=3)
        self.root.grid_rowconfigure(1, weight=2)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
        # Create metrics display
        self.create_detailed_metrics_display()
        
        # Create charts
        self.create_charts()
        
    def create_sidebar(self):
        """Create modern sidebar with clean design"""
        sidebar = tk.Frame(self.root, bg='#f8f9fa', width=280)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        sidebar.grid_propagate(False)
        
        # Title with modern styling
        title_frame = tk.Frame(sidebar, bg='#f8f9fa')
        title_frame.pack(fill='x', pady=(20, 25))
        
        title_label = tk.Label(title_frame, text="Eye Tracking & Cognitive Fatigue Detection", 
                              font=("SF Pro Display", 16, "bold"), 
                              bg='#f8f9fa', fg='#212529')
        title_label.pack()
        
        # Control buttons with modern design
        control_frame = tk.Frame(sidebar, bg='#f8f9fa')
        control_frame.pack(fill='x', pady=20, padx=20)
        
        # Modern button styling
        button_style = {
            'font': ("SF Pro Text", 11, "bold"),
            'relief': 'flat',
            'padx': 20,
            'pady': 12,
            'cursor': 'hand2',
            'borderwidth': 0
        }
        
        self.start_btn = tk.Button(control_frame, text="Start Tracking", 
                                  command=self.start_tracking,
                                  bg='#28a745', fg='white', 
                                  activebackground='#218838',
                                  **button_style)
        self.start_btn.pack(fill='x', pady=(0, 8))
        
        self.stop_btn = tk.Button(control_frame, text="Stop Tracking", 
                                 command=self.stop_tracking,
                                 bg='#dc3545', fg='white', 
                                 activebackground='#c82333',
                                 state='disabled',
                                 **button_style)
        self.stop_btn.pack(fill='x', pady=(0, 8))
        
        # Status indicator with modern design
        status_frame = tk.Frame(sidebar, bg='#f8f9fa')
        status_frame.pack(fill='x', pady=15, padx=20)
        
        # Status card
        status_card = tk.Frame(status_frame, bg='#ffffff', relief='flat', bd=1)
        status_card.pack(fill='x', pady=5)
        
        self.status_label = tk.Label(status_card, text="Ready", 
                                    font=("SF Pro Text", 12, "bold"), 
                                    bg='#ffffff', fg='#28a745')
        self.status_label.pack(pady=8)
        
        # FPS indicator
        self.fps_label = tk.Label(status_card, text="FPS: 0", 
                                 font=("SF Pro Text", 10), 
                                 bg='#ffffff', fg='#6c757d')
        self.fps_label.pack(pady=(0, 8))
        
        # Separator
        separator = tk.Frame(sidebar, height=1, bg='#dee2e6')
        separator.pack(fill='x', pady=20, padx=20)
        

        
    def create_main_content(self):
        """Create main content area with modern video display"""
        main_frame = tk.Frame(self.root, bg='#ffffff')
        main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Video display with modern styling
        video_frame = tk.Frame(main_frame, bg='#ffffff', relief='flat', bd=1)
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video title
        video_title = tk.Label(video_frame, text="Live Camera Feed", 
                              font=("SF Pro Display", 16, "bold"), 
                              bg='#ffffff', fg='#212529')
        video_title.pack(pady=12)
        
        # Video container with rounded corners effect
        video_container = tk.Frame(video_frame, bg='#000000', relief='flat', bd=2)
        video_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.video_label = tk.Label(video_container, bg='#000000', text="Camera feed will appear here",
                                   font=("SF Pro Text", 12), fg='#ffffff')
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Instructions with modern design
        instructions_frame = tk.Frame(main_frame, bg='#f8f9fa', relief='flat', bd=1)
        instructions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        instructions_title = tk.Label(instructions_frame, text="Instructions", 
                                     font=("SF Pro Display", 14, "bold"), 
                                     bg='#f8f9fa', fg='#212529')
        instructions_title.pack(pady=(12, 8))
        
        instructions = [
            "Position yourself in front of the camera",
            "Look straight ahead for optimal calibration",
            "Monitor real-time metrics and fatigue detection"
        ]
        
        for instruction in instructions:
            label = tk.Label(instructions_frame, text=f"• {instruction}", 
                           font=("SF Pro Text", 11), 
                           bg='#f8f9fa', fg='#6c757d')
            label.pack(anchor='w', padx=20, pady=2)
        
        tk.Label(instructions_frame, text="", bg='#f8f9fa').pack(pady=12)
    
    def create_detailed_metrics_display(self):
        """Create real-time metrics display with updating labels"""
        metrics_frame = tk.Frame(self.root, bg='#f8f9fa', relief='flat', bd=1)
        metrics_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Title
        metrics_title = tk.Label(metrics_frame, text="Real-time Metrics", 
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
        
        # Pack scrollable elements
        canvas.pack(side="left", fill="both", expand=True, padx=15)
        scrollbar.pack(side="right", fill="y")
        
        # Detailed metrics list with labels that update in place
        detailed_metrics = [
            ("Eye Openness", "eye_openness", "#007bff"),
            ("Blink Rate", "blink_rate", "#28a745"),
            ("Fatigue Score", "fatigue_score", "#dc3545"),
            ("Detection Quality", "quality", "#6f42c1"),
            ("Calibration Status", "calibration", "#6c757d"),
            ("Blink Duration", "blink_duration", "#17a2b8"),
            ("Total Blinks", "total_blinks", "#6610f2"),
            ("Session Duration", "session_time", "#20c997"),
            ("Avg Blink Duration", "avg_blink_duration", "#17a2b8"),
            ("Fatigue Trend", "fatigue_trend", "#dc3545"),
            ("Eye Movement Speed", "eye_speed", "#ff6b6b"),
            ("Fixation Duration", "fixation_duration", "#4ecdc4"),
            ("Pupil Diameter", "pupil_diameter", "#45b7d1"),
            ("Head Movement", "head_movement", "#96ceb4"),
            ("Attention Score", "attention_score", "#feca57")
        ]
        
        for label_text, key, color in detailed_metrics:
            # Create metric row
            metric_row = tk.Frame(scrollable_frame, bg='#ffffff', relief='flat', bd=1)
            metric_row.pack(fill='x', pady=2, padx=5)
            
            # Metric title
            title_label = tk.Label(metric_row, text=label_text, 
                                 font=("SF Pro Text", 10, "bold"), 
                                 bg='#ffffff', fg='#6c757d', width=20, anchor='w')
            title_label.pack(side='left', padx=12, pady=8)
            
            # Metric value (updates in place)
            value_label = tk.Label(metric_row, text="--", 
                                 font=("SF Pro Display", 11, "bold"), 
                                 bg='#ffffff', fg=color, width=15, anchor='w')
            value_label.pack(side='right', padx=12, pady=8)
            
            self.metric_labels[key] = value_label
    
    def create_charts(self):
        """Create modern, clean charts"""
        charts_frame = tk.Frame(self.root, bg='#ffffff', relief='flat', bd=1)
        charts_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        # Charts title
        charts_title = tk.Label(charts_frame, text="Analytics Dashboard", 
                               font=("SF Pro Display", 18, "bold"), 
                               bg='#ffffff', fg='#212529')
        charts_title.pack(pady=15)
        
        # Create figure with modern styling
        self.fig = Figure(figsize=(14, 6), facecolor='#ffffff', dpi=100)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # Eye openness
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Blink rate
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Saccade rate
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Fatigue score
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Quality score
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Microsaccade rate
        
        # Configure subplots with modern styling
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(colors='#6c757d', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dee2e6')
            ax.spines['bottom'].set_color('#dee2e6')
            
        # Set titles with modern styling
        titles = ['Eye Openness', 'Blink Rate', 'Additional Metric', 
                 'Fatigue Score', 'Quality', 'Additional Metric']
        axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]
        
        for ax, title in zip(axes, titles):
            ax.set_title(title, color='#212529', fontsize=11, fontweight='bold', pad=8)
            
        # Adjust layout
        self.fig.tight_layout(pad=2.0)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Start animation with reduced frequency
        self.start_animation()
    
    def start_animation(self):
        """Start the chart animation with optimized frequency"""
        self.ani = animation.FuncAnimation(self.fig, self.update_charts, 
                                          interval=200, blit=True, cache_frame_data=False)
    
    def update_charts(self, frame):
        """Update the charts with high accuracy and minimal lag"""
        try:
            if not self.is_running:
                return []
                
            current_time = time.time()
            
            # Get latest data with high accuracy
            data = self.tracker.get_current_data()
            if not data:
                return []
            
            # Update chart data with precision
            self.chart_data['time'].append(current_time)
            self.chart_data['left_eye'].append(float(data.get('left_eye_openness', 0)))
            self.chart_data['right_eye'].append(float(data.get('right_eye_openness', 0)))
            self.chart_data['blink_rate'].append(float(data.get('blink_rate', 0)))
            self.chart_data['fatigue'].append(float(data.get('overall_fatigue_score', 0)))
            self.chart_data['quality'].append(float(data.get('quality_score', 0)))
            
            # Keep only last 20 data points for better accuracy
            max_points = 20
            for key in self.chart_data:
                if len(self.chart_data[key]) > max_points:
                    self.chart_data[key] = self.chart_data[key][-max_points:]
            
            # Update detailed metrics with high frequency
            self.update_detailed_metrics(data)
            
            # Update charts efficiently and return artists for blit
            return self.update_chart_plots_efficient(data)
                
        except Exception as e:
            print(f"Error updating charts: {e}")
            return []
    
    def update_chart_plots_efficient(self, data):
        """Update charts efficiently using line objects for minimal lag"""
        try:
            if len(self.chart_data['time']) < 2:
                return []
            
            times = [t - self.chart_data['time'][0] for t in self.chart_data['time']]
            
            # Initialize charts if not done yet
            if not self.chart_initialized:
                self.initialize_charts()
                self.chart_initialized = True
            
            # Update line data efficiently
            self.chart_lines['left_eye'].set_data(times, self.chart_data['left_eye'])
            self.chart_lines['right_eye'].set_data(times, self.chart_data['right_eye'])
            self.chart_lines['blink_rate'].set_data(times, self.chart_data['blink_rate'])
            self.chart_lines['fatigue'].set_data(times, self.chart_data['fatigue'])
            self.chart_lines['quality'].set_data(times, self.chart_data['quality'])
            
            # Update axis limits for smooth scrolling
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                ax.relim()
                ax.autoscale_view()
            
            # Return all line artists for blit animation
            return list(self.chart_lines.values())
                
        except Exception as e:
            print(f"Error updating chart plots efficiently: {e}")
            return []
    
    def initialize_charts(self):
        """Initialize chart lines for efficient updates"""
        try:
            # Eye openness chart
            self.ax1.set_facecolor('#f8f9fa')
            self.ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax1.tick_params(colors='#6c757d', labelsize=8)
            self.ax1.spines['top'].set_visible(False)
            self.ax1.spines['right'].set_visible(False)
            self.ax1.spines['left'].set_color('#dee2e6')
            self.ax1.spines['bottom'].set_color('#dee2e6')
            self.ax1.set_title('Eye Openness', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['left_eye'], = self.ax1.plot([], [], color='#007bff', linewidth=2, alpha=0.8, label='Left')
            self.chart_lines['right_eye'], = self.ax1.plot([], [], color='#28a745', linewidth=2, alpha=0.8, label='Right')
            self.ax1.legend(fontsize=8, framealpha=0.9)
            
            # Blink rate chart
            self.ax2.set_facecolor('#f8f9fa')
            self.ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax2.tick_params(colors='#6c757d', labelsize=8)
            self.ax2.spines['top'].set_visible(False)
            self.ax2.spines['right'].set_visible(False)
            self.ax2.spines['left'].set_color('#dee2e6')
            self.ax2.spines['bottom'].set_color('#dee2e6')
            self.ax2.set_title('Blink Rate', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['blink_rate'], = self.ax2.plot([], [], color='#ffc107', linewidth=2, alpha=0.8)
            
            # Saccade rate chart
            self.ax3.set_facecolor('#f8f9fa')
            self.ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax3.tick_params(colors='#6c757d', labelsize=8)
            self.ax3.spines['top'].set_visible(False)
            self.ax3.spines['right'].set_visible(False)
            self.ax3.spines['left'].set_color('#dee2e6')
            self.ax3.spines['bottom'].set_color('#dee2e6')
            self.ax3.set_title('Additional Metric', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['additional_metric'], = self.ax3.plot([], [], color='#fd7e14', linewidth=2, alpha=0.8)
            
            # Fatigue chart
            self.ax4.set_facecolor('#f8f9fa')
            self.ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax4.tick_params(colors='#6c757d', labelsize=8)
            self.ax4.spines['top'].set_visible(False)
            self.ax4.spines['right'].set_visible(False)
            self.ax4.spines['left'].set_color('#dee2e6')
            self.ax4.spines['bottom'].set_color('#dee2e6')
            self.ax4.set_title('Fatigue Score', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['fatigue'], = self.ax4.plot([], [], color='#dc3545', linewidth=2, alpha=0.8)
            
            # Quality chart
            self.ax5.set_facecolor('#f8f9fa')
            self.ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax5.tick_params(colors='#6c757d', labelsize=8)
            self.ax5.spines['top'].set_visible(False)
            self.ax5.spines['right'].set_visible(False)
            self.ax5.spines['left'].set_color('#dee2e6')
            self.ax5.spines['bottom'].set_color('#dee2e6')
            self.ax5.set_title('Quality', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['quality'], = self.ax5.plot([], [], color='#6f42c1', linewidth=2, alpha=0.8)
            
            # Additional metric chart
            self.ax6.set_facecolor('#f8f9fa')
            self.ax6.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax6.tick_params(colors='#6c757d', labelsize=8)
            self.ax6.spines['top'].set_visible(False)
            self.ax6.spines['right'].set_visible(False)
            self.ax6.spines['left'].set_color('#dee2e6')
            self.ax6.spines['bottom'].set_color('#dee2e6')
            self.ax6.set_title('Additional Metric', color='#212529', fontsize=11, fontweight='bold', pad=8)
            
            self.chart_lines['additional_metric'], = self.ax6.plot([], [], color='#20c997', linewidth=2, alpha=0.8)
                
        except Exception as e:
            print(f"Error initializing charts: {e}")
    
    def update_detailed_metrics(self, data):
        """Update metrics display with current data"""
        try:
            # Update all detailed metrics in place
            self.metric_labels["eye_openness"].config(
                text=f"L:{data.get('left_eye_openness', 0):.3f} R:{data.get('right_eye_openness', 0):.3f}")
            
            self.metric_labels["blink_rate"].config(
                text=f"{data.get('blink_rate', 0):.1f}/min")
            
            self.metric_labels["fatigue_score"].config(
                text=f"{data.get('overall_fatigue_score', 0):.2f}")
            
            self.metric_labels["quality"].config(
                text=f"{data.get('quality_score', 0):.2f}")
            
            # Calibration status
            calibration_status = "Ready" if data.get('calibration_complete', False) else "Calibrating..."
            self.metric_labels["calibration"].config(text=calibration_status)
            
            # Blink duration
            blink_duration = data.get('blink_duration', 0)
            self.metric_labels["blink_duration"].config(text=f"{blink_duration:.3f}s")
            
            # Fatigue level
            fatigue_level = data.get('fatigue_level', 'Normal')
            self.metric_labels["fatigue_level"].config(text=fatigue_level)
            
            # Total counts
            total_blinks = data.get('blink_counter', 0)
            self.metric_labels["total_blinks"].config(text=f"{total_blinks}")
            
            # Session duration
            if hasattr(self, 'session_start_time'):
                session_time = time.time() - self.session_start_time
                self.metric_labels["session_time"].config(text=f"{session_time:.0f}s")
            
            # Average blink duration
            blink_durations = data.get('blink_durations', [])
            if blink_durations:
                avg_blink_duration = np.mean(blink_durations)
                self.metric_labels["avg_blink_duration"].config(text=f"{avg_blink_duration:.3f}s")
            else:
                self.metric_labels["avg_blink_duration"].config(text="--")
            
            # Fatigue trend
            fatigue_score = data.get('overall_fatigue_score', 0)
            fatigue_trend = "Increasing" if fatigue_score > 0.5 else "Decreasing" if fatigue_score < 0.3 else "Stable"
            self.metric_labels["fatigue_trend"].config(text=fatigue_trend)
            
            # Additional metrics (simulated values for demonstration)
            self.metric_labels["eye_speed"].config(text=f"{data.get('eye_speed', 0):.2f}°/s")
            self.metric_labels["fixation_duration"].config(text=f"{data.get('fixation_duration', 0.3):.2f}s")
            self.metric_labels["pupil_diameter"].config(text=f"{data.get('pupil_diameter', 4.2):.1f}mm")
            self.metric_labels["head_movement"].config(text=f"{data.get('head_movement', 'Stable')}")
            self.metric_labels["attention_score"].config(text=f"{data.get('attention_score', 0.85):.2f}")
                
        except Exception as e:
            print(f"Error updating detailed metrics: {e}")
    
    def start_tracking(self):
        """Start eye tracking with optimized performance"""
        try:
            self.tracker.start_camera()
            self.is_running = True
            self.session_start_time = time.time()
            
            # Update UI state
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Tracking", fg='#28a745')
            
            # Start processing threads
            self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.processing_thread.start()
            
            # Start UI updates
            self.update_ui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start tracking: {str(e)}")
            self.stop_tracking()
    
    def stop_tracking(self):
        """Stop eye tracking"""
        self.is_running = False
        self.tracker.stop_camera()
        
        # Update UI state
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Stopped", fg='#dc3545')
        
        # Clear video display
        if self.video_label:
            self.video_label.config(image='', text="Camera stopped")
    
    def process_frames(self):
        """Ultra-optimized frame processing with minimal lag"""
        while self.is_running:
            try:
                # Read frame with timeout
                frame = self.tracker.read_frame()
                if frame is not None:
                    # Process frame
                    processed_frame = self.tracker.process_frame(frame)
                    
                    # Single frame buffer - always replace for minimal lag
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(processed_frame)
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Minimal delay for maximum responsiveness
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                time.sleep(0.02)
    
    def update_ui(self):
        """Ultra-optimized UI updates with minimal lag"""
        current_time = time.time()
        
        # Update video display immediately (highest priority)
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                if frame is not None:
                    self.update_video_display(frame)
        except queue.Empty:
            pass
        
        # Throttle other updates to prevent lag
        if current_time - self.last_ui_update >= (self.ui_update_interval / 1000.0):
            self.last_ui_update = current_time
            
            # Update FPS display
            self.fps_label.config(text=f"FPS: {self.current_fps}")
        
        # Throttle metrics updates separately
        if current_time - self.last_metrics_update >= (self.metrics_update_interval / 1000.0):
            self.last_metrics_update = current_time
            self.update_metrics()
        
        # Schedule next update with minimal interval
        if self.is_running:
            self.root.after(10, self.update_ui)
    
    def update_video_display(self, frame):
        """Update video display with optimized image processing and mirroring fix"""
        try:
            # Fix camera mirroring by flipping horizontally
            frame = cv2.flip(frame, 1)
            
            # Resize frame for display - optimized for performance
            height, width = frame.shape[:2]
            
            # Target size for optimal display
            target_width = 640   # Standard resolution for better performance
            target_height = 480  # 4:3 aspect ratio
            
            # Calculate scaling to fit the target size while maintaining aspect ratio
            scale_x = target_width / width
            scale_y = target_height / height
            scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize the frame with optimized interpolation
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Video display error: {e}")
    
    def update_metrics(self):
        """Update metrics display with current tracker data"""
        try:
            # Get current data from tracker
            data = self.tracker.get_current_data()
            
            if data:
                # Update key metrics with modern formatting
                self.metric_labels["eye_openness"].config(
                    text=f"L: {data.get('left_eye_openness', 0):.3f} | R: {data.get('right_eye_openness', 0):.3f}")
                
                self.metric_labels["blink_rate"].config(
                    text=f"{data.get('blink_rate', 0):.1f}/min")
                
                self.metric_labels["saccade_rate"].config(
                    text=f"{data.get('saccade_rate', 0):.1f}/min")
                
                self.metric_labels["fatigue_score"].config(
                    text=f"{data.get('overall_fatigue_score', 0):.2f}")
                
                self.metric_labels["quality"].config(
                    text=f"{data.get('quality_score', 0):.2f}")
                
                # Session tracking
                if hasattr(self, 'session_start_time'):
                    session_time = time.time() - self.session_start_time
                    self.metric_labels["session_time"].config(
                        text=f"{session_time:.0f}s")
                
        except Exception as e:
            print(f"Metrics update error: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_tracking()
        self.root.destroy()
    
    def run(self):
        """Run the optimized UI"""
        self.create_ui()
        self.root.mainloop()

# Alias for backward compatibility
ModernEyeTrackerUI = EyeTrackerUI 