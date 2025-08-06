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
        self.data_queue = queue.Queue()
        
        # Chart data
        self.chart_data = {
            'time': [],
            'left_eye': [],
            'right_eye': [],
            'blink_rate': [],
            'fatigue': [],
            'quality': []
        }
        
    def create_ui(self):
        """Create the main UI window"""
        self.root = tk.Tk()
        self.root.title("Eye Tracking & Fatigue Detection - Comprehensive")
        self.root.geometry("1600x1000")
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
        
        # Create text data area
        self.create_text_data_area()
        
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
        self.video_label.pack(expand=True, fill='both', padx=10, pady=10)
        
    def create_text_data_area(self):
        """Create the text data display area"""
        text_frame = tk.Frame(self.root, bg='#1e1e1e')
        text_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        # Title
        text_title = tk.Label(text_frame, text="Live Data Log", 
                             font=("Arial", 16, "bold"), 
                             bg='#1e1e1e', fg='white')
        text_title.pack(pady=10)
        
        # Create scrolled text widget
        self.text_widget = scrolledtext.ScrolledText(
            text_frame, 
            width=60, 
            height=15,
            bg='#2d2d2d',
            fg='white',
            font=("Consolas", 10),
            insertbackground='white'
        )
        self.text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add initial text
        self.text_widget.insert(tk.END, "Eye Tracking Data Log\n")
        self.text_widget.insert(tk.END, "=" * 50 + "\n")
        self.text_widget.insert(tk.END, "Waiting for tracking to start...\n")
        
    def start_data_processing(self):
        """Start the data processing loop"""
        self.root.after(100, self.process_data_queue)
        
    def process_data_queue(self):
        """Process data from the queue"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self.update_ui(data)
                self.update_text_data(data)
                self.update_video(data)
        except queue.Empty:
            pass
        
        # Schedule next processing
        self.root.after(100, self.process_data_queue)
        
    def start_animation(self):
        """Start the chart animation"""
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_charts, 
            interval=500,  # Update every 500ms
            blit=False, 
            cache_frame_data=False
        )
        
    def update_charts(self, frame):
        """Update the charts with new data"""
        try:
            if not self.is_running:
                return
                
            # Get latest data
            chart_data = self.data_logger.get_chart_data()
            
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
        """Update the video display"""
        try:
            if 'frame' in data and data['frame'] is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit display
                height, width = frame_rgb.shape[:2]
                max_size = 300
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the label
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error updating video: {e}")
            
    def update_text_data(self, data):
        """Update the text data display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format the data as text
        text_data = f"""
[{timestamp}] Eye Tracking Data:
├─ Eye Openness: L={data['left_eye_openness']:.3f}, R={data['right_eye_openness']:.3f}
├─ Blink Rate: {data['blink_rate']:.1f}/min
├─ Saccade Rate: {data['saccade_rate']:.1f}/min
├─ Quality Score: {data['quality_score']:.2f}
├─ Fatigue Level: {data['fatigue_level']}
├─ Fatigue Scores:
│  ├─ Blink Rate: {data['fatigue_blink_rate_score']:.2f}
│  ├─ Blink Duration: {data['fatigue_blink_duration_score']:.2f}
│  ├─ Eye Openness: {data['fatigue_eye_openness_score']:.2f}
│  ├─ Yawn Detection: {data['fatigue_yawn_score']:.2f}
│  ├─ Gaze Drift: {data['fatigue_gaze_drift_score']:.2f}
│  └─ Head Slouch: {data['fatigue_head_slouch_score']:.2f}
└─ Overall Fatigue: {data['overall_fatigue_score']:.2f}
"""
        
        # Add to text widget
        self.text_widget.insert(tk.END, text_data)
        
        # Auto-scroll to bottom
        self.text_widget.see(tk.END)
        
        # Keep only last 50 entries
        lines = self.text_widget.get("1.0", tk.END).split('\n')
        if len(lines) > 1000:  # Keep reasonable amount of text
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, "\n".join(lines[-500:]))
            
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
                
                # Add to queue for UI update
                self.data_queue.put(result)
                
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