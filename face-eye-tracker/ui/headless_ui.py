import cv2
import time
import numpy as np
from datetime import datetime

class HeadlessUI:
    def __init__(self, tracker, data_logger):
        self.tracker = tracker
        self.data_logger = data_logger
        self.is_running = False

    def run(self):
        print("Running in headless mode. Press 'q' to quit.")
        if not self.tracker.start_camera():
            print("Failed to start camera.")
            return

        self.is_running = True
        self.data_logger.start_logging()

        while self.is_running:
            frame = self.tracker.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self.tracker.process_frame(frame)
            if result is None:
                cv2.imshow('Headless Eye Tracker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                continue
            
            self.data_logger.log_data(result)

            processed_frame = result.get('frame', frame)

            # Display the frame
            cv2.imshow('Headless Eye Tracker', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        self.tracker.stop_camera()
        self.data_logger.stop_logging()
        cv2.destroyAllWindows() 