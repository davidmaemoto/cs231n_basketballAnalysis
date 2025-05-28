import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


class BasketballShotAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Basketball Shot Analysis")
        self.root.geometry("1200x800")

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Data storage
        self.data_dir = None
        self.shot_vids_dir = None
        self.shot_timestamps_dir = None
        self.video_files = []
        self.current_video_idx = 0
        self.current_shot_idx = 0
        self.current_frame_idx = 0
        self.video_cap = None
        self.shots_data = None
        self.extracted_clip_frames = []
        self.processing_paused = True

        # Create UI elements
        self.setup_ui()

        # Default folder selection (can be updated via UI)
        self.select_data_directory()

    def setup_ui(self):
        # Main frame arrangement
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Select directory button
        ttk.Button(control_frame, text="Select Data Directory", command=self.select_data_directory).pack(fill=tk.X,
                                                                                                         padx=5, pady=5)

        # Video selection frame
        video_select_frame = ttk.LabelFrame(control_frame, text="Video Selection")
        video_select_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(video_select_frame, text="Previous Video", command=lambda: self.change_video(-1)).pack(side=tk.LEFT,
                                                                                                          padx=5,
                                                                                                          pady=5)
        ttk.Button(video_select_frame, text="Next Video", command=lambda: self.change_video(1)).pack(side=tk.RIGHT,
                                                                                                     padx=5, pady=5)

        self.video_label = ttk.Label(video_select_frame, text="No video selected")
        self.video_label.pack(fill=tk.X, padx=5, pady=5)

        # Shot selection frame
        shot_select_frame = ttk.LabelFrame(control_frame, text="Shot Selection")
        shot_select_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(shot_select_frame, text="Previous Shot", command=lambda: self.change_shot(-1)).pack(side=tk.LEFT,
                                                                                                       padx=5, pady=5)
        ttk.Button(shot_select_frame, text="Next Shot", command=lambda: self.change_shot(1)).pack(side=tk.RIGHT, padx=5,
                                                                                                  pady=5)

        self.shot_label = ttk.Label(shot_select_frame, text="No shot selected")
        self.shot_label.pack(fill=tk.X, padx=5, pady=5)

        # Playback control frame
        playback_frame = ttk.LabelFrame(control_frame, text="Playback Controls")
        playback_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(playback_frame, text="Play/Pause", command=self.toggle_playback).pack(fill=tk.X, padx=5, pady=5)

        # Frame navigation
        frame_nav_frame = ttk.Frame(playback_frame)
        frame_nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_nav_frame, text="< Prev Frame", command=lambda: self.change_frame(-1)).pack(side=tk.LEFT,
                                                                                                     padx=5)
        self.frame_label = ttk.Label(frame_nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_nav_frame, text="Next Frame >", command=lambda: self.change_frame(1)).pack(side=tk.RIGHT,
                                                                                                    padx=5)

        # Shot info
        self.shot_info = ttk.Label(control_frame, text="No shot loaded")
        self.shot_info.pack(fill=tk.X, padx=5, pady=10)

        # Video display panel (right side)
        display_frame = ttk.LabelFrame(main_frame, text="Video Display")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_data_directory(self):
        """Let user select the main data directory"""
        self.data_dir = filedialog.askdirectory(title="Select Data Directory")

        if not self.data_dir:
            return

        self.shot_vids_dir = os.path.join(self.data_dir, "shot_vids")
        self.shot_timestamps_dir = os.path.join(self.data_dir, "shot_timestamps")

        # Validate directories
        if not os.path.exists(self.shot_vids_dir) or not os.path.exists(self.shot_timestamps_dir):
            tk.messagebox.showerror("Error",
                                    "Invalid data directory structure. Expected 'shot_vids' and 'shot_timestamps' subdirectories.")
            return

        # Get all MOV files
        self.video_files = [f for f in os.listdir(self.shot_vids_dir) if f.endswith('.MOV')]
        self.video_files.sort()

        if not self.video_files:
            tk.messagebox.showinfo("Info", "No .MOV files found in the shot_vids directory.")
            return

        # Load the first video
        self.current_video_idx = 0
        self.load_current_video()

    def load_current_video(self):
        """Load the currently selected video and its associated timestamp data"""
        if not self.video_files:
            return

        # Close previous video if open
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

        # Get current video filename
        video_filename = self.video_files[self.current_video_idx]
        video_path = os.path.join(self.shot_vids_dir, video_filename)

        # Update video label
        self.video_label.config(text=f"Video: {video_filename} ({self.current_video_idx + 1}/{len(self.video_files)})")

        # Load video
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            tk.messagebox.showerror("Error", f"Could not open video: {video_path}")
            return

        # Get corresponding timestamp CSV
        base_name = os.path.splitext(video_filename)[0]
        csv_filename = f"{base_name}_timestamps.csv"
        csv_path = os.path.join(self.shot_timestamps_dir, csv_filename)

        if not os.path.exists(csv_path):
            tk.messagebox.showwarning("Warning", f"No timestamp file found: {csv_path}")
            self.shots_data = None
            return

        # Load timestamps
        self.shots_data = pd.read_csv(csv_path)

        # Reset shot index
        self.current_shot_idx = 0

        # Load the first shot
        self.load_current_shot()

    def load_current_shot(self):
        """Load the current shot based on timestamp data"""
        if self.shots_data is None or len(self.shots_data) == 0:
            self.shot_label.config(text="No shots available")
            self.shot_info.config(text="No timestamp data found")
            return

        # Get current shot data
        shot_data = self.shots_data.iloc[self.current_shot_idx]

        # Update shot label
        self.shot_label.config(text=f"Shot: {self.current_shot_idx + 1}/{len(self.shots_data)}")

        # Extract start time
        start_time_sec = shot_data.get('start_time_sec', 0)

        # Update shot info
        shot_info_text = f"Start Time: {start_time_sec:.2f} seconds\n"
        for col in self.shots_data.columns:
            if col != 'start_time_sec':
                shot_info_text += f"{col}: {shot_data[col]}\n"
        self.shot_info.config(text=shot_info_text)

        # Extract the clip
        self.extract_clip(start_time_sec, duration=2.5)

    def extract_clip(self, start_time_sec, duration=2.5):
        """Extract a clip of specified duration from the video starting at start_time_sec"""
        if not self.video_cap or not self.video_cap.isOpened():
            return

        # Clear previous clip
        self.extracted_clip_frames = []

        # Calculate frame positions
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time_sec * fps)
        num_frames = int(duration * fps)

        # Set video position to start frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Extract frames
        for _ in range(num_frames):
            ret, frame = self.video_cap.read()
            if not ret:
                break

            # Apply MediaPipe pose detection
            processed_frame = self.process_frame_with_mediapipe(frame)
            self.extracted_clip_frames.append(processed_frame)

        # Reset frame index
        self.current_frame_idx = 0
        self.processing_paused = True

        # Update frame label
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")

        # Display first frame
        self.display_current_frame()

    def process_frame_with_mediapipe(self, frame):
        """Apply MediaPipe pose detection to the frame"""
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(frame_rgb)

        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        return annotated_frame

    def display_current_frame(self):
        """Display the current frame on the canvas"""
        if not self.extracted_clip_frames or self.current_frame_idx >= len(self.extracted_clip_frames):
            return

        # Get current frame
        frame = self.extracted_clip_frames[self.current_frame_idx]

        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly initialized, use default size
            canvas_width = 800
            canvas_height = 600

        # Calculate scaling factor to maintain aspect ratio
        h, w = frame.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Convert to PIL format
        image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        # Update canvas
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(new_w // 2, new_h // 2, image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection

    def change_video(self, delta):
        """Change to the next or previous video"""
        if not self.video_files:
            return

        new_idx = self.current_video_idx + delta
        if 0 <= new_idx < len(self.video_files):
            self.current_video_idx = new_idx
            self.load_current_video()

    def change_shot(self, delta):
        """Change to the next or previous shot in the current video"""
        if self.shots_data is None:
            return

        new_idx = self.current_shot_idx + delta
        if 0 <= new_idx < len(self.shots_data):
            self.current_shot_idx = new_idx
            self.load_current_shot()

    def change_frame(self, delta):
        """Change to the next or previous frame in the current clip"""
        if not self.extracted_clip_frames:
            return

        new_idx = self.current_frame_idx + delta
        if 0 <= new_idx < len(self.extracted_clip_frames):
            self.current_frame_idx = new_idx
            self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")
            self.display_current_frame()

    def toggle_playback(self):
        """Toggle playback of the clip"""
        self.processing_paused = not self.processing_paused

        if not self.processing_paused:
            self.play_clip()

    def play_clip(self):
        """Play the clip from the current frame"""
        if self.processing_paused or not self.extracted_clip_frames:
            return

        # Display current frame
        self.display_current_frame()

        # Advance to next frame
        self.current_frame_idx += 1

        # Check if we've reached the end
        if self.current_frame_idx >= len(self.extracted_clip_frames):
            self.current_frame_idx = 0

        # Update frame label
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")

        # Schedule next frame
        self.root.after(33, self.play_clip)  # ~30fps


def main():
    root = tk.Tk()
    app = BasketballShotAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()