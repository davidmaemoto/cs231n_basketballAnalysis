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
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
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
        self.setup_ui()
        self.select_data_directory()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ttk.Button(control_frame, text="Select Data Directory", command=self.select_data_directory).pack(fill=tk.X,
                                                                                                         padx=5, pady=5)
        video_select_frame = ttk.LabelFrame(control_frame, text="Video Selection")
        video_select_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(video_select_frame, text="Previous Video", command=lambda: self.change_video(-1)).pack(side=tk.LEFT,
                                                                                                          padx=5,
                                                                                                          pady=5)
        ttk.Button(video_select_frame, text="Next Video", command=lambda: self.change_video(1)).pack(side=tk.RIGHT,
                                                                                                     padx=5, pady=5)

        self.video_label = ttk.Label(video_select_frame, text="No video selected")
        self.video_label.pack(fill=tk.X, padx=5, pady=5)
        shot_select_frame = ttk.LabelFrame(control_frame, text="Shot Selection")
        shot_select_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(shot_select_frame, text="Previous Shot", command=lambda: self.change_shot(-1)).pack(side=tk.LEFT,
        ttk.Button(shot_select_frame, text="Next Shot", command=lambda: self.change_shot(1)).pack(side=tk.RIGHT, padx=5,
                                                                                                  pady=5)
        self.shot_label = ttk.Label(shot_select_frame, text="No shot selected")
        self.shot_label.pack(fill=tk.X, padx=5, pady=5)
        playback_frame = ttk.LabelFrame(control_frame, text="Playback Controls")
        playback_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(playback_frame, text="Play/Pause", command=self.toggle_playback).pack(fill=tk.X, padx=5, pady=5)

        frame_nav_frame = ttk.Frame(playback_frame)
        frame_nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_nav_frame, text="< Prev Frame", command=lambda: self.change_frame(-1)).pack(side=tk.LEFT,
                                                                                                     padx=5)
        self.frame_label = ttk.Label(frame_nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_nav_frame, text="Next Frame >", command=lambda: self.change_frame(1)).pack(side=tk.RIGHT,
                                                                                                    padx=5)

        self.shot_info = ttk.Label(control_frame, text="No shot loaded")
        self.shot_info.pack(fill=tk.X, padx=5, pady=10)

        display_frame = ttk.LabelFrame(main_frame, text="Video Display")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_data_directory(self):
        self.data_dir = filedialog.askdirectory(title="Select Data Directory")

        if not self.data_dir:
            return

        self.shot_vids_dir = os.path.join(self.data_dir, "shot_vids")
        self.shot_timestamps_dir = os.path.join(self.data_dir, "shot_timestamps")
        self.video_files = [f for f in os.listdir(self.shot_vids_dir) if f.endswith('.MOV')]
        self.video_files.sort()

        self.current_video_idx = 0
        self.load_current_video()

    def load_current_video(self):
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

        video_filename = self.video_files[self.current_video_idx]
        video_path = os.path.join(self.shot_vids_dir, video_filename)
        self.video_label.config(text=f"Video: {video_filename} ({self.current_video_idx + 1}/{len(self.video_files)})")

        self.video_cap = cv2.VideoCapture(video_path)

        base_name = os.path.splitext(video_filename)[0]
        csv_filename = f"{base_name}_timestamps.csv"
        csv_path = os.path.join(self.shot_timestamps_dir, csv_filename)

        self.shots_data = pd.read_csv(csv_path)
        self.current_shot_idx = 0
        self.load_current_shot()

    def load_current_shot(self):
        if self.shots_data is None or len(self.shots_data) == 0:
            self.shot_label.config(text="No shots available")
            self.shot_info.config(text="No timestamp data found")
            return
        shot_data = self.shots_data.iloc[self.current_shot_idx]
        self.shot_label.config(text=f"Shot: {self.current_shot_idx + 1}/{len(self.shots_data)}")
        start_time_sec = shot_data.get('start_time_sec', 0)
        shot_info_text = f"Start Time: {start_time_sec:.2f} seconds\n"
        for col in self.shots_data.columns:
            if col != 'start_time_sec':
                shot_info_text += f"{col}: {shot_data[col]}\n"
        self.shot_info.config(text=shot_info_text)
        self.extract_clip(start_time_sec, duration=2.5)

    def extract_clip(self, start_time_sec, duration=2.5):
        self.extracted_clip_frames = []
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time_sec * fps)
        num_frames = int(duration * fps)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(num_frames):
            ret, frame = self.video_cap.read()
            if not ret:
                break
            processed_frame = self.process_frame_with_mediapipe(frame)
            self.extracted_clip_frames.append(processed_frame)
        self.current_frame_idx = 0
        self.processing_paused = True
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")
        self.display_current_frame()

    def process_frame_with_mediapipe(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        annotated_frame = frame.copy()
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
        frame = self.extracted_clip_frames[self.current_frame_idx]

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600

        h, w = frame.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_frame = cv2.resize(frame, (new_w, new_h))

        image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(new_w // 2, new_h // 2, image=photo)
        self.canvas.image = photo

    def change_video(self, delta):
        if not self.video_files:
            return

        new_idx = self.current_video_idx + delta
        if 0 <= new_idx < len(self.video_files):
            self.current_video_idx = new_idx
            self.load_current_video()

    def change_shot(self, delta):
        new_idx = self.current_shot_idx + delta
        if 0 <= new_idx < len(self.shots_data):
            self.current_shot_idx = new_idx
            self.load_current_shot()

    def change_frame(self, delta):
        new_idx = self.current_frame_idx + delta
        if 0 <= new_idx < len(self.extracted_clip_frames):
            self.current_frame_idx = new_idx
            self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")
            self.display_current_frame()

    def toggle_playback(self):
        self.processing_paused = not self.processing_paused
        if not self.processing_paused:
            self.play_clip()

    def play_clip(self):
        if self.processing_paused or not self.extracted_clip_frames:
            return

        self.display_current_frame()
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.extracted_clip_frames):
            self.current_frame_idx = 0
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{len(self.extracted_clip_frames)}")
        self.root.after(33, self.play_clip)

def main():
    root = tk.Tk()
    app = BasketballShotAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()