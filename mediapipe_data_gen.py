import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import json
import argparse
from tqdm import tqdm
from pathlib import Path


class BasketballShotDataExtractor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.shot_vids_dir = os.path.join(data_dir, "shot_vids")
        self.shot_timestamps_dir = os.path.join(data_dir, "shot_timestamps")
        self.output_dir = os.path.join(data_dir, "mediapipe_data")
        os.makedirs(self.output_dir, exist_ok=True)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


    def get_video_files(self):
        video_files = [f for f in os.listdir(self.shot_vids_dir) if f.endswith('.MOV')]
        video_files.sort()
        return video_files

    def load_timestamp_data(self, video_filename):
        base_name = os.path.splitext(video_filename)[0]
        csv_filename = f"{base_name}_timestamps.csv"
        csv_path = os.path.join(self.shot_timestamps_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"Warning: No timestamp file found for {video_filename}")
            return None

        return pd.read_csv(csv_path)

    def extract_shot_landmarks(self, video_path, start_time_sec, duration=1.5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time_sec * fps)
        end_frame = min(start_frame + int(duration * fps), total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        landmarks_data = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            frame_data = {
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "landmarks": None
            }

            if results.pose_landmarks:
                landmarks_list = []
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks_list.append({
                        "idx": i,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                frame_data["landmarks"] = landmarks_list

            landmarks_data.append(frame_data)
            frame_idx += 1

        cap.release()
        metadata = {
            "start_time_sec": start_time_sec,
            "end_time_sec": start_time_sec + duration,
            "duration_sec": duration,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": len(landmarks_data)
        }
        return {
            "metadata": metadata,
            "frames": landmarks_data
        }

    def process_all_videos(self):
        self.validate_directories()
        video_files = self.get_video_files()

        for video_idx, video_filename in enumerate(video_files):
            shots_data = self.load_timestamp_data(video_filename)
            video_path = os.path.join(self.shot_vids_dir, video_filename)
            video_base_name = os.path.splitext(video_filename)[0]

            for shot_idx, shot_data in tqdm(shots_data.iterrows(), total=len(shots_data), desc="Processing shots"):
                start_time_sec = shot_data.get('start_time_sec', 0)
                output_filename = f"{video_base_name}_{shot_idx + 1}_mediapipe.json"
                output_path = os.path.join(self.output_dir, output_filename)
                landmarks_data = self.extract_shot_landmarks(
                    video_path,
                    start_time_sec,
                    duration=1.5
                )
                if landmarks_data:
                    landmarks_data["shot_metadata"] = shot_data.to_dict()
                    with open(output_path, 'w') as f:
                        json.dump(landmarks_data, f, indent=2)



def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe pose data from basketball shot videos')
    parser.add_argument('--data_dir', type=str,
                        help='Path to data directory containing shot_vids and shot_timestamps folders')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        data_dir = input("Enter path to data directory: ")

    extractor = BasketballShotDataExtractor(data_dir)
    extractor.process_all_videos()


if __name__ == "__main__":
    main()