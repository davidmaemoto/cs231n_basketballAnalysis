import json
import os
import pandas as pd
import bisect
import numpy as np
import math

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Update paths to use absolute paths
basket_vids_dir = os.path.join(PROJECT_ROOT, "data", "basket_vids")
markers_dir = os.path.join(PROJECT_ROOT, "data", "markers")
shot_timestamps_dir = os.path.join(PROJECT_ROOT, "data", "shot_timestamps")
mediapipe_dir = os.path.join(PROJECT_ROOT, "data", "mediapipe_data")

def calculate_angle_3d(p1, p2, p3):
    """
    Calculate the angle between three points in 3D space.
    p1, p2, p3 are dictionaries with x, y, z coordinates
    Returns angle in degrees
    """
    # Convert points to numpy arrays
    v1 = np.array([p1['x'], p1['y'], p1['z']])
    v2 = np.array([p2['x'], p2['y'], p2['z']])
    v3 = np.array([p3['x'], p3['y'], p3['z']])
    
    # Calculate vectors
    v1 = v1 - v2
    v3 = v3 - v2
    
    # Calculate angle
    cos_angle = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in valid range
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def get_binary_classifications(video_name="1.MP4"):
    with open(os.path.join(markers_dir, f"{video_name}.json"), "r") as file:
        data = json.load(file)
    time_list = [float(entry["time"]) for entry in data if "time" in entry]
    shot_starts = list(pd.read_csv(os.path.join(shot_timestamps_dir, f"{os.path.splitext(video_name)[0]}_timestamps.csv"))["start_time_sec"])

    out = []
    for shot_start in shot_starts:
        i = bisect.bisect_left(time_list, shot_start)
        if i < len(time_list) and time_list[i] < shot_start + 3:
            out.append(0)
        else:
            out.append(1)

    return out
"""
print(get_binary_classifications())
print(len(get_binary_classifications()))
print(get_binary_classifications().count(0))
print(get_binary_classifications().count(1))
"""
import random

def get_linear_angle(landmarks):
    """
    Create a linear combination of body part coordinates to represent rebound angle.
    Uses right shoulder, elbow, wrist, hip, knee, and ankle coordinates.
    """
    try:
        # Extract coordinates for each body part
        shoulder = landmarks[12]  # Right shoulder
        elbow = landmarks[14]     # Right elbow
        wrist = landmarks[16]     # Right wrist
        hip = landmarks[24]       # Right hip
        knee = landmarks[26]      # Right knee
        ankle = landmarks[28]     # Right ankle
        
        # Check if any landmark is None
        if any(landmark is None for landmark in [shoulder, elbow, wrist, hip, knee, ankle]):
            return None
            
        # Create feature vector
        features = [
            shoulder['x'], shoulder['y'], shoulder['z'],
            elbow['x'], elbow['y'], elbow['z'],
            wrist['x'], wrist['y'], wrist['z'],
            hip['x'], hip['y'], hip['z'],
            knee['x'], knee['y'], knee['z'],
            ankle['x'], ankle['y'], ankle['z']
        ]
        
        # Convert to numpy array
        features = np.array(features)
        
        # Normalize features to 0-1 range
        features = (features - np.min(features)) / (np.max(features) - np.min(features))
        
        # Create linear combination (you can adjust these weights)
        weights = np.ones(len(features)) / len(features)  # Equal weights for now
        angle = np.dot(features, weights)
        
        # Scale to 0-180 range
        angle = angle * 180
        
        return angle
    except (IndexError, KeyError, TypeError, ValueError):
        return None

def get_angles(video_name="1.MP4"):
    """
    Calculate rebound angle as a linear combination of body part coordinates.
    Uses MediaPipe pose data to extract coordinates of right side body parts.
    Each mediapipe file corresponds to one shot.
    """
    # Create angles directory if it doesn't exist
    angles_dir = os.path.join(PROJECT_ROOT, "data", "angles")
    os.makedirs(angles_dir, exist_ok=True)

    # Get binary classifications (0=make, 1=miss)
    shots = get_binary_classifications(video_name)
    
    # Get all mediapipe files for this video
    video_base = os.path.splitext(video_name)[0]
    mediapipe_files = [f for f in os.listdir(mediapipe_dir) if f.startswith(f"{video_base}_")]
    
    # Sort files by frame number (descending to match shot order)
    mediapipe_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    
    angles = []
    for i, shot in enumerate(shots):
        if shot == 0:  # Make
            angles.append(None)
        else:  # Miss
            if i < len(mediapipe_files):
                try:
                    with open(os.path.join(mediapipe_dir, mediapipe_files[i]), 'r') as f:
                        data = json.load(f)
                        if data['frames']:
                            # Use the first frame of the shot
                            frame = data['frames'][0]
                            landmarks = frame['landmarks']
                            
                            # Calculate linear angle
                            angle = get_linear_angle(landmarks)
                            angles.append(angle)
                        else:
                            angles.append(None)
                except (json.JSONDecodeError, KeyError, IndexError):
                    angles.append(None)
            else:
                angles.append(None)
    
    # Save to CSV
    output_file = os.path.join(angles_dir, f"{video_base}_angles.csv")
    df = pd.DataFrame({"angle": angles})
    df.to_csv(output_file, index=False)
    
    return df

#get_angles()

