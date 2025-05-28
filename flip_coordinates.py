import json
import os
from pathlib import Path

def flip_coordinates(data, axis):
    """Flip coordinates along the specified axis."""
    for frame in data['frames']:
        if frame['landmarks'] is None:
            continue
        for landmark in frame['landmarks']:
            if axis == 'x':
                landmark['x'] = abs(landmark['x'] - 1)
            elif axis == 'y':
                landmark['y'] = abs(landmark['y'] - 1)
            elif axis == 'z':
                landmark['z'] = -landmark['z']
    return data

def process_file(file_path, axis):
    """Process a single file by flipping coordinates along the specified axis."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Flip the coordinates
        modified_data = flip_coordinates(data, axis)
        
        # Write back to the same file
        with open(file_path, 'w') as f:
            json.dump(modified_data, f, indent=2)
            
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Directory containing the mediapipe data files
    data_dir = Path('data/mediapipe_data')
    """
    # Process files 11-17 (x-axis)
    for i in range(11, 18):
        pattern = f"{i}_*_mediapipe.json"
        for file_path in data_dir.glob(pattern):
            process_file(file_path, 'x')
    
    # Process files 18-24 (y-axis)
    for i in range(18, 25):
        pattern = f"{i}_*_mediapipe.json"
        for file_path in data_dir.glob(pattern):
            process_file(file_path, 'y')
    """
    # Process files 25-30 (z-axis)
    for i in range(25, 31):
        pattern = f"{i}_*_mediapipe.json"
        for file_path in data_dir.glob(pattern):
            process_file(file_path, 'z')

if __name__ == "__main__":
    main() 