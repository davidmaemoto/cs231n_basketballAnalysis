from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from pathlib import Path

app = Flask(__name__)

# Configuration
VIDEO_DIRECTORY = Path('./data/basket_vids')
MARKERS_DIRECTORY = Path('./data/markers')

# Ensure the markers directory exists
MARKERS_DIRECTORY.mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    """Main page with video player and controls"""
    return render_template('index.html')


@app.route('/videos')
def list_videos():
    """List all videos in the videos directory"""
    videos = []
    # Get all video files - common video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if VIDEO_DIRECTORY.exists():
        for file in VIDEO_DIRECTORY.iterdir():
            if file.suffix.lower() in video_extensions:
                videos.append(file.name)

    return jsonify(videos)


@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve a video file"""
    return send_from_directory(VIDEO_DIRECTORY, filename)


@app.route('/markers/<filename>', methods=['GET'])
def get_markers(filename):
    """Get markers for a specific video"""
    marker_file = MARKERS_DIRECTORY / f"{filename}.json"

    if marker_file.exists():
        with open(marker_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify([])


@app.route('/markers/<filename>', methods=['POST'])
def save_markers(filename):
    """Save markers for a specific video"""
    markers = request.json

    marker_file = MARKERS_DIRECTORY / f"{filename}.json"

    with open(marker_file, 'w') as f:
        json.dump(markers, f, indent=2)

    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True)
