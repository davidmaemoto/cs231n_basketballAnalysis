# Basketball Shot Analysis with Deep Learning

This project leverages pose estimation data and deep learning to analyze basketball shots. It tackles two primary tasks:
1. **Shot Success Prediction** (Binary Classification)
2. **Rebound Angle Estimation** (Regression)

The project uses MediaPipe for pose extraction and a CNN-LSTM-based neural network for joint task learning.

## Project Structure

```
project/
├── app.py                         # Main application script
├── binary_classifications.py     # Functions for binary classification metrics
├── cnn_lstm.py                   # CNN-LSTM model definition
├── flip_coordinates.py           # Utility to flip pose data
├── mediapipe_data_gen.py         # MediaPipe-based data generation
├── mediapipe_time.py             # Time alignment utilities for MediaPipe data
├── preprocessing.py              # Data preprocessing pipeline
├── train_multi_task.py           # Multi-task model training script
```

## Features

- **CNN-LSTM architecture** to model temporal and spatial aspects of basketball shots.
- **Multi-task training** to simultaneously predict shot outcomes and rebound angles.
- **MediaPipe-based pose extraction** for generating model input.
- **Preprocessing utilities** for data cleaning and augmentation.

## Getting Started

### Installation

Ensure you have Python 3.8+ and the required dependencies:

```bash
pip install -r requirements.txt
```

If not present, create a `requirements.txt` with packages like:
```text
numpy
torch
mediapipe
opencv-python
matplotlib
tqdm
```

### Running the Model

1. **Preprocess data:**
```bash
python preprocessing.py
```

2. **Train the model:**
```bash
python train_multi_task.py
```

3. **Launch application or evaluation:**
```bash
python app.py
```

## Data

- Input features are extracted using **MediaPipe** from synchronized multi-angle videos.
- Each example includes 2D/3D keypoints, annotated shot success, and rebound angle.

## Acknowledgments

- MediaPipe for landmark extraction.
- PyTorch for model training.

---

*Developed as part of Stanford's CS231N deep learning project.*
