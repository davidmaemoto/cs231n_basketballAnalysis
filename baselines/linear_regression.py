import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binary_classifications import get_binary_classifications

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update paths to use absolute paths
mediapipe_dir = os.path.join(PROJECT_ROOT, "data", "mediapipe_data")
angles_dir = os.path.join(PROJECT_ROOT, "data", "angles_1")

class LinearAngleRegressor:
    def __init__(self):
        # Define key body parts for shooting (MediaPipe indices)
        self.shooting_keypoints = {
            'right_shoulder': 12,
            'right_elbow': 14,
            'right_wrist': 16,
            'right_hip': 24,
            'right_knee': 26,
            'right_ankle': 28
        }
        
        # Initialize feature statistics
        self.feature_means = None
        self.feature_stds = None
        
    def _calculate_frame_features(self, keypoints):
        """Calculate normalized features for a single frame"""
        features = []
        
        # 1. Shooting Arm Mechanics
        right_shoulder = keypoints[self.shooting_keypoints['right_shoulder']]
        right_elbow = keypoints[self.shooting_keypoints['right_elbow']]
        right_wrist = keypoints[self.shooting_keypoints['right_wrist']]
        
        # Elbow angle (normalized to [0, 1])
        v1 = right_shoulder - right_elbow
        v2 = right_wrist - right_elbow
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            elbow_angle = np.arccos(cos_angle) / np.pi  # Normalize to [0, 1]
        else:
            elbow_angle = 0.5  # Middle value for invalid angles
        features.append(elbow_angle)
        
        # Shooting arm alignment (normalized to [0, 1])
        arm_vector = right_wrist - right_shoulder
        arm_norm = np.linalg.norm(arm_vector)
        if arm_norm > 0:
            cos_angle = np.clip(np.dot(arm_vector, np.array([0, 1, 0])) / arm_norm, -1.0, 1.0)
            arm_alignment = (np.arccos(cos_angle) / np.pi)  # Normalize to [0, 1]
        else:
            arm_alignment = 0.5
        features.append(arm_alignment)
        
        # 2. Lower Body Mechanics
        right_hip = keypoints[self.shooting_keypoints['right_hip']]
        right_knee = keypoints[self.shooting_keypoints['right_knee']]
        right_ankle = keypoints[self.shooting_keypoints['right_ankle']]
        
        # Knee bend (normalized to [0, 1])
        v1 = right_hip - right_knee
        v2 = right_ankle - right_knee
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            knee_angle = np.arccos(cos_angle) / np.pi  # Normalize to [0, 1]
        else:
            knee_angle = 0.5
        features.append(knee_angle)
        
        # 3. Shot Arc (normalized relative to body height)
        body_height = np.linalg.norm(right_shoulder - right_ankle)
        shot_arc = (right_wrist[1] - right_shoulder[1]) / (body_height + 1e-8)  # Normalize by body height
        features.append(shot_arc)
        
        return np.array(features)
    
    def _calculate_global_stats(self, all_features):
        """Calculate mean and std for each feature"""
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
        print("\nFeature Statistics:")
        print("Means:", self.feature_means)
        print("Stds:", self.feature_stds)
    
    def prepare_data(self, video_name="1.MP4"):
        """Prepare data for training"""
        # Get labels and angles
        labels = get_binary_classifications(video_name)
        angles_df = pd.read_csv(os.path.join(angles_dir, f"{os.path.splitext(video_name)[0]}_angles.csv"))
        angles = angles_df['angle'].values
        
        # Get all mediapipe files and sort them by shot number
        data_files = sorted([f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
                          key=lambda x: int(x.split('_')[1]))  # Sort by shot number
        
        # Extract features for each shot
        all_features = []
        valid_indices = []  # Keep track of which shots have valid angles
        
        for i, data_file in enumerate(data_files):
            # Skip if this is a made shot (no angle)
            if pd.isna(angles[i]):
                continue
                
            with open(os.path.join(mediapipe_dir, data_file), 'r') as f:
                data = json.load(f)
            
            frames = data['frames']
            frame_features = []
            
            for frame in frames:
                landmarks = frame.get('landmarks', [])
                if not landmarks:
                    continue
                
                # Extract keypoints
                keypoints = np.zeros((33, 3))
                for j, landmark in enumerate(landmarks):
                    if j >= 33 or landmark is None:
                        continue
                    keypoints[j] = [
                        landmark.get('x', 0.0),
                        landmark.get('y', 0.0),
                        landmark.get('z', 0.0)
                    ]
                
                # Calculate features for this frame
                features = self._calculate_frame_features(keypoints)
                frame_features.append(features)
            
            # Average features across frames for this shot
            if frame_features:
                shot_features = np.mean(frame_features, axis=0)
                all_features.append(shot_features)
                valid_indices.append(i)
        
        # Convert to numpy array
        X = np.array(all_features)
        y = angles[valid_indices]  # Only use angles for missed shots
        
        # Calculate and apply normalization
        self._calculate_global_stats(X)
        X = (X - self.feature_means) / self.feature_stds
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate the model"""
        # First split: 90% train+val, 10% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.1, #random_state=80
        )
        
        # Second split: 70% train, 20% val (from the remaining 90%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.222,  # 0.222 * 0.9 ≈ 0.2 (20% of total)
            random_state=42
        )
        
        # Print split sizes
        print("\nDataset split sizes:")
        print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions for all sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for all sets
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        val_metrics = {
            'mse': mean_squared_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'mae': mean_absolute_error(y_val, y_val_pred),
            'r2': r2_score(y_val, y_val_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
            'feature_importance': dict(zip(['elbow_angle', 'arm_alignment', 'knee_angle', 'shot_arc'],
                                        np.abs(model.coef_)))
        }
        
        # Print metrics
        print("\nTraining Metrics:")
        print(f"Mean Squared Error: {train_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {train_metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: {train_metrics['mae']:.4f}")
        print(f"R² Score: {train_metrics['r2']:.4f}")
        
        print("\nValidation Metrics:")
        print(f"Mean Squared Error: {val_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {val_metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: {val_metrics['mae']:.4f}")
        print(f"R² Score: {val_metrics['r2']:.4f}")
        
        print("\nTest Metrics:")
        print(f"Mean Squared Error: {test_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {test_metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: {test_metrics['mae']:.4f}")
        print(f"R² Score: {test_metrics['r2']:.4f}")
        print("\nFeature Importance:")
        for feature, importance in sorted(test_metrics['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'set': ['train', 'val', 'test'],
            'mse': [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']],
            'rmse': [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
            'mae': [train_metrics['mae'], val_metrics['mae'], test_metrics['mae']],
            'r2': [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
        })
        metrics_df.to_csv('linear_regression_metrics.csv', index=False)
        
        # Plot predicted vs actual angles for all sets
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.5)
        plt.plot([0, 180], [0, 180], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Angle')
        plt.ylabel('Predicted Angle')
        plt.title('Training Set Predictions')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_val, y_val_pred, alpha=0.5)
        plt.plot([0, 180], [0, 180], 'r--')
        plt.xlabel('Actual Angle')
        plt.ylabel('Predicted Angle')
        plt.title('Validation Set Predictions')
        
        plt.tight_layout()
        plt.savefig('linear_regression_predictions.png')
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(8, 6))
        features = list(test_metrics['feature_importance'].keys())
        importance = list(test_metrics['feature_importance'].values())
        plt.bar(features, importance)
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('linear_regression_feature_importance.png')
        plt.close()
        
        return test_metrics

def main():
    # Create regressor
    regressor = LinearAngleRegressor()
    
    # Prepare data
    X, y = regressor.prepare_data()
    
    # Train and evaluate
    metrics = regressor.train_and_evaluate(X, y)

if __name__ == '__main__':
    main()
