import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binary_classifications import get_binary_classifications
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mediapipe_dir = os.path.join(PROJECT_ROOT, "data", "mediapipe_data")
angles_dir = os.path.join(PROJECT_ROOT, "data", "angles_1")

class LinearAngleRegressor:
    def __init__(self):
        self.shooting_keypoints = {
            'right_shoulder': 12,
            'right_elbow': 14,
            'right_wrist': 16,
            'right_hip': 24,
            'right_knee': 26,
            'right_ankle': 28
        }
        self.feature_means = None
        self.feature_stds = None
        
    def _calculate_frame_features(self, keypoints):
        features = []
        right_shoulder = keypoints[self.shooting_keypoints['right_shoulder']]
        right_elbow = keypoints[self.shooting_keypoints['right_elbow']]
        right_wrist = keypoints[self.shooting_keypoints['right_wrist']]
        v1 = right_shoulder - right_elbow
        v2 = right_wrist - right_elbow
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            elbow_angle = np.arccos(cos_angle) / np.pi
        else:
            elbow_angle = 0.5
        features.append(elbow_angle)
        arm_vector = right_wrist - right_shoulder
        arm_norm = np.linalg.norm(arm_vector)
        if arm_norm > 0:
            cos_angle = np.clip(np.dot(arm_vector, np.array([0, 1, 0])) / arm_norm, -1.0, 1.0)
            arm_alignment = (np.arccos(cos_angle) / np.pi)
        else:
            arm_alignment = 0.5
        features.append(arm_alignment)

        right_hip = keypoints[self.shooting_keypoints['right_hip']]
        right_knee = keypoints[self.shooting_keypoints['right_knee']]
        right_ankle = keypoints[self.shooting_keypoints['right_ankle']]
        
        v1 = right_hip - right_knee
        v2 = right_ankle - right_knee
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            knee_angle = np.arccos(cos_angle) / np.pi
        else:
            knee_angle = 0.5
        features.append(knee_angle)

        body_height = np.linalg.norm(right_shoulder - right_ankle)
        shot_arc = (right_wrist[1] - right_shoulder[1]) / (body_height + 1e-8)
        features.append(shot_arc)
        
        return np.array(features)
    
    def _calculate_global_stats(self, all_features):
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0) + 1e-8
        print("Means:", self.feature_means)
        print("Stds:", self.feature_stds)
    
    def prepare_data(self, video_name="1.MP4"):
        labels = get_binary_classifications(video_name)
        angles_df = pd.read_csv(os.path.join(angles_dir, f"{os.path.splitext(video_name)[0]}_angles.csv"))
        angles = angles_df['angle'].values
        
        data_files = sorted([f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
                          key=lambda x: int(x.split('_')[1]))
        
        all_features = []
        valid_indices = []
        for i, data_file in enumerate(data_files):
            if pd.isna(angles[i]):
                continue
                
            with open(os.path.join(mediapipe_dir, data_file), 'r') as f:
                data = json.load(f)
            
            frames = data['frames']
            frame_features = []
            
            for frame in frames:
                landmarks = frame.get('landmarks', [])
                keypoints = np.zeros((33, 3))
                for j, landmark in enumerate(landmarks):
                    if j >= 33 or landmark is None:
                        continue
                    keypoints[j] = [
                        landmark.get('x', 0.0),
                        landmark.get('y', 0.0),
                        landmark.get('z', 0.0)
                    ]
                
                features = self._calculate_frame_features(keypoints)
                frame_features.append(features)
            
            if frame_features:
                shot_features = np.mean(frame_features, axis=0)
                all_features.append(shot_features)
                valid_indices.append(i)
        
        X = np.array(all_features)
        y = angles[valid_indices]
        self._calculate_global_stats(X)
        X = (X - self.feature_means) / self.feature_stds
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.1,
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.222,
            random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
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
        
        metrics_df = pd.DataFrame({
            'set': ['train', 'val', 'test'],
            'mse': [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']],
            'rmse': [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
            'mae': [train_metrics['mae'], val_metrics['mae'], test_metrics['mae']],
            'r2': [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
        })
        metrics_df.to_csv('linear_regression_metrics.csv', index=False)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.5)
        plt.plot([0, 180], [0, 180], 'r--')
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
    regressor = LinearAngleRegressor()
    X, y = regressor.prepare_data()
    metrics = regressor.train_and_evaluate(X, y)

if __name__ == '__main__':
    main()
