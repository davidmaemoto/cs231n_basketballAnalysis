import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_curve, 
                           roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binary_classifications import get_binary_classifications
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mediapipe_dir = os.path.join(PROJECT_ROOT, "data", "mediapipe_data")

class LinearPoseClassifier:
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
        data_files = sorted([f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
                          key=lambda x: int(x.split('_')[1]))  # Sort by shot number
        
        all_features = []
        for data_file in data_files:
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
        
        X = np.array(all_features)
        y = np.array(labels)
        self._calculate_global_stats(X)
        X = (X - self.feature_means) / self.feature_stds
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, #random_state=80
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.222,
            stratify=y_train_val,
            random_state=42
        )
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        train_metrics = {
            'accuracy': model.score(X_train, y_train),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'precision': precision_recall_fscore_support(y_train, y_train_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_train, y_train_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_train, y_train_pred, average='binary')[2],
            'auc': roc_auc_score(y_train, y_train_prob),
            'confusion_matrix': confusion_matrix(y_train, y_train_pred)
        }
        
        val_metrics = {
            'accuracy': model.score(X_val, y_val),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
            'precision': precision_recall_fscore_support(y_val, y_val_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_val, y_val_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_val, y_val_pred, average='binary')[2],
            'auc': roc_auc_score(y_val, y_val_prob),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }
        
        test_metrics = {
            'accuracy': model.score(X_test, y_test),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'precision': precision_recall_fscore_support(y_test, y_test_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_test, y_test_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_test, y_test_pred, average='binary')[2],
            'auc': roc_auc_score(y_test, y_test_prob),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        print("\nTraining Metrics:")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f}")
        print(f"Recall: {train_metrics['recall']:.4f}")
        print(f"F1 Score: {train_metrics['f1']:.4f}")
        print(f"AUC Score: {train_metrics['auc']:.4f}")
        print("\nTraining Confusion Matrix:")
        print(train_metrics['confusion_matrix'])
        
        print("\nValidation Metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        print(f"F1 Score: {val_metrics['f1']:.4f}")
        print(f"AUC Score: {val_metrics['auc']:.4f}")
        print("\nValidation Confusion Matrix:")
        print(val_metrics['confusion_matrix'])
        
        print("\nTest Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"AUC Score: {test_metrics['auc']:.4f}")
        print("\nTest Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        metrics_df = pd.DataFrame({
            'set': ['train', 'val', 'test'],
            'accuracy': [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
            'balanced_accuracy': [train_metrics['balanced_accuracy'], val_metrics['balanced_accuracy'], test_metrics['balanced_accuracy']],
            'precision': [train_metrics['precision'], val_metrics['precision'], test_metrics['precision']],
            'recall': [train_metrics['recall'], val_metrics['recall'], test_metrics['recall']],
            'f1': [train_metrics['f1'], val_metrics['f1'], test_metrics['f1']],
            'auc': [train_metrics['auc'], val_metrics['auc'], test_metrics['auc']]
        })
        metrics_df.to_csv('linear_classifier_metrics.csv', index=False)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(train_metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Training Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Make', 'Miss'])
        plt.yticks(tick_marks, ['Make', 'Miss'])
        self._add_confusion_matrix_text(train_metrics['confusion_matrix'])
        plt.subplot(1, 2, 2)
        plt.imshow(val_metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Validation Confusion Matrix')
        plt.colorbar()
        plt.xticks(tick_marks, ['Make', 'Miss'])
        plt.yticks(tick_marks, ['Make', 'Miss'])
        self._add_confusion_matrix_text(val_metrics['confusion_matrix'])
        plt.tight_layout()
        plt.savefig('linear_classifier_confusion_matrices.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_train, y_train_prob)
        plt.plot(fpr, tpr, label=f'Train ROC (AUC = {train_metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Training ROC Curve')
        plt.legend(loc="lower right")
        
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_val, y_val_prob)
        plt.plot(fpr, tpr, label=f'Val ROC (AUC = {val_metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC Curve')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('linear_classifier_roc_curves.png')
        plt.close()
        
        return test_metrics
    
    def _add_confusion_matrix_text(self, cm):
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

def main():
    classifier = LinearPoseClassifier()
    X, y = classifier.prepare_data()
    metrics = classifier.train_and_evaluate(X, y)

if __name__ == '__main__':
    main()
