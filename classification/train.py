import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binary_classifications import get_binary_classifications

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update paths to use absolute paths
basket_vids_dir = os.path.join(PROJECT_ROOT, "data", "basket_vids")
markers_dir = os.path.join(PROJECT_ROOT, "data", "markers")
shot_timestamps_dir = os.path.join(PROJECT_ROOT, "data", "shot_timestamps")
mediapipe_dir = os.path.join(PROJECT_ROOT, "data", "mediapipe_data")

# Add memory management for MPS
if torch.backends.mps.is_available():
    # Set memory management parameters
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'  # Enable garbage collection

class PoseDataset(Dataset):
    def __init__(self, mediapipe_dir, labels):
        self.mediapipe_dir = mediapipe_dir
        self.labels = labels
        # Get all mediapipe files and sort them by shot number
        self.data_files = sorted([f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
                               key=lambda x: int(x.split('_')[1]))  # Sort by shot number
        
        # Define key body parts for shooting (MediaPipe indices)
        self.shooting_keypoints = {
            'right_shoulder': 12,
            'right_elbow': 14,
            'right_wrist': 16,
            'right_hip': 24,
            'right_knee': 26,
            'right_ankle': 28
        }
        
        # Calculate global statistics for normalization
        self.feature_means = None
        self.feature_stds = None
        self._calculate_global_stats()
        
    def _calculate_global_stats(self):
        """Calculate mean and std for each feature across all shots"""
        all_features = []
        
        # First pass: collect all features
        for idx in range(len(self.data_files)):
            with open(os.path.join(self.mediapipe_dir, self.data_files[idx]), 'r') as f:
                data = json.load(f)
            
            frames = data['frames']
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
                
                # Calculate features
                features = self._calculate_frame_features(keypoints)
                all_features.append(features)
        
        # Calculate global statistics
        all_features = np.array(all_features)
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
        print("\nFeature Statistics:")
        print("Means:", self.feature_means)
        print("Stds:", self.feature_stds)
    
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
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load MediaPipe data
        with open(os.path.join(self.mediapipe_dir, self.data_files[idx]), 'r') as f:
            data = json.load(f)
        
        # Extract pose keypoints
        frames = data['frames']
        num_frames = len(frames)
        keypoints = np.zeros((num_frames, 33, 3))  # 33 keypoints, 3 values (x,y,z)
        
        for i, frame in enumerate(frames):
            landmarks = frame.get('landmarks', [])
            if landmarks is None:
                continue
                
            for j, landmark in enumerate(landmarks):
                if j >= 33:  # Skip if we have more landmarks than expected
                    break
                if landmark is None:
                    continue
                keypoints[i, j] = [
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0)
                ]
        
        # Extract shooting-specific features for each frame
        features = []
        for i in range(num_frames):
            frame_features = self._calculate_frame_features(keypoints[i])
            features.append(frame_features)
            
        
        # Convert to numpy array and normalize using global statistics
        features = np.array(features)
        features = (features - self.feature_means) / self.feature_stds
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return features, label

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=3, num_heads=8, dropout=0.2):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 44, hidden_dim))  # 44 frames
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, num_features)
        batch_size, num_frames, num_features = x.shape
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        pos_encoding = self.pos_encoder.repeat(batch_size, 1, 1)
        x = x + pos_encoding
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def calculate_metrics(y_true, y_pred, y_prob):
    # Convert to numpy arrays, detaching from computation graph
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_prob = y_prob.detach().cpu().numpy()
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Calculate balanced accuracy
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    }

def plot_metrics(metrics_history, save_dir='metrics_plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # Plot F1 scores
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['train_f1'], label='Train F1')
    plt.plot(metrics_history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curves.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    best_val_acc = 0.0
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_bal_acc': [], 'val_bal_acc': []
    }
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_probs = []
        train_labels = []
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, position=1)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Step the scheduler for each batch
            
            train_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Move tensors to CPU and detach from computation graph
            train_preds.extend(predicted.detach().cpu())
            train_probs.extend(probs.detach().cpu())
            train_labels.extend(labels.cpu())
            
            # Update training progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Calculate training metrics
        train_metrics = calculate_metrics(
            torch.tensor(train_labels),
            torch.tensor(train_preds),
            torch.stack(train_probs)
        )
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_probs = []
        val_labels = []
        
        # Create progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False, position=1)
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Move tensors to CPU and detach from computation graph
                val_preds.extend(predicted.detach().cpu())
                val_probs.extend(probs.detach().cpu())
                val_labels.extend(labels.cpu())
                
                # Debug: Print predictions and probabilities for first epoch
                if epoch == 0 and len(val_preds) <= 20:
                    print(f"\nValidation Sample {len(val_preds)}:")
                    print(f"True label: {labels[0].item()}")
                    print(f"Predicted: {predicted[0].item()}")
                    print(f"Probabilities: {probs[0].tolist()}")
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(
            torch.tensor(val_labels),
            torch.tensor(val_preds),
            torch.stack(val_probs)
        )
        
        # Update metrics history
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['train_f1'].append(train_metrics['f1_score'])
        metrics_history['val_f1'].append(val_metrics['f1_score'])
        metrics_history['train_bal_acc'].append(train_metrics['balanced_accuracy'])
        metrics_history['val_bal_acc'].append(val_metrics['balanced_accuracy'])
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'val_bal_acc': f'{val_metrics["balanced_accuracy"]:.2f}'
        })
        
        # Print detailed metrics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Train F1: {train_metrics["f1_score"]:.4f}')
        print(f'Train Balanced Acc: {train_metrics["balanced_accuracy"]:.4f}')
        print(f'Train Confusion Matrix:\n{train_metrics["confusion_matrix"]}')
        print(f'\nVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val F1: {val_metrics["f1_score"]:.4f}')
        print(f'Val Balanced Acc: {val_metrics["balanced_accuracy"]:.4f}')
        print(f'Val Confusion Matrix:\n{val_metrics["confusion_matrix"]}')
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_metrics['f1_score'],
            'val_f1': val_metrics['f1_score'],
            'train_bal_acc': train_metrics['balanced_accuracy'],
            'val_bal_acc': val_metrics['balanced_accuracy']
        }, index=[0])
        
        metrics_file = 'training_metrics.csv'
        if epoch == 0:
            metrics_df.to_csv(metrics_file, index=False)
        else:
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
        
        # Save best model based on balanced accuracy
        if val_metrics['balanced_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['balanced_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bal_acc': val_metrics['balanced_accuracy'],
                'val_metrics': val_metrics
            }, 'best_model.pth')
    
    # Plot and save metrics
    plot_metrics(metrics_history)

def setup_gpu():
    if torch.cuda.is_available():
        # NVIDIA GPU available
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} CUDA GPU(s)")
        for i in range(n_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        return device
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU available
        print("Found Apple Silicon GPU (MPS backend)")
        device = torch.device("mps")
        # Clear any existing memory
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return device
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu")

def main():
    # Set up GPU
    device = setup_gpu()
    print(f'Using device: {device}')
    
    # Load data
    video_name = "1.MP4"
    labels = get_binary_classifications(video_name)
    
    # Print data statistics
    print(f"Number of shots: {len(labels)}")
    print(f"Number of made shots (0): {labels.count(0)}")
    print(f"Number of missed shots (1): {labels.count(1)}")
    
    # Create dataset
    dataset = PoseDataset(mediapipe_dir, labels)
    
    # Print dataset statistics
    print(f"Number of MediaPipe files: {len(dataset)}")
    if len(dataset) != len(labels):
        print("WARNING: Number of MediaPipe files does not match number of labels!")
        print("This might indicate a mismatch between the data and labels.")
        return
    
    # Calculate class weights based on class imbalance
    labels_tensor = torch.tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    total_samples = len(labels_tensor)
    # Use square root of inverse frequency for more balanced weights
    class_weights = torch.sqrt(total_samples / (2 * class_counts))
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Create stratified train/val split
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels,
        #80
    )
    
    # Debug: Print label distributions
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    print("\nLabel Distributions:")
    print("Train class distribution:", np.bincount(train_labels))
    print("Val class distribution:", np.bincount(val_labels))
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Smaller batch sizes for more iterations
    train_batch_size = 8 if device.type == 'mps' else 16
    val_batch_size = 4 if device.type == 'mps' else 8
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Print iterations per epoch
    print(f"\nIterations per epoch:")
    print(f"Training: {len(train_loader)}")
    print(f"Validation: {len(val_loader)}")
    
    # Create model and move to GPU
    model = PoseTransformer().to(device)
    
    # Use weighted cross-entropy with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Use AdamW optimizer with moderate weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,  # Moderate learning rate
        weight_decay=0.01  # Moderate weight decay
    )
    
    # Add learning rate scheduler with longer warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,  # Moderate max learning rate
        epochs=40,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=50
    )
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device=device)

if __name__ == '__main__':
    main()