import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultiTaskCNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,  # 6 keypoints × 3 coordinates + 4 derived features
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lambda_cls: float = 0.65  # Balanced weight between classification and regression
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        
        # CNN for spatial feature extraction with batch norm and dropout
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head with batch norm
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 classes for binary classification
        )
        
        # Regression head with batch norm
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # CNN processing
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        
        # Classification output (no sigmoid, using cross entropy)
        cls_out = self.cls_head(last_hidden)
        
        # Regression output
        reg_out = self.reg_head(last_hidden)
        
        return cls_out, reg_out
    
    def compute_loss(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the multi-task loss combining classification and regression losses.
        
        Args:
            cls_pred: Classification predictions (batch_size, 2)
            reg_pred: Regression predictions (batch_size, 1)
            reg_target: Regression targets (batch_size, 1)
            class_weights: Optional weights for classification classes
            
        Returns:
            Tuple of (total_loss, cls_loss, reg_loss)
        """
        # Ensure cls_target is the right shape (batch_size,)
        cls_target = cls_target.view(-1)
        
        # Cross entropy with label smoothing for classification
        cls_loss = F.cross_entropy(
            cls_pred, 
            cls_target.long(),
            weight=class_weights,
            label_smoothing=0.1
        )
        
        # MSE for regression (only for missed shots)
        mask = (reg_target != -1).float()  # Create mask for valid regression targets
        # Ensure predictions are in [0, 180] range
        reg_pred = torch.clamp(reg_pred, 0, 180)
        # Calculate MSE and normalize by 180
        reg_loss = F.mse_loss(reg_pred * mask, reg_target * mask) / 180.0
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=cls_pred.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        # Combined loss with balanced weights and L2 regularization
        total_loss = self.lambda_cls * cls_loss + (1 - self.lambda_cls) * reg_loss + l2_lambda * l2_reg
        
        return total_loss, cls_loss, reg_loss

class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,  # 6 keypoints × 3 coordinates + 4 derived features
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        lambda_cls: float = 0.7
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 44, hidden_dim))  # 44 frames max
        
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
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        pos_encoding = self.pos_encoder.repeat(batch_size, 1, 1)
        x = x + pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification output
        cls_out = torch.sigmoid(self.cls_head(x))
        
        # Regression output
        reg_out = self.reg_head(x)
        
        return cls_out, reg_out
    
    def compute_loss(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the multi-task loss combining classification and regression losses.
        
        Args:
            cls_pred: Classification predictions (batch_size, 1)
            reg_pred: Regression predictions (batch_size, 1)
            cls_target: Classification targets (batch_size, 1)
            reg_target: Regression targets (batch_size, 1)
            
        Returns:
            Tuple of (total_loss, cls_loss, reg_loss)
        """
        # Binary cross entropy for classification
        cls_loss = F.binary_cross_entropy(cls_pred, cls_target)
        
        # MSE for regression
        reg_loss = F.mse_loss(reg_pred, reg_target)
        
        # Combined loss
        total_loss = self.lambda_cls * cls_loss + (1 - self.lambda_cls) * reg_loss
        
        return total_loss, cls_loss, reg_loss

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> dict:
    """
    Train the multi-task model.
    
    Args:
        model: The model to train (either MultiTaskCNNLSTM or MultiTaskTransformer)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to train on
        class_weights: Optional weights for classification classes
        
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_reg_loss': [], 'val_reg_loss': [],
        'train_cls_acc': [], 'val_cls_acc': [],
        'train_reg_rmse': [], 'val_reg_rmse': [],
        'train_reg_mae': [], 'val_reg_mae': []
    }
    
    # Learning rate scheduler with OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=50
    )
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_cls_losses = []
        train_reg_losses = []
        train_cls_correct = 0
        train_cls_total = 0
        train_reg_errors = []
        train_reg_abs_errors = []
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            inputs, cls_targets, reg_targets = batch
            inputs = inputs.to(device)
            cls_targets = cls_targets.to(device).view(-1)  # Ensure correct shape
            reg_targets = reg_targets.to(device)
            
            # Add random noise to inputs for regularization
            if model.training:
                noise = torch.randn_like(inputs) * 0.01
                inputs = inputs + noise
            
            optimizer.zero_grad()
            cls_preds, reg_preds = model(inputs)
            
            # Ensure predictions are in [0, 180] range
            reg_preds = torch.clamp(reg_preds, 0, 180)
            
            loss, cls_loss, reg_loss = model.compute_loss(
                cls_preds, reg_preds, cls_targets, reg_targets, class_weights
            )
            
            # Skip batch if loss is NaN
            if torch.isnan(loss):
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            train_reg_losses.append(reg_loss.item())
            
            # Classification accuracy
            _, predicted = cls_preds.max(1)
            train_cls_correct += predicted.eq(cls_targets).sum().item()
            train_cls_total += cls_targets.size(0)
            
            # Regression metrics (only for missed shots)
            mask = (reg_targets != -1).float()
            if mask.sum() > 0:  # Only calculate if there are missed shots
                # RMSE
                reg_errors = ((reg_preds - reg_targets) * mask).pow(2).sqrt().detach().cpu().numpy()
                train_reg_errors.extend(reg_errors[mask.cpu().numpy().astype(bool)])
                
                # MAE
                abs_errors = ((reg_preds - reg_targets) * mask).abs().detach().cpu().numpy()
                train_reg_abs_errors.extend(abs_errors[mask.cpu().numpy().astype(bool)])
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls_acc': f'{train_cls_correct/train_cls_total:.4f}'
            })
        
        # Validation
        model.eval()
        val_losses = []
        val_cls_losses = []
        val_reg_losses = []
        val_cls_correct = 0
        val_cls_total = 0
        val_reg_errors = []
        val_reg_abs_errors = []
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                inputs, cls_targets, reg_targets = batch
                inputs = inputs.to(device)
                cls_targets = cls_targets.to(device).view(-1)  # Ensure correct shape
                reg_targets = reg_targets.to(device)
                
                cls_preds, reg_preds = model(inputs)
                
                # Ensure predictions are in [0, 180] range
                reg_preds = torch.clamp(reg_preds, 0, 180)
                
                loss, cls_loss, reg_loss = model.compute_loss(
                    cls_preds, reg_preds, cls_targets, reg_targets, class_weights
                )
                
                val_losses.append(loss.item())
                val_cls_losses.append(cls_loss.item())
                val_reg_losses.append(reg_loss.item())
                
                # Classification accuracy
                _, predicted = cls_preds.max(1)
                val_cls_correct += predicted.eq(cls_targets).sum().item()
                val_cls_total += cls_targets.size(0)
                
                # Regression metrics (only for missed shots)
                mask = (reg_targets != -1).float()
                if mask.sum() > 0:  # Only calculate if there are missed shots
                    # RMSE
                    reg_errors = ((reg_preds - reg_targets) * mask).pow(2).sqrt().cpu().numpy()
                    val_reg_errors.extend(reg_errors[mask.cpu().numpy().astype(bool)])
                    
                    # MAE
                    abs_errors = ((reg_preds - reg_targets) * mask).abs().cpu().numpy()
                    val_reg_abs_errors.extend(abs_errors[mask.cpu().numpy().astype(bool)])
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_acc': f'{val_cls_correct/val_cls_total:.4f}'
                })
        
        # Compute metrics
        train_loss = np.mean(train_losses)
        train_cls_loss = np.mean(train_cls_losses)
        train_reg_loss = np.mean(train_reg_losses)
        train_cls_acc = train_cls_correct / train_cls_total
        train_reg_rmse = np.sqrt(np.mean(train_reg_errors)) if train_reg_errors else 0.0
        train_reg_mae = np.mean(train_reg_abs_errors) if train_reg_abs_errors else 0.0
        
        val_loss = np.mean(val_losses)
        val_cls_loss = np.mean(val_cls_losses)
        val_reg_loss = np.mean(val_reg_losses)
        val_cls_acc = val_cls_correct / val_cls_total
        val_reg_rmse = np.sqrt(np.mean(val_reg_errors)) if val_reg_errors else 0.0
        val_reg_mae = np.mean(val_reg_abs_errors) if val_reg_abs_errors else 0.0
        
        # Update history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_cls_loss'].append(float(train_cls_loss))
        history['val_cls_loss'].append(float(val_cls_loss))
        history['train_reg_loss'].append(float(train_reg_loss))
        history['val_reg_loss'].append(float(val_reg_loss))
        history['train_cls_acc'].append(float(train_cls_acc))
        history['val_cls_acc'].append(float(val_cls_acc))
        history['train_reg_rmse'].append(float(train_reg_rmse))
        history['val_reg_rmse'].append(float(val_reg_rmse))
        history['train_reg_mae'].append(float(train_reg_mae))
        history['val_reg_mae'].append(float(val_reg_mae))
        
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Cls Acc: {train_cls_acc:.4f}, Val Cls Acc: {val_cls_acc:.4f}')
        print(f'Train Reg RMSE: {train_reg_rmse:.4f}°, Val Reg RMSE: {val_reg_rmse:.4f}°')
        print(f'Train Reg MAE: {train_reg_mae:.4f}°, Val Reg MAE: {val_reg_mae:.4f}°')
        
        # Save model after each epoch
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    
    return history

class MultiTaskPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mediapipe_dir: str,
        shot_success_labels: list,
        rebound_angles: list,
        max_frames: int = 44,
        normalize: bool = True
    ):
        """
        Dataset for multi-task learning of shot success prediction and rebound angle estimation.
        
        Args:
            mediapipe_dir: Directory containing MediaPipe pose data
            shot_success_labels: List of binary labels (0 for made, 1 for missed)
            rebound_angles: List of rebound angles in degrees
            max_frames: Maximum number of frames to use per shot
            normalize: Whether to normalize the pose features
        """
        self.mediapipe_dir = mediapipe_dir
        self.shot_success_labels = shot_success_labels
        self.rebound_angles = rebound_angles
        self.max_frames = max_frames
        
        # Get all mediapipe files and sort them by shot number
        self.data_files = sorted(
            [f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
            key=lambda x: int(x.split('_')[1])
        )
        
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
        if normalize:
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
                
                # Calculate features without normalization
                features = self._calculate_frame_features(keypoints, normalize=False)
                all_features.append(features)
        
        # Calculate global statistics
        all_features = np.array(all_features)
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    
    def _calculate_frame_features(self, keypoints, normalize=True):
        """Calculate features for a single frame"""
        features = []
        
        # Add only shooting keypoint coordinates (6 keypoints × 3 coordinates)
        for i, (keypoint_name, idx) in enumerate(self.shooting_keypoints.items()):
            # Get coordinates for this keypoint
            coords = keypoints[idx]
            if normalize and self.feature_means is not None:
                # Normalize each coordinate
                for j in range(3):
                    # Calculate the correct index in feature_means/feature_stds
                    # Each keypoint has 3 coordinates, so multiply by 3
                    coord_idx = i * 3 + j
                    normalized_coord = (coords[j] - self.feature_means[coord_idx]) / self.feature_stds[coord_idx]
                    features.append(normalized_coord)
            else:
                # Add raw coordinates
                features.extend(coords)
        
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
        num_frames = min(len(frames), self.max_frames)
        features = []
        
        for i in range(num_frames):
            frame = frames[i]
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
            frame_features = self._calculate_frame_features(keypoints)
            features.append(frame_features)
        
        # Pad or truncate to max_frames
        if len(features) < self.max_frames:
            # Pad with zeros
            padding = [np.zeros_like(features[0]) for _ in range(self.max_frames - len(features))]
            features.extend(padding)
        else:
            features = features[:self.max_frames]
        
        # Convert to numpy array
        features = np.array(features)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        shot_label = torch.FloatTensor([self.shot_success_labels[idx]])
        rebound_angle = torch.FloatTensor([self.rebound_angles[idx]])
        
        return features, shot_label, rebound_angle

def create_data_loaders(
    mediapipe_dir: str,
    shot_success_labels: list,
    rebound_angles: list,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        mediapipe_dir: Directory containing MediaPipe pose data
        shot_success_labels: List of binary labels (0 for made, 1 for missed)
        rebound_angles: List of rebound angles in degrees
        batch_size: Batch size for data loaders
        val_split: Validation set split ratio
        test_split: Test set split ratio
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Create dataset
    dataset = MultiTaskPoseDataset(
        mediapipe_dir=mediapipe_dir,
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Calculate class weights
    labels_tensor = torch.tensor(shot_success_labels)
    class_counts = torch.bincount(labels_tensor)
    total_samples = len(labels_tensor)
    # Use square root of inverse frequency for more balanced weights
    class_weights = torch.sqrt(total_samples / (2 * class_counts))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

def setup_training(
    model: nn.Module,
    device: torch.device,
    class_weights: torch.Tensor
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Set up optimizer and learning rate scheduler for training.
    
    Args:
        model: The model to train
        device: Device to train on
        class_weights: Weights for classification classes
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    # Use AdamW optimizer with moderate weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,  # Moderate learning rate
        weight_decay=0.01  # Moderate weight decay
    )
    
    return optimizer, class_weights

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load data
    video_name = "1.MP4"
    shot_success_labels, rebound_angles = load_data(video_name)
    
    # Print data statistics
    print(f"Number of shots: {len(shot_success_labels)}")
    print(f"Number of made shots (0): {shot_success_labels.count(0)}")
    print(f"Number of missed shots (1): {shot_success_labels.count(1)}")
    
    # Create data loaders and get class weights
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        mediapipe_dir="data/mediapipe_data",
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles,
        batch_size=16 if device.type == 'mps' else 32
    )
    
    # Create model
    model = MultiTaskCNNLSTM(
        input_dim=22,  # 6 keypoints × 3 coordinates + 4 derived features
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        lambda_cls=0.5
    ).to(device)
    
    # Set up training
    optimizer, class_weights = setup_training(model, device, class_weights)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=40,
        device=device,
        class_weights=class_weights
    )
    
    # Save training history
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print("Training completed. History saved to 'training_history.json'")

if __name__ == '__main__':
    main()
