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
        input_dim: int = 22,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lambda_cls: float = 0.65
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
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
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
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
        batch_size, seq_len, _ = x.shape

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # Two outputs, one for regression one for classification
        cls_out = self.cls_head(last_hidden)
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
        cls_target = cls_target.view(-1)
        
        cls_loss = F.cross_entropy(
            cls_pred, 
            cls_target.long(),
            weight=class_weights,
            label_smoothing=0.1
        )
        mask = (reg_target != -1).float()
        reg_pred = torch.clamp(reg_pred, 0, 180)
        reg_loss = F.mse_loss(reg_pred * mask, reg_target * mask) / 180.0
        
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=cls_pred.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = self.lambda_cls * cls_loss + (1 - self.lambda_cls) * reg_loss + l2_lambda * l2_reg
        
        return total_loss, cls_loss, reg_loss

class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        lambda_cls: float = 0.7
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 44, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        x = self.input_embedding(x)
        pos_encoding = self.pos_encoder.repeat(batch_size, 1, 1)
        x = x + pos_encoding[:, :seq_len, :]

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        
        # Once again, two outputs, one for regression and one for classification
        cls_out = torch.sigmoid(self.cls_head(x))
        reg_out = self.reg_head(x)
        
        return cls_out, reg_out
    
    def compute_loss(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = F.binary_cross_entropy(cls_pred, cls_target)
        reg_loss = F.mse_loss(reg_pred, reg_target)

        total_loss =  (1-self.lambda_cls)*reg_loss +  self.lambda_cls * cls_loss
        
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
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_reg_loss': [], 'val_reg_loss': [],
        'train_cls_acc': [], 'val_cls_acc': [],
        'train_reg_rmse': [], 'val_reg_rmse': [],
        'train_reg_mae': [], 'val_reg_mae': []
    }
    
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
        model.train()
        train_losses = []
        train_cls_losses = []
        train_reg_losses = []
        train_cls_correct = 0
        train_cls_total = 0
        train_reg_errors = []
        train_reg_abs_errors = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            inputs, cls_targets, reg_targets = batch
            inputs = inputs.to(device)
            cls_targets = cls_targets.to(device).view(-1)
            reg_targets = reg_targets.to(device)
            
            if model.training:
                noise = torch.randn_like(inputs) * 0.01
                inputs = inputs + noise
            
            optimizer.zero_grad()
            cls_preds, reg_preds = model(inputs)

            reg_preds = torch.clamp(reg_preds, 0, 180)
            loss, cls_loss, reg_loss = model.compute_loss(
                cls_preds, reg_preds, cls_targets, reg_targets, class_weights
            )
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            train_reg_losses.append(reg_loss.item())
            
            _, predicted = cls_preds.max(1)
            train_cls_correct += predicted.eq(cls_targets).sum().item()
            train_cls_total += cls_targets.size(0)
            
            mask = (reg_targets != -1).float()
            if mask.sum() > 0:
                reg_errors = ((reg_preds - reg_targets) * mask).pow(2).sqrt().detach().cpu().numpy()
                train_reg_errors.extend(reg_errors[mask.cpu().numpy().astype(bool)])
                
                abs_errors = ((reg_preds - reg_targets) * mask).abs().detach().cpu().numpy()
                train_reg_abs_errors.extend(abs_errors[mask.cpu().numpy().astype(bool)])
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls_acc': f'{train_cls_correct/train_cls_total:.4f}'
            })
        
        model.eval()
        val_losses = []
        val_cls_losses = []
        val_reg_losses = []
        val_cls_correct = 0
        val_cls_total = 0
        val_reg_errors = []
        val_reg_abs_errors = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                inputs, cls_targets, reg_targets = batch
                inputs = inputs.to(device)
                cls_targets = cls_targets.to(device).view(-1)
                reg_targets = reg_targets.to(device)
                
                cls_preds, reg_preds = model(inputs)
                reg_preds = torch.clamp(reg_preds, 0, 180)
                
                loss, cls_loss, reg_loss = model.compute_loss(
                    cls_preds, reg_preds, cls_targets, reg_targets, class_weights
                )
                
                val_losses.append(loss.item())
                val_cls_losses.append(cls_loss.item())
                val_reg_losses.append(reg_loss.item())

                _, predicted = cls_preds.max(1)
                val_cls_correct += predicted.eq(cls_targets).sum().item()
                val_cls_total += cls_targets.size(0)
                
                mask = (reg_targets != -1).float()
                if mask.sum() > 0:
                    reg_errors = ((reg_preds - reg_targets) * mask).pow(2).sqrt().cpu().numpy()
                    val_reg_errors.extend(reg_errors[mask.cpu().numpy().astype(bool)])

                    abs_errors = ((reg_preds - reg_targets) * mask).abs().cpu().numpy()
                    val_reg_abs_errors.extend(abs_errors[mask.cpu().numpy().astype(bool)])

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_acc': f'{val_cls_correct/val_cls_total:.4f}'
                })

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
        self.mediapipe_dir = mediapipe_dir
        self.shot_success_labels = shot_success_labels
        self.rebound_angles = rebound_angles
        self.max_frames = max_frames
        
        self.data_files = sorted(
            [f for f in os.listdir(mediapipe_dir) if f.endswith('_mediapipe.json')],
            key=lambda x: int(x.split('_')[1])
        )
        
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
        if normalize:
            self._calculate_global_stats()
    
    def _calculate_global_stats(self):
        all_features = []
        for idx in range(len(self.data_files)):
            with open(os.path.join(self.mediapipe_dir, self.data_files[idx]), 'r') as f:
                data = json.load(f)
            
            frames = data['frames']
            for frame in frames:
                landmarks = frame.get('landmarks', [])
                if not landmarks:
                    continue
                
                keypoints = np.zeros((33, 3))
                for j, landmark in enumerate(landmarks):
                    if j >= 33 or landmark is None:
                        continue
                    keypoints[j] = [
                        landmark.get('x', 0.0),
                        landmark.get('y', 0.0),
                        landmark.get('z', 0.0)
                    ]
                
                features = self._calculate_frame_features(keypoints, normalize=False)
                all_features.append(features)
        
        all_features = np.array(all_features)
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0) + 1e-8
    
    def _calculate_frame_features(self, keypoints, normalize=True):
        features = []

        for i, (keypoint_name, idx) in enumerate(self.shooting_keypoints.items()):
            coords = keypoints[idx]
            if normalize and self.feature_means is not None:
                for j in range(3):
                    coord_idx = i * 3 + j
                    normalized_coord = (coords[j] - self.feature_means[coord_idx]) / self.feature_stds[coord_idx]
                    features.append(normalized_coord)
            else:
                features.extend(coords)
        
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
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        with open(os.path.join(self.mediapipe_dir, self.data_files[idx]), 'r') as f:
            data = json.load(f)
        
        frames = data['frames']
        num_frames = min(len(frames), self.max_frames)
        features = []
        
        for i in range(num_frames):
            frame = frames[i]
            landmarks = frame.get('landmarks', [])
            if not landmarks:
                continue
            
            keypoints = np.zeros((33, 3))
            for j, landmark in enumerate(landmarks):
                if j >= 33 or landmark is None:
                    continue
                keypoints[j] = [
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0)
                ]
            
            frame_features = self._calculate_frame_features(keypoints)
            features.append(frame_features)
        
        if len(features) < self.max_frames:
            padding = [np.zeros_like(features[0]) for _ in range(self.max_frames - len(features))]
            features.extend(padding)
        else:
            features = features[:self.max_frames]
        
        features = np.array(features)
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
    dataset = MultiTaskPoseDataset(
        mediapipe_dir=mediapipe_dir,
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles
    )
    
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    labels_tensor = torch.tensor(shot_success_labels)
    class_counts = torch.bincount(labels_tensor)
    total_samples = len(labels_tensor)
    class_weights = torch.sqrt(total_samples / (2 * class_counts))
    
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
    class_weights = class_weights.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01
    )
    
    return optimizer, class_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    video_name = "1.MP4"
    shot_success_labels, rebound_angles = load_data(video_name)

    print(f"Number of shots: {len(shot_success_labels)}")
    print(f"Number of made shots (0): {shot_success_labels.count(0)}")
    print(f"Number of missed shots (1): {shot_success_labels.count(1)}")

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        mediapipe_dir="data/mediapipe_data",
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles,
        batch_size=16 if device.type == 'mps' else 32
    )
    
    model = MultiTaskCNNLSTM(
        input_dim=22,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        lambda_cls=0.5
    ).to(device)
    
    optimizer, class_weights = setup_training(model, device, class_weights)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=40,
        device=device,
        class_weights=class_weights
    )
    
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print("Training completed. History saved to 'training_history.json'")

if __name__ == '__main__':
    main()
