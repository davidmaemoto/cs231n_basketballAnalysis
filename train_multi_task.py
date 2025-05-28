import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from cnn_lstm import MultiTaskCNNLSTM, MultiTaskTransformer, MultiTaskPoseDataset, train_model
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def load_data(video_name):
    """Load shot success labels and rebound angles for a video"""
    # Load shot success labels (0 for made, 1 for missed)
    from binary_classifications import get_binary_classifications
    shot_success_labels = get_binary_classifications(video_name)
    
    # Load rebound angles (in data/angles directory)
    rebound_angles = load_angles(video_name)
    return shot_success_labels, rebound_angles

def load_angles(video_name):
    """Load angles from CSV file and handle missing values"""
    angles_dir = os.path.join(PROJECT_ROOT, "final_project", "data", "angles")
    angles_file = os.path.join(angles_dir, f"{video_name.split('.')[0]}_angles.csv")
    angles_df = pd.read_csv(angles_file)
    
    # Convert empty strings to -1 (indicating made shots)
    angles = angles_df['angle'].replace('', np.nan).fillna(-1).values
    
    return angles


def plot_metrics(history, test_metrics=None, save_dir='metrics_plots'):
    """Plot and save training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Total Loss')
    
    # Plot classification accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_cls_acc'], label='Train Accuracy')
    plt.plot(history['val_cls_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy')
    
    # Plot regression RMSE
    plt.subplot(2, 2, 3)
    plt.plot(history['train_reg_rmse'], label='Train RMSE')
    plt.plot(history['val_reg_rmse'], label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (degrees)')
    plt.legend()
    plt.title('Rebound Angle RMSE')
    
    # Plot individual losses
    plt.subplot(2, 2, 4)
    plt.plot(history['train_cls_loss'], label='Train Cls Loss')
    plt.plot(history['val_cls_loss'], label='Val Cls Loss')
    plt.plot(history['train_reg_loss'], label='Train Reg Loss')
    plt.plot(history['val_reg_loss'], label='Val Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Individual Losses')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_additional_metrics(model, val_loader, device, save_dir='metrics_plots'):
    """Generate additional visualization plots similar to baseline models"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Lists to store predictions and targets
    all_cls_preds = []
    all_cls_probs = []
    all_cls_targets = []
    all_reg_preds = []
    all_reg_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, cls_targets, reg_targets = batch
            inputs = inputs.to(device)
            cls_targets = cls_targets.to(device)
            reg_targets = reg_targets.to(device)
            
            # Store inputs for feature importance
            all_inputs.append(inputs.cpu().numpy())
            
            # Get model predictions
            cls_preds, reg_preds = model(inputs)
            
            # Store classification predictions
            if isinstance(model, MultiTaskCNNLSTM):
                cls_probs = torch.softmax(cls_preds, dim=1)[:, 1]  # Probability of class 1
                cls_preds = cls_preds.argmax(dim=1)
            else:  # MultiTaskTransformer
                cls_probs = cls_preds
                cls_preds = (cls_preds > 0.5).float()
            
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_probs.extend(cls_probs.cpu().numpy())
            all_cls_targets.extend(cls_targets.cpu().numpy())
            
            # Store regression predictions (only for missed shots)
            mask = (reg_targets != -1).bool()
            if mask.any():
                all_reg_preds.extend(reg_preds[mask].cpu().numpy())
                all_reg_targets.extend(reg_targets[mask].cpu().numpy())
    
    # Convert to numpy arrays
    all_cls_preds = np.array(all_cls_preds)
    all_cls_probs = np.array(all_cls_probs)
    all_cls_targets = np.array(all_cls_targets)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_targets = np.array(all_reg_targets)
    all_inputs = np.concatenate(all_inputs, axis=0)
    
    # Plot feature importance
    plot_feature_importance(model, all_inputs, all_cls_targets, all_reg_targets, device, save_dir)
    
    # 1. Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_cls_targets, all_cls_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Make', 'Miss'])
    plt.yticks(tick_marks, ['Make', 'Miss'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_cls_targets, all_cls_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # 3. Plot predicted vs actual angles (only for missed shots)
    if len(all_reg_preds) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_reg_targets, all_reg_preds, alpha=0.5)
        plt.plot([0, 180], [0, 180], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Angle')
        plt.ylabel('Predicted Angle')
        plt.title('Predicted vs Actual Rebound Angles')
        plt.savefig(os.path.join(save_dir, 'angle_predictions.png'))
        plt.close()

def plot_feature_importance(model, inputs, cls_targets, reg_targets, device, save_dir):
    """Calculate and plot feature importance for engineered features"""
    from sklearn.inspection import permutation_importance
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define feature names
    feature_names = [
        'Elbow Angle',
        'Arm Alignment',
        'Knee Bend',
        'Shot Arc'
    ]
    
    # Calculate feature importance for classification
    def cls_scorer(model, X, y):
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            preds, _ = model(X_tensor)
            if isinstance(model, MultiTaskCNNLSTM):
                preds = torch.softmax(preds, dim=1)[:, 1]
            return preds.cpu().numpy()
    
    # Calculate feature importance for regression
    def reg_scorer(model, X, y):
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            _, preds = model(X_tensor)
            return preds.cpu().numpy()
    
    # Calculate permutation importance for classification
    cls_importance = permutation_importance(
        model, inputs, cls_targets,
        n_repeats=10,
        random_state=42,
        scoring=cls_scorer
    )
    
    # Calculate permutation importance for regression
    reg_mask = (reg_targets != -1)
    if reg_mask.any():
        reg_importance = permutation_importance(
            model, inputs[reg_mask], reg_targets[reg_mask],
            n_repeats=10,
            random_state=42,
            scoring=reg_scorer
        )
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    
    # Plot classification importance
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, cls_importance.importances_mean)
    plt.errorbar(feature_names, cls_importance.importances_mean,
                yerr=cls_importance.importances_std,
                fmt='none', color='black', capsize=5)
    plt.title('Feature Importance for Shot Success Prediction')
    plt.xticks(rotation=45)
    plt.ylabel('Importance Score')
    
    # Plot regression importance if there are missed shots
    if reg_mask.any():
        plt.subplot(1, 2, 2)
        plt.bar(feature_names, reg_importance.importances_mean)
        plt.errorbar(feature_names, reg_importance.importances_mean,
                    yerr=reg_importance.importances_std,
                    fmt='none', color='black', capsize=5)
        plt.title('Feature Importance for Rebound Angle Prediction')
        plt.xticks(rotation=45)
        plt.ylabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def main():
    # Set up GPU
    device = setup_gpu()
    print(f'Using device: {device}')
    
    # Load data
    video_name = "1.MP4"
    shot_success_labels, rebound_angles = load_data(video_name)
    
    # Print data statistics
    print(f"Number of shots: {len(shot_success_labels)}")
    print(f"Number of made shots (0): {shot_success_labels.count(0)}")
    print(f"Number of missed shots (1): {shot_success_labels.count(1)}")
    
    # Calculate statistics for missed shots only (angles != -1)
    valid_angles = rebound_angles[rebound_angles != -1]
    left_misses = valid_angles[valid_angles < 45]  # Left misses (0-45°)
    straight_misses = valid_angles[(valid_angles >= 45) & (valid_angles <= 135)]  # Straight misses (45-135°)
    right_misses = valid_angles[valid_angles > 135]  # Right misses (135-180°)
    
    print(f"Number of left misses (0-45°): {len(left_misses)}")
    print(f"Number of straight misses (45-135°): {len(straight_misses)}")
    print(f"Number of right misses (135-180°): {len(right_misses)}")
    
    # Calculate average and std for all missed shots
    print(f"Average miss angle: {np.mean(valid_angles):.2f}°")
    print(f"Std of miss angles: {np.std(valid_angles):.2f}°")
    
    # Create dataset
    mediapipe_dir = os.path.join("data", "mediapipe_data")
    dataset = MultiTaskPoseDataset(
        mediapipe_dir=mediapipe_dir,
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles
    )
    
    # Print dataset statistics
    print(f"Number of MediaPipe files: {len(dataset)}")
    if len(dataset) != len(shot_success_labels):
        print("WARNING: Number of MediaPipe files does not match number of labels!")
        print("This might indicate a mismatch between the data and labels.")
        return
    
    # Create train/val/test split
    indices = list(range(len(dataset)))
    # First split: 90% train+val, 10% test
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=0.1,  # 10% for test
        random_state=80
    )
    
    # Second split: 70% train, 20% val (from the remaining 90%)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.222,  # 0.222 * 0.9 ≈ 0.2 (20% of total)
        random_state=80
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Smaller batch sizes for more iterations
    train_batch_size = 8 if device.type == 'mps' else 16
    val_batch_size = 4 if device.type == 'mps' else 8
    test_batch_size = 4 if device.type == 'mps' else 8
    
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Print iterations per epoch
    print(f"\nIterations per epoch:")
    print(f"Training: {len(train_loader)}")
    print(f"Validation: {len(val_loader)}")
    print(f"Test: {len(test_loader)}")
    
    # Print split sizes
    print(f"\nDataset split sizes:")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/len(dataset)*100:.1f}%)")
    print(f"Validation: {len(val_indices)} samples ({len(val_indices)/len(dataset)*100:.1f}%)")
    print(f"Test: {len(test_indices)} samples ({len(test_indices)/len(dataset)*100:.1f}%)")
    
    # Choose model type (CNN-LSTM or Transformer)
    model_type = "cnn_lstm"  # or "transformer"
    
    if model_type == "cnn_lstm":
        model = MultiTaskCNNLSTM(
            input_dim=22,  # 6 keypoints × 3 coordinates + 4 derived features
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            lambda_cls=0.7  # Weight for classification loss
        ).to(device)
    else:
        model = MultiTaskTransformer(
            input_dim=22,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.2,
            lambda_cls=0.7
        ).to(device)
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=40,
        device=device
    )
    
    # Save training history
    with open('multi_task_training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_cls_loss = 0
    test_reg_loss = 0
    test_cls_correct = 0
    test_reg_preds = []
    test_reg_targets = []
    test_cls_preds = []
    test_cls_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, cls_targets, reg_targets = batch
            inputs = inputs.to(device)
            cls_targets = cls_targets.to(device).view(-1)  # Ensure correct shape
            reg_targets = reg_targets.to(device)
            
            # Forward pass
            cls_preds, reg_preds = model(inputs)
            
            # Calculate losses
            cls_loss = nn.CrossEntropyLoss()(cls_preds, cls_targets.long())
            reg_loss = nn.MSELoss()(reg_preds, reg_targets)
            total_loss = cls_loss + reg_loss
            
            # Update metrics
            test_loss += total_loss.item()
            test_cls_loss += cls_loss.item()
            test_reg_loss += reg_loss.item()
            
            # Classification accuracy
            _, predicted = torch.max(cls_preds, 1)
            test_cls_correct += (predicted == cls_targets).sum().item()
            
            # Store predictions and targets for metrics
            test_reg_preds.extend(reg_preds.cpu().numpy())
            test_reg_targets.extend(reg_targets.cpu().numpy())
            test_cls_preds.extend(predicted.cpu().numpy())
            test_cls_targets.extend(cls_targets.cpu().numpy())
    
    # Calculate average metrics
    num_test_batches = len(test_loader)
    test_loss /= num_test_batches
    test_cls_loss /= num_test_batches
    test_reg_loss /= num_test_batches
    test_cls_acc = test_cls_correct / len(test_loader.dataset)
    
    # Calculate regression metrics
    test_reg_preds = np.array(test_reg_preds)
    test_reg_targets = np.array(test_reg_targets)
    test_reg_rmse = np.sqrt(np.mean((test_reg_preds - test_reg_targets) ** 2))
    test_reg_mae = np.mean(np.abs(test_reg_preds - test_reg_targets))
    
    # Calculate R^2 for regression
    from sklearn.metrics import r2_score
    test_reg_r2 = r2_score(test_reg_targets, test_reg_preds)
    
    # Calculate F1 score for classification
    from sklearn.metrics import f1_score, accuracy_score
    test_cls_f1 = f1_score(test_cls_targets, test_cls_preds, average='binary')
    test_cls_acc = accuracy_score(test_cls_targets, test_cls_preds)
    
    # Print test metrics
    print("\nTest Set Metrics:")
    print(f"Total Loss: {test_loss:.4f}")
    print(f"Classification Loss: {test_cls_loss:.4f}")
    print(f"Regression Loss: {test_reg_loss:.4f}")
    print(f"Classification Accuracy: {test_cls_acc:.4f}")
    print(f"Classification F1 Score: {test_cls_f1:.4f}")
    print(f"Regression RMSE: {test_reg_rmse:.4f}°")
    print(f"Regression MAE: {test_reg_mae:.4f}°")
    print(f"Regression R^2: {test_reg_r2:.4f}")
    
    # Save test metrics to history
    history['test_loss'] = test_loss
    history['test_cls_loss'] = test_cls_loss
    history['test_reg_loss'] = test_reg_loss
    history['test_cls_acc'] = test_cls_acc
    history['test_cls_f1'] = test_cls_f1
    history['test_reg_rmse'] = test_reg_rmse
    history['test_reg_mae'] = test_reg_mae
    history['test_reg_r2'] = test_reg_r2
    
    # Save history to JSON
    with open('training_history.json', 'w') as f:
        # Convert all values to Python native types
        serializable_history = {
            k: float(v) if isinstance(v, (np.float32, np.float64, torch.Tensor)) else v 
            for k, v in history.items()
        }
        json.dump(serializable_history, f)
    
    # Plot metrics with test set results
    plot_metrics(history, test_metrics={
        'loss': test_loss,
        'cls_loss': test_cls_loss,
        'reg_loss': test_reg_loss,
        'cls_acc': test_cls_acc,
        'reg_rmse': test_reg_rmse,
        'reg_mae': test_reg_mae
    })
    
    # Generate additional visualizations for test set
    plot_additional_metrics(model, test_loader, device, 'test_metrics_plots')
    
    return history

if __name__ == '__main__':
    main() 