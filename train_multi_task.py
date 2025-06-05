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
from binary_classifications import get_binary_classifications
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def setup_gpu():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return device
    else:
        return torch.device("cpu")

def load_data(video_name):
    shot_success_labels = get_binary_classifications(video_name)
    rebound_angles = load_angles(video_name)
    return shot_success_labels, rebound_angles

def load_angles(video_name):
    angles_dir = os.path.join(PROJECT_ROOT, "final_project", "data", "angles")
    angles_file = os.path.join(angles_dir, f"{video_name.split('.')[0]}_angles.csv")
    angles_df = pd.read_csv(angles_file)
    angles = angles_df['angle'].replace('', np.nan).fillna(-1).values
    return angles


def makeplotsforproj(history, save_dir='metrics_plots'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Total Loss')
    
=    plt.subplot(2, 2, 2)
    plt.plot(history['train_cls_acc'], label='Train Accuracy')
    plt.plot(history['val_cls_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_reg_rmse'], label='Train RMSE')
    plt.plot(history['val_reg_rmse'], label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (degrees)')
    plt.legend()
    plt.title('Rebound Angle RMSE')
    
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
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
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
            all_inputs.append(inputs.cpu().numpy())
            cls_preds, reg_preds = model(inputs)
            if isinstance(model, MultiTaskCNNLSTM):
                cls_probs = torch.softmax(cls_preds, dim=1)[:, 1]
                cls_preds = cls_preds.argmax(dim=1)
            else:
                cls_probs = cls_preds
                cls_preds = (cls_preds > 0.5).float()
            
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_probs.extend(cls_probs.cpu().numpy())
            all_cls_targets.extend(cls_targets.cpu().numpy())
            mask = (reg_targets != -1).bool()
            if mask.any():
                all_reg_preds.extend(reg_preds[mask].cpu().numpy())
                all_reg_targets.extend(reg_targets[mask].cpu().numpy())
    
    all_cls_preds = np.array(all_cls_preds)
    all_cls_probs = np.array(all_cls_probs)
    all_cls_targets = np.array(all_cls_targets)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_targets = np.array(all_reg_targets)
    all_inputs = np.concatenate(all_inputs, axis=0)
    
    plot_feature_importance(model, all_inputs, all_cls_targets, all_reg_targets, device, save_dir)
    
    cm = confusion_matrix(all_cls_targets, all_cls_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Make', 'Miss'])
    plt.yticks(tick_marks, ['Make', 'Miss'])
    
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
    
    if len(all_reg_preds) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_reg_targets, all_reg_preds, alpha=0.5)
        plt.plot([0, 180], [0, 180], 'r--')
        plt.xlabel('Actual Angle')
        plt.ylabel('Predicted Angle')
        plt.title('Predicted vs Actual Rebound Angles')
        plt.savefig(os.path.join(save_dir, 'angle_predictions.png'))
        plt.close()

def plot_feature_importance(model, inputs, cls_targets, reg_targets, device, save_dir):
    feature_names = [
        'Elbow Angle',
        'Arm Alignment',
        'Knee Bend',
        'Shot Arc'
    ]
    def cls_scorer(model, X, y):
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            preds, aa = model(X_tensor)
            if isinstance(model, MultiTaskCNNLSTM):
                preds = torch.softmax(preds, dim=1)[:, 1]
            return preds.cpu().numpy()
    
    def reg_scorer(model, X, y):
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            aa, preds = model(X_tensor)
            return preds.cpu().numpy()
    
    cls_importance = permutation_importance(
        model, inputs, cls_targets,
        n_repeats=10,
        random_state=42,
        scoring=cls_scorer
    )
    
    reg_mask = (reg_targets != -1)
    if reg_mask.any():
        reg_importance = permutation_importance(
            model, inputs[reg_mask], reg_targets[reg_mask],
            n_repeats=10,
            random_state=42,
            scoring=reg_scorer
        )
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, cls_importance.importances_mean)
    plt.errorbar(feature_names, cls_importance.importances_mean,
                yerr=cls_importance.importances_std,
                fmt='none', color='black', capsize=5)
    plt.title('Feature Importance for Shot Success Prediction')
    plt.xticks(rotation=45)
    plt.ylabel('Importance Score')
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
    device = setup_gpu()
    video_name = "1.MP4"
    shot_success_labels, rebound_angles = load_data(video_name)

    valid_angles = rebound_angles[rebound_angles != -1]
    left_misses = valid_angles[valid_angles < 45]
    straight_misses = valid_angles[(valid_angles >= 45) & (valid_angles <= 135)]
    right_misses = valid_angles[valid_angles > 135]

    mediapipe_dir = os.path.join("data", "mediapipe_data")
    dataset = MultiTaskPoseDataset(
        mediapipe_dir=mediapipe_dir,
        shot_success_labels=shot_success_labels,
        rebound_angles=rebound_angles
    )


    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=80
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.222,
        random_state=80
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

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

    model_type = "cnn_lstm"

    if model_type == "cnn_lstm":
        model = MultiTaskCNNLSTM(
            input_dim=22,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            lambda_cls=0.7
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01
    )
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=40,
        device=device
    )
    with open('multi_task_training_history.json', 'w') as f:
        json.dump(history, f)

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
            cls_targets = cls_targets.to(device).view(-1)
            reg_targets = reg_targets.to(device)
            cls_preds, reg_preds = model(inputs)
            cls_loss = nn.CrossEntropyLoss()(cls_preds, cls_targets.long())
            reg_loss = nn.MSELoss()(reg_preds, reg_targets)
            total_loss = cls_loss + reg_loss
            test_loss += total_loss.item()
            test_cls_loss += cls_loss.item()
            test_reg_loss += reg_loss.item()
            _, predicted = torch.max(cls_preds, 1)
            test_cls_correct += (predicted == cls_targets).sum().item()
            test_reg_preds.extend(reg_preds.cpu().numpy())
            test_reg_targets.extend(reg_targets.cpu().numpy())
            test_cls_preds.extend(predicted.cpu().numpy())
            test_cls_targets.extend(cls_targets.cpu().numpy())

    num_test_batches = len(test_loader)
    test_loss /= num_test_batches
    test_cls_loss /= num_test_batches
    test_reg_loss /= num_test_batches
    test_cls_acc = test_cls_correct / len(test_loader.dataset)

    test_reg_preds = np.array(test_reg_preds)
    test_reg_targets = np.array(test_reg_targets)
    test_reg_rmse = np.sqrt(np.mean((test_reg_preds - test_reg_targets) ** 2))
    test_reg_mae = np.mean(np.abs(test_reg_preds - test_reg_targets))

    from sklearn.metrics import r2_score
    test_reg_r2 = r2_score(test_reg_targets, test_reg_preds)

    from sklearn.metrics import f1_score, accuracy_score
    test_cls_f1 = f1_score(test_cls_targets, test_cls_preds, average='binary')
    test_cls_acc = accuracy_score(test_cls_targets, test_cls_preds)

    print("\nTest Set Metrics:")
    print(f"Total Loss: {test_loss:.4f}")
    print(f"Classification Loss: {test_cls_loss:.4f}")
    print(f"Regression Loss: {test_reg_loss:.4f}")
    print(f"Classification Accuracy: {test_cls_acc:.4f}")
    print(f"Classification F1 Score: {test_cls_f1:.4f}")
    print(f"Regression RMSE: {test_reg_rmse:.4f}°")
    print(f"Regression MAE: {test_reg_mae:.4f}°")
    print(f"Regression R^2: {test_reg_r2:.4f}")

    history['test_loss'] = test_loss
    history['test_cls_loss'] = test_cls_loss
    history['test_reg_loss'] = test_reg_loss
    history['test_cls_acc'] = test_cls_acc
    history['test_cls_f1'] = test_cls_f1
    history['test_reg_rmse'] = test_reg_rmse
    history['test_reg_mae'] = test_reg_mae
    history['test_reg_r2'] = test_reg_r2

    with open('training_history.json', 'w') as f:
        serializable_history = {
            k: float(v) if isinstance(v, (np.float32, np.float64, torch.Tensor)) else v
            for k, v in history.items()
        }
        json.dump(serializable_history, f)

    makeplotsforproj(history, test_metrics={
        'loss': test_loss,
        'cls_loss': test_cls_loss,
        'reg_loss': test_reg_loss,
        'cls_acc': test_cls_acc,
        'reg_rmse': test_reg_rmse,
        'reg_mae': test_reg_mae
    })

    plot_additional_metrics(model, test_loader, device, 'test_metrics_plots')

    return history

if __name__ == '__main__':
    main() 