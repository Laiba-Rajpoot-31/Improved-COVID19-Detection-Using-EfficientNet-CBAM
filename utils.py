import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc)

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across runs.
    Ensures deterministic behavior in PyTorch and Numpy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_hardware_info(model):
    """
    Prints the device being used (GPU or CPU) and the number of
    trainable parameters in the provided PyTorch model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*40)
    print("HARDWARE & MODEL INFO")
    print("="*40)
    
    if device.type == 'cuda':
        print(f"Device: {torch.cuda.get_device_name(0)} (GPU)")
    else:
        print("Device: CPU")
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")
    print("="*40 + "\n")

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the Best Model (.pth) checkpoint automatically.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_training_graphs(train_losses, val_losses, train_accs, val_accs, save_dir='results'):
    """
    Automatically generates and saves Training/Validation Accuracy and Loss Graphs.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss Graphs
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
    plt.close()
    
    # Accuracy Graphs
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accs, label='Training Accuracy', color='green', marker='o')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy', color='red', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_graph.png'))
    plt.close()

def evaluate_and_save_metrics(y_true, y_pred, y_scores, class_names, save_dir='results'):
    """
    Calculates Specificity, F1, AUC, etc., and generates Confusion Matrix, ROC curve, 
    Classification Report, and Metrics Report inside the results/ folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle binary metrics like specificity and ROC
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1]) # Assuming index 1 is COVID positive
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
    else:
        specificity = 0.0
        roc_auc = 0.0

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Classification Report File
    cr = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(cr)
        
    # Metrics Report File
    with open(os.path.join(save_dir, 'metrics_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        
    # Print Console Output explicitly matching the requirements
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Accuracy      : {acc*100:.2f} %")
    print(f"Precision     : {prec*100:.2f} %")
    print(f"Recall        : {rec*100:.2f} %")
    print(f"Specificity   : {specificity*100:.2f} %")
    print(f"F1 Score      : {f1*100:.2f} %")
    print(f"ROC AUC       : {roc_auc*100:.2f} %")
    print("="*40 + "\n")
