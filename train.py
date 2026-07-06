import os
import time
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloaders
from efficientnet_cbam import EfficientNetCBAM
from utils import set_seed, print_hardware_info, EarlyStopping, save_training_graphs

def train_single_model(data_dir, seed, epochs, batch_size, model_save_path, results_dir):
    set_seed(seed)
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training with Random Seed: {seed} on {device.type.upper()} ---")
    
    model = EfficientNetCBAM(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    if seed == 42:
        print_hardware_info(model)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=7, verbose=True, path=model_save_path)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc.item()
            
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        
        scheduler.step(val_epoch_loss)
        early_stopping(val_epoch_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    elapsed_time = time.time() - start_time
    print(f"Training Time for Seed {seed}: {elapsed_time/60:.2f} minutes")
    
    graph_dir = os.path.join(results_dir, f"graphs_seed_{seed}")
    save_training_graphs(train_losses, val_losses, train_accs, val_accs, save_dir=graph_dir)
    
    return best_val_acc, early_stopping.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COVID-19 EfficientNet+CBAM Ensemble")
    parser.add_argument('--data_dir', type=str, default="data", help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--results_dir', type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
   

    print("="*60)
    print("Current Working Directory :", os.getcwd())
    print("data_dir =", args.data_dir)
    print("Absolute Path =", os.path.abspath(args.data_dir))
    print("Data Exists :", os.path.exists(args.data_dir))
    print("Train Exists :", os.path.exists(os.path.join(args.data_dir,"train")))
    print("Val Exists :", os.path.exists(os.path.join(args.data_dir,"val")))
    print("="*60)

    os.makedirs(args.results_dir, exist_ok=True)
    
    seeds = [42, 101, 2024, 777]
    overall_best_acc = 0.0
    overall_best_path = ""
    total_start_time = time.time()
    
    for i, seed in enumerate(seeds):
        model_save_path = os.path.join(args.results_dir, f"model_seed_{seed}.pth")
        best_acc, saved_path = train_single_model(
            data_dir=args.data_dir, seed=seed, epochs=args.epochs, 
            batch_size=args.batch_size, model_save_path=model_save_path, 
            results_dir=args.results_dir
        )
        if best_acc > overall_best_acc:
            overall_best_acc = best_acc
            overall_best_path = saved_path
            
    final_best_path = os.path.join(args.results_dir, "best_model.pth")
    if os.path.exists(overall_best_path):
        shutil.copy(overall_best_path, final_best_path)
        
    total_time = time.time() - total_start_time
    print("\n" + "="*40)
    print(f"Total Ensemble Training Time: {total_time/60:.2f} minutes")
    print(f"Best Individual Model Validation Accuracy: {overall_best_acc*100:.2f}%")
    print(f"Best Model Saved To: {final_best_path}")
    print("="*40 + "\n")
