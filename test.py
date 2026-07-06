import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_dataloaders
from efficientnet_cbam import EfficientNetCBAM
from sugeno import ensemble_sugeno
from utils import evaluate_and_save_metrics

def test_ensemble(data_dir, models_dir, results_dir, batch_size=32):
    _, val_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Testing on {device.type.upper()} ---")
    
    seeds = [42, 101, 2024, 777]
    model_paths = [os.path.join(models_dir, f"model_seed_{s}.pth") for s in seeds]
    
    for p in model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing ensemble model: {p}. Please train the models first.")
            
    models = []
    for p in model_paths:
        model = EfficientNetCBAM(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(p, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
        
    print(f"Successfully loaded {len(models)} models for the ensemble.")
    
    start_time = time.time()
    
    all_labels = []
    all_paths = []
    prob_list = [[] for _ in range(len(models))]
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(val_loader, desc="Testing Models"):
            inputs = inputs.to(device)
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            
            for i, model in enumerate(models):
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                prob_list[i].extend(probs.cpu().numpy())
                
    all_labels = np.array(all_labels)
    for i in range(len(prob_list)):
        prob_list[i] = np.array(prob_list[i])
        
    print("\nApplying Sugeno Fuzzy Integral Fusion...")
    sugeno_preds, sugeno_integral_values = ensemble_sugeno(prob_list)
    
    test_time = time.time() - start_time
    print(f"Testing Time: {test_time:.2f} seconds")
    
    evaluate_and_save_metrics(
        y_true=all_labels, 
        y_pred=sugeno_preds, 
        y_scores=sugeno_integral_values,
        class_names=class_names, 
        save_dir=results_dir
    )
    
    csv_path = os.path.join(results_dir, "predictions.csv")
    actual_class_names = [class_names[lbl] for lbl in all_labels]
    predicted_class_names = [class_names[pred] for pred in sugeno_preds]
    
    df_data = {
        'File_Path': all_paths,
        'Actual_Label': actual_class_names,
        'Predicted_Label': predicted_class_names
    }
    
    for c in range(num_classes):
        df_data[f'Integral_Score_{class_names[c]}'] = sugeno_integral_values[:, c]
        
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    
    print(f"Prediction CSV successfully saved to: {csv_path}")
    print("Testing Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test COVID-19 Sugeno Ensemble")
    parser.add_argument('--data_dir', type=str, default="data", help="Path to dataset")
    parser.add_argument('--models_dir', type=str, default="results", help="Directory containing .pth models")
    parser.add_argument('--results_dir', type=str, default="results", help="Output directory for results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    test_ensemble(
        data_dir=args.data_dir, models_dir=args.models_dir,
        results_dir=args.results_dir, batch_size=args.batch_size
    )
