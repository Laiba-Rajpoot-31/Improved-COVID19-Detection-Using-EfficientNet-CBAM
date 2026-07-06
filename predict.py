import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from efficientnet_cbam import EfficientNetCBAM
from sugeno import ensemble_sugeno

def predict_single_image(image_path, models_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    class_names = ['COVID', 'Non-COVID']
    num_classes = len(class_names)
    
    seeds = [42, 101, 2024, 777]
    models = []
    
    for s in seeds:
        model_path = os.path.join(models_dir, f"model_seed_{s}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ensemble model {model_path} missing. Train models first.")
            
        model = EfficientNetCBAM(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
        
    prob_list = [[] for _ in range(len(models))]
    
    with torch.no_grad():
        for i, model in enumerate(models):
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            prob_list[i].extend(probs.cpu().numpy())
            
    for i in range(len(prob_list)):
        prob_list[i] = np.array(prob_list[i])
        
    sugeno_preds, sugeno_integral_values = ensemble_sugeno(prob_list)
    
    final_prediction_index = sugeno_preds[0]
    final_class_name = class_names[final_prediction_index]
    confidence_score = sugeno_integral_values[0][final_prediction_index]
    
    print("\n" + "="*40)
    print("PREDICTION RESULT")
    print("="*40)
    print(f"File         : {os.path.basename(image_path)}")
    print(f"Prediction   : {final_class_name}")
    print(f"Integral Score (Confidence): {confidence_score:.4f}")
    print("="*40 + "\n")
    
    return final_class_name, confidence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict COVID-19 from a single CT scan image")
    parser.add_argument('image_path', type=str, help="Path to the input CT scan image")
    parser.add_argument('--models_dir', type=str, default="results", help="Directory containing the .pth ensemble models")
    
    args = parser.parse_args()
    predict_single_image(args.image_path, args.models_dir)
