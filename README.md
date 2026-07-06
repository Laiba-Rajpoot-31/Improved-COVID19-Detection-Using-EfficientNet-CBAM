# COVID-19 Detection from Lung CT-Scans (Assignment 3)

## Overview
This repository contains the improved implementation for **Assignment 3**, building upon the base paper *"COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble" (Kundu et al., 2021)*.

The core contribution of the original paper—the **Sugeno Fuzzy Integral**—is preserved for fusing probability predictions. However, the legacy multi-CNN backbone has been entirely replaced with a far more robust, single-architecture ensemble using **EfficientNet-B3** coupled with a **Convolutional Block Attention Module (CBAM)**.

### Improvements Made:
1. **Backbone Replacement**: Replaced VGG-11, GoogLeNet, SqueezeNet, and WideResNet with **EfficientNet-B3**.
2. **Attention Mechanism**: Added a **CBAM** (Channel & Spatial Attention) after EfficientNet-B3 to heavily focus on relevant GGO (Ground-Glass Opacity) features in the lungs.
3. **Data Augmentation**: Added a strict augmentation pipeline (Resize, Random Horizontal Flip, Random Rotation, Brightness, Contrast, Crop, Affine).
4. **Optimization Pipeline**: Switched to the `Adam` optimizer (LR: 0.0001) with a `ReduceLROnPlateau` scheduler and `EarlyStopping`.
5. **Automated Evaluation**: Code automatically generates Training/Validation Accuracy and Loss graphs, ROC Curves, Confusion Matrices, Classification Reports, and a final Prediction CSV.

## Project Structure
- `dataset.py`: PyTorch dataset handling with all required data augmentations.
- `cbam.py`: Implementation of Channel and Spatial Attention logic.
- `efficientnet_cbam.py`: The newly proposed architecture combining EfficientNet-B3 and CBAM.
- `sugeno.py`: The Sugeno Fuzzy Integral mathematical logic.
- `utils.py`: Contains Early Stopping, plotting (Loss/Acc/ROC), metric calculations, and Hardware printing.
- `train.py`: Trains an ensemble of models using different random seeds for diversity.
- `test.py`: Loads the models, evaluates them, applies Softmax, performs the Sugeno fusion, and saves results.
- `predict.py`: Standalone CLI to run inference on a single image.

## How to Run in Google Colab

### 1. Setup Environment
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure the SARS-CoV-2 CT Scan Dataset is extracted and structured exactly like this:
```text
data/
├── train/
│   ├── COVID/
│   └── Non-COVID/
└── val/
    ├── COVID/
    └── Non-COVID/
```

### 3. Training the Ensemble
Run `train.py` to train the ensemble models (uses 4 random seeds by default). The results and models will automatically be saved into a `results/` folder.
```bash
python train.py --data_dir data/ --epochs 50 --batch_size 32
```

### 4. Testing and Evaluation
Run `test.py` to load the ensemble, apply the Sugeno Fuzzy Integral fusion, and generate the final outputs (Graphs, CSVs, Metrics Report).
```bash
python test.py --data_dir data/ --models_dir results/ --results_dir results/
```

### 5. Single Image Prediction
To predict a single CT scan image:
```bash
python predict.py "data/val/COVID/some_image.png" --models_dir results/
```
