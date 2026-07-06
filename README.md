# Improved COVID-19 Detection from Lung CT-Scans using EfficientNet-B3, CBAM Attention Module and Sugeno Fuzzy Integral

## Overview

This project presents an improved deep learning framework for the automatic detection of COVID-19 from lung CT scan images. The work extends the research paper **"COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble"** by enhancing its feature extraction capability while preserving the original decision fusion strategy.

The proposed model replaces the conventional convolutional neural network (CNN) backbone with **EfficientNet-B3** and integrates the **Convolutional Block Attention Module (CBAM)** to improve feature representation and focus on clinically significant regions of the CT images. The **Sugeno Fuzzy Integral** is retained as the decision fusion mechanism to maintain consistency with the original research while improving overall classification performance.

This project has been developed for academic and research purposes as part of a Bachelor's degree in Data Science.

---

## Key Features

- EfficientNet-B3 as the backbone feature extractor
- Convolutional Block Attention Module (CBAM)
- Sugeno Fuzzy Integral for decision fusion
- Transfer Learning
- Image preprocessing and data augmentation
- Training and validation pipeline
- Confusion matrix and ROC curve generation
- Classification report and evaluation metrics
- Single-image prediction support
- Google Colab compatible implementation

---

# Project Structure

```text
COVID19-CTScan/

│── dataset/
│   ├── COVID/
│   └── Non-COVID/

├── models/
│   ├── cbam.py
│   ├── efficientnet_cbam.py
│   └── sugeno.py

├── train.py
├── test.py
├── predict.py
├── dataset.py
├── utils.py
├── config.py
├── requirements.txt
└── README.md
```

---

# Dataset

This implementation uses the **SARS-CoV-2 CT-Scan Dataset**, which was also employed in the original research paper to ensure a fair comparison between the reproduced and the proposed models.

**Kaggle Dataset**

https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset

After downloading the dataset, organize it using the following directory structure:

```text
dataset/

    COVID/

    Non-COVID/
```

---

# Original Research Paper

**Title**

COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble

**Download Paper**

https://www.sciencedirect.com/science/article/pii/S001048252100214X

---

# Research Citation

Kundu, R., Singh, P. K., Mirjalili, S., & Sarkar, R.

*COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble.*

Computers in Biology and Medicine, Volume 138, 2021.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# Training

Run the following command to train the proposed model.

```bash
python train.py
```

---

# Testing

Evaluate the trained model using the test dataset.

```bash
python test.py
```

---

# Prediction

Predict the class of a single CT scan image.

```bash
python predict.py image.png
```

Example output:

```text
Prediction : COVID

Confidence : 99.47%
```

---

# Generated Results

After training and evaluation, the framework automatically generates and stores the following outputs inside the **results/** directory:

- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve
- Confusion Matrix
- ROC Curve
- Classification Report
- Prediction CSV File
- Best Trained Model (.pth)

```text
results/
```

---

# Proposed Architecture

The proposed framework follows the pipeline shown below.

<p align="center">
  <img src="images/proposed_architecture.png" alt="Proposed Architecture" width="100%">
</p>

The workflow begins with CT scan image preprocessing and data augmentation. Features are extracted using EfficientNet-B3 and refined through the CBAM attention module. The resulting feature maps are processed using global average pooling and a fully connected layer to generate class probabilities. Finally, the Sugeno Fuzzy Integral combines the predictions to produce the final COVID or Non-COVID classification.

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

# References

1. Kundu, R., Singh, P. K., Mirjalili, S., & Sarkar, R. *COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble.* Computers in Biology and Medicine, 2021.

2. Tan, M., & Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* International Conference on Machine Learning (ICML), 2019.

3. Woo, S., Park, J., Lee, J., & Kweon, I. *CBAM: Convolutional Block Attention Module.* European Conference on Computer Vision (ECCV), 2018.

---

# Author

**Laiba Mubashar**

Bachelor of Science in Data Science

Gift University, Gujranwala, Pakistan

---

# License

This project is developed solely for educational and research purposes. The implementation is based on the original research paper and is intended to demonstrate an improved deep learning framework for COVID-19 detection from lung CT scan images.
