# Improved COVID-19 Detection from Lung CT-Scans using EfficientNet-B3, CBAM Attention Module and Sugeno Fuzzy Integral

## 📌 Overview

This project presents an improved deep learning framework for COVID-19 detection from Lung CT-Scan images.

The implementation is based on the research paper:

**COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble**

Instead of redesigning the complete system, this project enhances the original architecture by replacing the conventional CNN backbone with **EfficientNet-B3** and integrating the **Convolutional Block Attention Module (CBAM)** while preserving the **Sugeno Fuzzy Integral** for decision fusion.

---

## 🚀 Features

- EfficientNet-B3 Backbone
- CBAM Attention Module
- Sugeno Fuzzy Integral Fusion
- Transfer Learning
- Data Augmentation
- Training & Validation Pipeline
- Confusion Matrix
- ROC Curve
- Classification Report
- Prediction Script
- Google Colab Compatible

---

# 📂 Project Structure

```
COVID19-CTScan/

│── dataset/

│ ├── COVID/

│ └── Non-COVID/

│

├── models/

│ ├── cbam.py

│ ├── efficientnet_cbam.py

│ └── sugeno.py

│

├── train.py

├── test.py

├── predict.py

├── dataset.py

├── utils.py

├── config.py

├── requirements.txt

└── README.md
```
 Dataset

This project uses the **SARS-CoV-2 CT-Scan Dataset**.

### Kaggle Dataset

https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset

If this dataset becomes unavailable, use the dataset recommended in the original research paper.

Dataset Structure

```
dataset/

    COVID/

    Non-COVID/
```
 📄 Original Research Paper

**Title**

COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble

Download Paper

https://www.sciencedirect.com/science/article/pii/S001048252100214X

---

# 📚 Research Citation

R. Kundu, P. K. Singh, S. Mirjalili, and R. Sarkar,

"COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble,"

Computers in Biology and Medicine,

Volume 138,

2021.

---

# ⚙ Installation

Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶ Training

```bash
python train.py
```

---

# 🧪 Testing

```bash
python test.py
```

---

# 🔍 Prediction

```bash
python predict.py image.png
```

Example Output

```
Prediction : COVID

Confidence : 99.47%
```

---

# 📈 Generated Results

The project automatically generates:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- ROC Curve
- Confusion Matrix
- Classification Report
- Prediction CSV
- Best Model (.pth)

All outputs are saved inside

```
results/
```

---

# 🧠 Proposed Architecture

```
Input CT Images

↓

Image Preprocessing

↓

Data Augmentation

↓

EfficientNet-B3

↓

CBAM Attention Module

↓

Global Average Pooling

↓

Fully Connected Layer

↓

Sugeno Fuzzy Integral

↓

Softmax

↓

COVID / Non-COVID
```

---

# 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

# 📑 References

1. Kundu R., Singh P.K., Mirjalili S., Sarkar R.

COVID-19 Detection from Lung CT-Scans using a Fuzzy Integral-Based CNN Ensemble.

Computers in Biology and Medicine, 2021.

2. Tan M., Le Q.

EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.

ICML, 2019.

3. Woo S., Park J., Lee J., Kweon I.

CBAM: Convolutional Block Attention Module.

ECCV, 2018.

---

# 👩‍💻 Author

Laiba Mubashar

BS Data Science

Gift University, Gujranwala

---

# 📜 License

This project is developed for educational and research purposes.
