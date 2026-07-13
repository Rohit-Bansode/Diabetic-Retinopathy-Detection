# 🩺 Diabetic Retinopathy Severity Classification

## Project Title: 
Automated Diabetic Retinopathy Severity Classification using CBAM-Enhanced EfficientNetB3, Ordinal-Aware Learning, and Grad-CAM Explainability


## Table of Contents

1. [Overview](#overview)
2. [Goals and Objectives](#goals-and-objectives)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Database](#database)
7. [Model Details](#model-details)
8. [Results](#results)
9. [Explainability (Grad-CAM)](#Explainability)
10. [Streamlit Clinical Application](#streamlit-clinical-application)
11. [Installation](#installation)
13. [Contributing](#contributing)
14. [Contact](#contact)

______________
## Overview:
Diabetic Retinopathy (DR) is an eye disease caused by diabetes that can damage the retina and may lead to vision loss or blindness if it is not detected early. Regular screening is important, but examining retinal images manually takes time and requires experienced ophthalmologists. In many rural and remote areas, access to eye specialists is limited, making early diagnosis difficult.

This project presents an AI-based system that automatically detects and classifies Diabetic Retinopathy from retinal fundus images. The model uses a CBAM-enhanced EfficientNet-B3 deep learning architecture with Ordinal-Aware Learning to classify retinal images into five severity levels. It also generates Grad-CAM visualizations to highlight the regions of the retina that influenced the model's prediction, making the results more understandable and explainable.

To make the system practical for real-world use, the trained model is deployed as an interactive Streamlit web application, allowing users to upload a retinal image and receive the predicted DR grade along with a confidence score and visual explanation.

The model classifies retinal fundus images into the following five severity grades based on the International Clinical Diabetic Retinopathy Disease Severity Scale:

| Grade | Class |
|-------|-------|
| 0 | No DR     |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |
______________
## 🎯 Goals and Objectives

**Primary Goal:** To build an automated Diabetic Retinopathy severity classification system using a CBAM-enhanced EfficientNetB3 model that accurately predicts disease severity, explains predictions with Grad-CAM, and provides real-time results through a Streamlit web application.

**Specific Objectives:**
1. Design an image preprocessing pipeline using **black border removal, green-channel extraction, CLAHE enhancement**, and image resizing (300×300).
2. Develop a **CBAM-enhanced EfficientNetB3** model with channel and spatial attention for improved retinal feature extraction.
3. Implement an **ordinal-aware hybrid loss function** combining Focal Loss and an ordinal penalty to handle class imbalance and preserve DR severity order.
4. Train the model using a **two-phase transfer learning strategy** with warm-up training and fine-tuning of the top EfficientNetB3 layers.
5. Improve minority-grade classification using **data augmentation, class weighting, and ordinal-aware focal loss**.
6. Generate **Grad-CAM heatmaps** to visualize retinal regions influencing the model's predictions.
7. Deploy the trained model as a **Streamlit web application** for real-time DR severity prediction, confidence scores, risk assessment, and clinical recommendations.
8. Evaluate the model using **Quadratic Weighted Kappa (QWK), Accuracy, Precision, Recall, F1-score, and Confusion Matrix**.

**What Makes This Different:**
- First application of a combined Focal Loss + MSE ordinal penalty on APTOS 2019.
- CBAM integrated into EfficientNetB3 (300×300) rather than the smaller B0 (224×224) used in comparable studies.
- Five grade-specific Grad-CAM heatmaps (rather than generic/black-box outputs).
- Three-pronged imbalance mitigation strategy (no prior reviewed study combines all three).
- A fully deployed clinical decision-support interface — grade, confidence, risk tier, follow-up timeline, and heatmap — not just a bare label.
____________________
## 🏗️ System Architecture
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 1 — Setup & Data                                    │
│                                                                              │
│  • Library Imports (TensorFlow, OpenCV, Scikit-learn, GPU+seed setup)         │
│  • Load APTOS 2019 Dataset (3,662 Images)                                    │
│  • Stratified Dataset Split (70% Train | 15% Validation | 15% Test)          │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                Phase 2 — Preprocessing & Data Pipeline                       │
│                                                                              │
│  • Green Channel Extraction                                                  │
│  • CLAHE Image Enhancement                                                   │
│  • Black Border Cropping                                                     │
│  • Resize Images (300 × 300)                                                 │
│  • tf.data pipeline + Data Augmentation                                      │
│  • EfficientNet preprocess_input()                                           │
│  • Class Weight(0:0.6 ... 4:2.2)                                             │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Phase 3 — Model Architecture                                │
│                                                                              │
│  • CBAM Attention Module                                                     │
│      ├── Channel Attention                                                   │
│      └── Spatial Attention                                                   │
│                                                                              │
│  • Ordinal-Aware Loss                                                        │
│      ├── Focal Loss +                                                        │
│      └── MSE Ordinal γ=2.0,α=0.25,λ=0.5                                      │
│                                                                              │
│  • EfficientNet-B3 Backbone                                                  │
│      ├── Frozen ImageNet Weights                                             │
│      ├── CBAM Integration                                                    │
│      └── 5-Class Softmax Output                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 4 — Model Training                                  │
│                                                                              │
│  Phase 1 : Warm-up Training                                                  │
│      • Frozen Backbone                                                       │
│      • 10 Epochs                                                             │
│      • Adam Optimizer                                                        │
│      • Early Stopping                                                        │
│                                                                              │
│  Phase 2 : Fine-Tuning                                                       │
│      • Unfreeze Top 120 Layers                                               │
│      • 40 Epochs                                                             │
│      • Cosine Learning Rate Decay                                            │
│      • Save Best Model (dr_model.keras)                                      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 5 — Model Evaluation                                │
│                                                                              │
│  • Training & Validation Curves                                              │
│  • Accuracy & Loss                                                           │
│  • Test Evaluation                                                           │
│  • Quadratic Weighted Kappa (QWK)                                            │
│  • Classification Report                                                     │
│  • Confusion Matrix                                                          |
|  • Final Metrix                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                Phase 6 — Explainability (Grad-CAM)                           │
│                                                                              │
│  • Load Trained Model                                                        │
│  • Generate Heatmaps                                                         │
│  • Visualize Important Retina Regions                                        │
│  • Save Heatmaps for Grades 0–4                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Phase 7 — Save & Export                                  │
│                                                                              │
│  • Save Final Model (.keras)                                                 │
│  • Save Model Weights (.weights.h5)                                          │
│  • Save Training Summary (JSON)                                              │
│  • Export Confusion Matrix                                                   │
│  • Export Grad-CAM Images                                                    │
│  • Export Performance Metrics                                                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│             Phase 8 — Streamlit Web Application                              │
│                                                                              │
│  Upload Fundus Image                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  CLAHE Preprocessing                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  EfficientNet-B3 + CBAM Prediction                                           │
│          │                                                                   │
│          ▼                                                                   │
│  DR Grade (0–4) + Confidence Score                                           │
│          │                                                                   │
│          ▼                                                                   │
│  Grad-CAM Heatmap + Risk Level + Clinical Recommendation                     │
└──────────────────────────────────────────────────────────────────────────────┘
```
## 📊 Dataset
 
- **Source:** [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **Size:** 3,662 retinal fundus images, 5 severity grades
- **Split:** Stratified 70% train / 15% val / 15% test
  
| Grade | Class | Count | % | Train | Val / Test |
|:---:|---|---:|---:|---:|---:|
| 0 | No DR | 1,805 | 49.3% | 1,264 | 271 / 271 |
| 1 | Mild NPDR | 370 | 10.1% | 259 | 56 / 56 |
| 2 | Moderate NPDR | 999 | 27.3% | 699 | 150 / 150 |
| 3 | Severe NPDR | 193 | 5.3% | ~135 | 29 / 29 |
| 4 | Proliferative DR | 295 | 8.1% | ~207 | 44 / 44 |
 
The dataset is heavily imbalanced (Grade 0 is ~9.4× more frequent than Grade 3), which directly motivated the three-pronged imbalance mitigation strategy used during training.
 
---
 
## 🧠 Model Details
 
| Component | Configuration |
|---|---|
| Backbone | EfficientNetB3 (300×300 input), ImageNet pretrained |
| Attention | CBAM (channel + spatial attention) after final conv block |
| Loss | Ordinal Focal Loss = Focal(γ=2.0, α=0.25) + λ·MSE_ordinal (λ=0.5) |
| Class weights | {0: 0.6, 1: 1.8, 2: 1.0, 3: 2.5, 4: 2.2} |
| Training | 2-phase curriculum — frozen warm-up (10 epochs) + fine-tune top 120 layers (40 epochs) |
| Optimizer | Adam (Phase 1: LR=1e-3, Phase 2: Cosine Decay 5e-5 → 1e-6) |
| Params | ~12M |
| Explainability | Grad-CAM via TensorFlow GradientTape (split backbone/head models) |
 
---
 
## 📈 Results
 
| Metric | Value |
|---|---:|
| Test Accuracy | **82.55%** |
| Quadratic Weighted Kappa (QWK) | **0.9035** |
| Validation Accuracy (best) | ~82–83% |
 
**Per-Class Performance (Test Set, n=550):**
 
| Grade | Class | Precision | Recall | F1-Score | Support |
|:---:|---|---:|---:|---:|---:|
| 0 | No DR | 0.97 | 0.98 | 0.97 | 271 |
| 1 | Mild NPDR | 0.61 | 0.50 | 0.55 | 56 |
| 2 | Moderate NPDR | 0.75 | 0.85 | 0.79 | 150 |
| 3 | Severe NPDR | 0.48 | 0.34 | 0.40 | 29 |
| 4 | Proliferative DR | 0.59 | 0.50 | 0.54 | 44 |
| **Macro Avg** | — | **0.68** | **0.64** | **0.65** | 550 |
| **Weighted Avg** | — | **0.82** | **0.83** | **0.82** | 550 |

## 🔍 Explainability (Grad-CAM)
 
Grade-specific Grad-CAM heatmaps confirm the model attends to pathologically meaningful regions:
- **Grade 0:** diffuse, low-intensity activation (no lesion focus).
- **Grades 1–2:** activation around optic disc periphery / vessel arcades (early microaneurysms, haemorrhages).
- **Grades 3–4:** elevated, widespread activation over haemorrhage clusters, venous beading, and neovascularization near the disc margin.

## 💻 Streamlit Clinical Application
 
The trained model is deployed as a local clinical decision-support web app. For each uploaded fundus image, it returns:
 
1. Predicted DR grade (with clinical name)
2. Confidence score & per-grade probability bars
3. Risk stratification tier (None / Low / Moderate / High / Critical)
4. Recommended follow-up interval
5. Grad-CAM heatmap overlay
6. Clinical description + medical disclaimer

**Upload an Model Info :**
<img width="2076" height="2532" alt="Upload   model_info" src="https://github.com/user-attachments/assets/2cb6ea7a-51ef-4a18-ae77-2120d724a50d" />

**Prediction and GradCam Explainability:**
<img width="1555" height="2540" alt="Prediction   Gradcam Explanability" src="https://github.com/user-attachments/assets/12a73f76-784c-430d-a93f-85a37b546ad8" />

## 📁 Project Structure
 
```
Diabetic-Retinopathy-Detection/
├── .streamlit/
│   └── config.toml
├── models/
│   └── dr_model.keras
├── results of Train_model/
│   ├── 1. class_distribution.png
│   ├── 2. preprocessing_comparison.png
│   ├── 3. Train_accuracy_curve.png
│   ├── 4. val_accuracy_curve.png
│   ├── 5. final_metrics.txt
│   ├── 6. confusion_matrix.png
│   └── 7. gradcam_per_grade.png
├── Streamlit_Screenshot/
│   ├── Prediction & Gradcam Explainability.png
│   └── Upload & model_info.png
├── utils/
│   ├── __init__.py
│   ├── gradcam.py          # Split backbone/head Grad-CAM generation
│   ├── inference.py        # Model loading + prediction logic
│   ├── preprocessing.py    # CLAHE + green-channel preprocessing
│   └── report.py           # Risk tier & clinical recommendation logic
├── app.py                  # Streamlit application entry point
├── dr-detection-model.ipynb # Full training pipeline (Kaggle notebook)
├── requirements.txt
├── test.py
├── .gitignore
└── README.md
```
 
---
## ⚙️ Installation
 
```bash
# Clone the repository
git clone https://github.com/Rohit-Bansode/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection

## Download Python version 3.12
 
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
```
 
---
 
## 🚀 Usage
 
**Run the Streamlit clinical app:**
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`), upload a retinal fundus image (PNG/JPG/JPEG), and view the predicted grade, confidence, risk tier, and Grad-CAM overlay.
 
**Train the model from scratch:**
Open and run `dr-detection-model.ipynb` (built for Kaggle with GPU) — covers data loading, preprocessing, CBAM-EfficientNetB3 construction, two-phase training, evaluation, and Grad-CAM generation.
