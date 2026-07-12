# 🩺 Diabetic Retinopathy Severity Classification

## Project Title: 
Automated Diabetic Retinopathy Severity Classification using CBAM-Enhanced EfficientNetB3, Ordinal-Aware Learning, and Grad-CAM Explainability

## Overview :
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

# Goals & Objectives :

# 🏗️ System Architecture

| Phase | Components | Description |
|--------|------------|-------------|
| **Phase 1** | **Setup & Data** | Import required libraries, load the APTOS 2019 dataset, visualize class distribution, and perform a stratified train/validation/test split (70% / 15% / 15%). |
| **Phase 2** | **Preprocessing & Pipeline** | Apply Green-channel CLAHE enhancement, black border removal, resize images to **300 × 300**, create TensorFlow data pipeline, perform data augmentation, preprocess images using EfficientNet, and compute class weights to handle class imbalance. |
| **Phase 3** | **Model Architecture** | Build a **CBAM-Enhanced EfficientNet-B3** model with Channel & Spatial Attention and **Ordinal-Aware Loss** (Focal Loss + Ordinal Loss) for five-grade DR classification. |
| **Phase 4** | **Model Training** | Train the model in two phases: **Phase 1 (Warm-up)** with frozen EfficientNet layers, followed by **Phase 2 (Fine-tuning)** by unfreezing the last layers using Cosine Learning Rate Decay and Early Stopping. |
| **Phase 5** | **Model Evaluation** | Evaluate the trained model using **Accuracy, Quadratic Weighted Kappa (QWK), Classification Report, Confusion Matrix**, and visualize training/validation curves. |
| **Phase 6** | **Explainability (Grad-CAM)** | Generate **Grad-CAM heatmaps** to highlight retinal regions responsible for predictions, improving model interpretability and clinical transparency. |
| **Phase 7** | **Save & Export** | Save the trained model (`dr_model.keras`), model weights, confusion matrix, Grad-CAM outputs, and evaluation metrics for deployment. |
| **Phase 8** | **Streamlit Web Application** | Deploy the trained model as an interactive Streamlit application where users upload a retinal image to receive the predicted DR grade, confidence score, Grad-CAM visualization, risk level, and clinical recommendation. |

## Download the Python Version 3.12

>>>>>>> c4e72f4b6ac380ba70dfd87ba1eb366559975b17
# Create a virtual environment

    python -m venv venv

# Activate the Environment

    venv\Scripts\activate
    
# Install dependencies
    pip install -r requirements.txt

## Download Trained Model

The trained model is **not included** in this repository because it exceeds GitHub's file size limit.

## Download the model from Google Drive:

https://drive.google.com/file/d/1EOkCJ9H37lzMpI8XmsD1x2hB5ULIJV4h/view?usp=sharing

After downloading, place the file here:

models/dr_model.keras

## Run the Application

    streamlit run app.py

