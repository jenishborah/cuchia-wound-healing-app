# 🐟 Cuchia Wound Healing Prediction - Deep Learning Project

## Overview
This project focuses on the automatic classification of wound healing images of *Monopterus cuchia* (Cuchia fish).  
The objective is to predict whether a wound healing image is:
- **Normal Repairing** or
- **Retinoic Acid Treated Repairing**.

A deep learning model based on MobileNet was fine-tuned and deployed using Streamlit for easy use.

---

## Dataset
- **Structure**: Augmented_dataset/ normal_wound/ 1hpw/ 3hpw/ ... retonic_acid/ 1hRA/ 3hRA/ ...
- **Total images**: ~3000
- **Image type**: Microscopic color images
- **Image size**: Resized to 224×224 pixels
- **Augmentation**: Applied to improve generalization

---

## Model Details
- **Base Model**: MobileNet (pre-trained on ImageNet)
- **Architecture Modifications**:
- GlobalAveragePooling2D
- Dense layer with 128 units (ReLU activation)
- Dropout layer (0.2)
- Dense output layer with 2 units (Softmax activation)
- **Optimization**:
- Adam optimizer
- Hyperparameter tuning using RandomSearch
- Best hyperparameters:
  - Units: 128
  - Dropout: 0.2
  - Learning Rate: 0.001
- **Loss Function**: Categorical Crossentropy
- **Validation Accuracy**: ~90%+

---

## App Features
- Upload a microscopic wound healing image.
- Predicts if the image belongs to a normal or retinoic acid treated wound.
- Displays the predicted class and confidence percentage.
- Built with **Streamlit** for an intuitive interface.

---

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/your-username/cuchia-wound-healing-app.git
  cd cuchia-wound-healing-app
  ```

2. Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

---

## Running the App Locally

```bash
streamlit run app.py
