# 📧 Indonesian Email Spam Classification using CNN

An end-to-end **Deep Learning project** for **Indonesian-language email spam classification**, comparing classical NLP features with transformer-based embeddings.

This project was developed to analyze the effectiveness and trade-offs between:

- **TF-IDF + Convolutional Neural Network (CNN)**
- **IndoBERT + Convolutional Neural Network (CNN)**

---

## 📌 Project Overview

Email spam remains one of the most widespread digital threats today.  
Spam messages often contain **phishing links**, **scam offers**, **malware**, and **misleading promotions**, which may lead to financial loss and security breaches.

This project focuses on building a robust **Indonesian email spam classifier** using deep learning techniques and conducting a structured comparison between traditional and transformer-based NLP pipelines.

---

## 🎯 Objectives

- Build an email spam classifier for **Indonesian text**
- Compare two modeling pipelines:
  - **TF-IDF + CNN**
  - **IndoBERT + CNN**
- Evaluate and analyze trade-offs between:
  - Classical feature extraction
  - Transformer-based contextual embeddings

---

## 🧠 System Architecture

**Pipeline Overview**

Raw Text  
↓  
Case Folding → Filtering → Emoji Removal → Stopwords Removal → Stemming  
↓  
Tokenization  
↓  
Exploratory Data Analysis (EDA)  
↓  
Train-Test Split (80% / 20%)  
↓  
Feature Extraction  
- TF-IDF → CNN  
- IndoBERT → CNN  
↓  
Evaluation



---

## 📁 Project Structure

```text
deep-learning-final-project/
├── app/                # Streamlit demo application
├── data/               # Raw & processed datasets
├── notebooks/          # EDA & experimentation notebooks
├── src/                # Training, preprocessing & evaluation code
├── config/             # Experiment configuration files
├── outputs/            # Trained models & experiment results
├── report/             # Final project report (PDF)
├── requirements.txt    # Project dependencies
├── README.md           # Main documentation
├── .gitignore          # Ignored files & folders
└── LICENSE             # MIT License
```
---

## 🚀 Getting Started

### 📦 Clone Repository

Clone the repository from GitHub:

```
git clone https://github.com/freyzaza/final_project_deep_learning.git
```

### 🔹 System Requirements

Make sure you have the following installed:

- **Python 3.10.9**
- **Anaconda / Miniconda**
- **Visual Studio Code** (recommended)

🔗 **Download Anaconda**  
https://www.anaconda.com/products/distribution

---

## 🧪 Environment Setup (Step-by-Step)

Follow the steps below to set up the development environment for this project.


### ✅ Step 1 — Create Conda Environment

Create a new Conda environment with the required Python version:

```
conda create -n environment name (example: DL_Project) python=3.10.9
```

### ✅ Step 2 — Install Dependencies (IMPORTANT ORDER)
```
pip install -r requirements.txt
```

### ✅ Step 3 — Register Kernel to VS Code

Open terminal in VS Code (`Ctrl + ``):
```
python -m ipykernel install --user --name DL_Project --display-name "DL_Project"
```

### ✅ Step 4 — Restart VS Code

After registering the kernel, restart Visual Studio Code:

1. **Close VS Code**
2. **Reopen VS Code**
3. Select the kernel: **DL_Project**

### ✅ Step 5 — Environment Test (MANDATORY)
```
import tensorflow as tf
import typing_extensions

print(tf.__version__)
print("typing OK")
```

Expected output:
```
2.13.0
typing OK
```

✅ **If successful → Environment setup COMPLETE**

---
## 🏋️ Model Training

Run the training process from the `src` directory:

```
python train.py
```

This process will:

- Train **TF-IDF + Convolutional Neural Network (CNN)**
- Train **IndoBERT + Convolutional Neural Network (CNN)**
- Save trained models will go to:

```
outputs/models/
```

---
## 🖥️ Run the Streamlit Application
After the training process is completed, run the Streamlit application from the `app` directory:

```
# streamlit run app.py
```
Open the application in your browser:

```
http://localhost:8501 (example)

```
### 📌 Note

- Streamlit does **NOT always open automatically**
- The VS Code terminal will display a local URL
- Use **Ctrl + Click** on the link, or copy it into your browser
---



