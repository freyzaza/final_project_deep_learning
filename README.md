# 📧 Indonesian Email Spam Classification using CNN

## 📌 Project Overview

Email spam is one of the most widespread digital threats today.  
Spam messages often contain **phishing links**, **scam offers**, **malware**, and **misleading promotions**, which can cause financial loss and security breaches.

This project focuses on building an **Indonesian-language email spam classifier** and comparing two different deep learning approaches:

- **TF-IDF + Convolutional Neural Network (CNN)**
- **IndoBERT + Convolutional Neural Network (CNN)**

---

## 🎯 Objectives

- Build an email spam classifier for Indonesian text
- Compare two modeling pipelines:
  - **TF-IDF + Convolutional Neural Network (CNN)**
  - **IndoBERT + Convolutional Neural Network (CNN)**
- Evaluate model performance and analyze trade-offs between:
  - Classical feature extraction
  - Transformer-based embeddings

---

## 🧠 System Architecture

### Pipeline Overview

```text
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
  ├── TF-IDF   → CNN
  └── IndoBERT → CNN
  ↓
Evaluation

📁 Project Structure
deep-learning-final-project/
├── app/                # Streamlit demo application
├── data/               # Raw & processed datasets
├── notebooks/          # EDA & experimentation notebooks
├── src/                # Training, preprocessing & evaluation code
├── config/             # Experiment configuration files
├── outputs/            # Trained models & experiment results
├── report/             # Final project report (PDF)
├── presentation.pdf    # Project presentation slides
├── requirements.txt    # Project dependencies
├── README.md           # Main documentation
├── .gitignore          # Ignored files & folders
└── LICENSE             # MIT License

🚀 Getting Started
🔹 Prerequisites

Python 3.10.9

Anaconda / Miniconda

Visual Studio Code (recommended)

📥 Download Anaconda:
https://www.anaconda.com/products/distribution

🧪 Environment Setup (Step-by-Step)
✅ Step 1 — Create Conda Environment
conda create -n DL_Project python=3.10.9
