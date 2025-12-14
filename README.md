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

---

### 📦 Clone Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/freyzaza/final_project_deep_learning.git


### 🔹 Prerequisites
```

Make sure you have the following installed:

- **Python 3.10.9**
- **Anaconda / Miniconda**
- **Visual Studio Code** (recommended)

🔗 **Download Anaconda**  
https://www.anaconda.com/products/distribution

---


