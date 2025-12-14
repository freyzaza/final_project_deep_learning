# 📧 Indonesian Email Spam Classification using CNN

## 📌 Project Overview

Email spam is one of the most widespread digital threats today.  
Spam messages often contain **phishing links, scam offers, malware**, and **misleading promotions**, which can cause financial loss and security breaches.

This project focuses on building an **Indonesian-language email spam classifier** and comparing two different deep learning approaches.

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
   ├── TF-IDF → CNN
   └── IndoBERT → CNN
   ↓
Evaluation

📂 Project Structure
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
├── .gitignore
└── LICENSE             # MIT License

🚀 Getting Started
🔹 Prerequisites

Python 3.10.9

Anaconda / Miniconda

VS Code (recommended)

📥 Download Anaconda:
👉 https://www.anaconda.com/products/distribution

🧪 Environment Setup (Step-by-Step)
✅ STEP 1 — Create Conda Environment
conda create -n DL_Project python=3.10.9


If prompted, type y to continue.

Activate environment:

conda activate DL_Project

✅ STEP 2 — Install Dependencies (IMPORTANT ORDER)
pip install -r requirements.txt

📄 requirements.txt
# ================================
# CORE DEEP LEARNING
# ================================
tensorflow==2.13.0
keras==2.13.1

# ================================
# NLP / TRANSFORMERS
# ================================
transformers==4.33.3

# ================================
# DATA & ML & DL
# ================================
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
gymnasium==0.28.1

# ================================
# VISUALIZATION
# ================================
matplotlib==3.7.2
seaborn==0.13.2
tqdm==4.66.1

# ================================
# TEXT PREPROCESSING
# ================================
Sastrawi==1.0.1
emoji==2.10.1
typing_extensions==4.5.0

# ================================
# STREAMLIT (FRONTEND)
# ================================
streamlit==1.30.0

# ================================
# JUPYTER / KERNEL (STABLE)
# ================================
ipykernel==6.29.5
ipython==8.12.0
jupyter-client==8.2.0
traitlets==5.9.0

✅ STEP 3 — Register Kernel to VS Code

Open terminal in VS Code (Ctrl + `):

python -m ipykernel install --user --name DL_Project --display-name "DL_Project"

✅ STEP 4 — Restart VS Code

Close VS Code

Reopen VS Code

Select kernel DL_Project

🧪 STEP 5 — Environment Test (MANDATORY)
import tensorflow as tf
import typing_extensions

print(tf.__version__)
print("typing OK")


Expected output:

2.13.0
typing OK


🔥 Environment setup SUCCESSFUL

🏋️ Model Training

Navigate to the training script:

cd src
python train.py


This will:

Train TF-IDF + CNN

Train IndoBERT + CNN

Save trained models to:

outputs/models/

🖥️ Run the Streamlit Application

After training is completed:

cd app
streamlit run app.py


Open browser at:

http://localhost:8501

📊 Evaluation

Evaluation includes:

Accuracy

Precision

Recall

F1-score

Model comparison between:

Traditional NLP features (TF-IDF)

Transformer embeddings (IndoBERT)

📄 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.

🙌 Acknowledgements

IndoBERT — IndoBenchmark

TensorFlow & Keras

Sastrawi Indonesian NLP Library
