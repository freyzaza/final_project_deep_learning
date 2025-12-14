📧 Indonesian Email Spam Classification using CNN
📌 Project Overview

Email spam is one of the most widespread digital threats today.
Spam messages often contain phishing links, scam offers, malware, and misleading promotions, which can cause financial loss and security breaches.

This project focuses on building an Indonesian-language email spam classifier and comparing two different deep learning approaches:

TF-IDF + Convolutional Neural Network (CNN)

IndoBERT + Convolutional Neural Network (CNN)

🎯 Objectives

Build an email spam classifier for Indonesian text

Compare two modeling pipelines:

TF-IDF + CNN

IndoBERT + CNN

Evaluate model performance and analyze trade-offs between:

Classical feature extraction

Transformer-based embeddings

🧠 System Architecture
Pipeline Overview
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


⬆️ INI AMAN 100% buat GitHub (pakai ```text)

🚀 Getting Started
🔹 Prerequisites

Python 3.10.9

Anaconda / Miniconda

Visual Studio Code (recommended)

🔗 Download Anaconda:
https://www.anaconda.com/products/distribution

🧪 Environment Setup (Step-by-Step)
✅ Step 1 — Create Conda Environment
conda create -n DL_Project python=3.10.9


If prompted, type y.

Activate environment:

conda activate DL_Project

✅ Step 2 — Install Dependencies (IMPORTANT ORDER)
pip install -r requirements.txt

✅ Step 3 — Register Kernel to VS Code

Open terminal in VS Code (`Ctrl + ``):

python -m ipykernel install --user --name DL_Project --display-name "DL_Project"

✅ Step 4 — Restart VS Code

Close VS Code

Reopen VS Code

Select kernel: DL_Project

✅ Step 5 — Environment Test (MANDATORY)
import tensorflow as tf
import typing_extensions

print(tf.__version__)
print("typing OK")


Expected output:

2.13.0
typing OK


✅ If successful → Environment setup COMPLETE

🏋️ Model Training

Run training from the src folder:

cd src
python train.py


This will:

Train TF-IDF + CNN

Train IndoBERT + CNN

Save models to:

outputs/models/

🖥️ Run the Streamlit Application

After training is completed:

cd app
streamlit run app.py


Open in browser:

http://localhost:8501


📌 Note:
Streamlit does NOT always open automatically.
VS Code terminal will show a link → Ctrl + Click or copy to browser.

📊 Evaluation

Evaluation metrics include:

Accuracy

Precision

Recall

F1-score

Model comparison:

TF-IDF (traditional NLP features)

IndoBERT (transformer-based embeddings)

📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project with proper attribution.
