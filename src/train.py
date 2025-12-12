import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import save_cleaned_dataset
from data_loader import load_data, split_data
from model_tfidf import fit_tokenizer, to_sequence, build_token_cnn
from model_bert import encode_bert, build_bert_cnn
from evaluate import evaluate_model

import numpy as np
import os
from transformers import BertTokenizer

# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


# ============================================================
# LOAD & CLEAN DATA
# ============================================================

df = load_data("data/raw/email_spam_indo.csv")
save_cleaned_dataset(df, "Pesan", "data/processed/email_spam_clean.csv")

df = pd.read_csv("data/processed/email_spam_clean.csv")


# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["Kategori"])
labels = le.classes_

# 🔥 SAVE LABEL ENCODER (WAJIB UNTUK STREAMLIT)
joblib.dump(le, "outputs/models/label_encoder.pkl")


# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)


# ============================================================
# TF-IDF PIPELINE
# ============================================================

from model_tfidf import fit_tokenizer, to_sequence, build_token_cnn

# FIT TOKENIZER
tokenizer = fit_tokenizer(X_train)

# Convert teks → sequence
X_train_seq = to_sequence(tokenizer, X_train)
X_val_seq = to_sequence(tokenizer, X_val)
X_test_seq = to_sequence(tokenizer, X_test)

# Save tokenizer
import joblib
joblib.dump(tokenizer, "outputs/models/tokenizer.pkl")

# Build model
model_tfidf = build_token_cnn(num_classes=len(labels))

history = model_tfidf.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)

model_tfidf.save("outputs/models/model_tfidf_cnn.h5")

# Evaluate + save plots
pred_tfidf = model_tfidf.predict(X_test_seq).argmax(axis=1)
evaluate_model("TF-IDF + CNN", y_test, pred_tfidf, labels)


# ============================================================
# BERT PIPELINE
# ============================================================

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# 🔥 SAVE TOKENIZER (WAJIB)
tokenizer.save_pretrained("outputs/models/bert_tokenizer")

train_enc = encode_bert(X_train)
val_enc = encode_bert(X_val)
test_enc = encode_bert(X_test)

model_bert = build_bert_cnn(num_classes=len(labels))
history_bert = model_bert.fit(
    [train_enc["input_ids"], train_enc["attention_mask"]],
    y_train,
    validation_data=(
        [val_enc["input_ids"], val_enc["attention_mask"]],
        y_val
    ),
    epochs=3,
    batch_size=16
)

# SAVE BERT MODEL (format folder)
model_bert.save("outputs/models/model_bert_cnn")

# Evaluate + save plots
pred_bert = model_bert.predict([
    test_enc["input_ids"], test_enc["attention_mask"]
]).argmax(axis=1)

evaluate_model("IndoBERT + CNN", y_test, pred_bert, labels)
