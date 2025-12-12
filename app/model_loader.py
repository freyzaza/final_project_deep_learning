import os
import re
import emoji
import joblib
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ======================================================================
# BASE DIRECTORY
# ======================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

# === TF-IDF paths ===
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, "model_tfidf_cnn.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# === BERT paths ===
BERT_TOKENIZER_DIR = os.path.join(MODEL_DIR, "bert_tokenizer")
BERT_MODEL_DIR = os.path.join(MODEL_DIR, "model_bert_cnn")   # <-- FOLDER, NOT FILE


# ======================================================================
# TF-IDF LOADING
# ======================================================================
def load_tfidf_model():
    tokenizer = joblib.load(os.path.join(MODEL_DIR, "tokenizer.pkl"))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_tfidf_cnn.h5"))
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, model, label_encoder



# ======================================================================
# BERT LOADING
# ======================================================================
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_DIR)
    bert_model = tf.keras.models.load_model(BERT_MODEL_DIR, compile=False)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, bert_model, label_encoder



# ======================================================================
# CLEANING FUNCTION
# ======================================================================
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
STOPWORDS.update({
    "hou","kaminski","vince","enron","corp","edu","cc","re","fw",
    "subject","email","houston","pm","am","com","net","org",
    "ltd","co","inc","ect"
})

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
NON_ALPHA = re.compile(r'[^a-zA-Z\s]')

if hasattr(emoji, "get_emoji_regexp"):
    strip_emoji = lambda txt: emoji.get_emoji_regexp().sub(" ", txt)
else:
    strip_emoji = lambda txt: emoji.replace_emoji(txt, replace=" ")


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = strip_emoji(text)
    text = NON_ALPHA.sub(" ", text)

    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2 and t.isalpha()]
    return " ".join(tokens)


# ======================================================================
# TF-IDF PREDICT
# ======================================================================
def predict_tfidf(text, tokenizer, model, label_encoder):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    seq = pad_sequences(seq, maxlen=300)

    pred = model.predict(seq).argmax(axis=1)[0]
    return label_encoder.inverse_transform([pred])[0]


# ======================================================================
# BERT PREDICT
# ======================================================================
MAX_LEN = 128

def predict_bert(text, tokenizer, model, label_encoder):
    clean = clean_text(text)

    enc = tokenizer(
        [clean],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf"
    )

    pred = model.predict([
        enc["input_ids"],
        enc["attention_mask"]
    ]).argmax(axis=1)[0]

    return label_encoder.inverse_transform([pred])[0]

