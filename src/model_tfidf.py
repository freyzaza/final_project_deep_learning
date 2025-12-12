import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
MAX_WORDS = 5000
MAX_LEN = 300

# ============================================
# FIT TOKENIZER
# ============================================

def fit_tokenizer(texts):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


# ============================================
# TEXT → SEQUENCE
# ============================================

def to_sequence(tokenizer, texts):
    seq = tokenizer.texts_to_sequences(texts)
    seq = pad_sequences(seq, maxlen=MAX_LEN)
    return seq


# ============================================
# BUILD CNN MODEL
# ============================================

def build_token_cnn(num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
