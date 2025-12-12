import tensorflow as tf
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel

MAX_LEN = 128

# Load backbone once
bert_backbone = TFBertModel.from_pretrained("indobenchmark/indobert-base-p1")
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")


# ============================================================
# BERT ENCODING (dipakai di training)
# ============================================================
def encode_bert(texts):
    """Encode list/Series of texts menjadi input BERT."""
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )


# ============================================================
# BERT + CNN MODEL
# ============================================================
def build_bert_cnn(num_classes):
    # input tensors
    input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

    # backbone BERT
    bert_output = bert_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask
    )[0]  # shape: (batch, MAX_LEN, 768)

    # CNN Head
    x = layers.Conv1D(128, 3, activation='relu')(bert_output)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# EXPORT tokenizer (dipakai Streamlit)
# ============================================================
def get_tokenizer():
    return tokenizer
