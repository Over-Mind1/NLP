import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- Load tokenizer ---
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = json.load(f)
    return tokenizer_from_json(json.dumps(data))

# --- Load model ---
@st.cache_resource
def load_sentiment_model():
    return load_model("review_amazon_sentiment5.h5")

# --- Predict function ---
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_tokens)
    preds = model.predict(padded)
    return np.hstack([preds, 1-preds])  # For LIME binary classifier

# --- Visualize explanation ---
def plot_explanation(exp):
    fig = exp.as_pyplot_figure()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.image(buf.getvalue(), use_container_width=True)

# --- Initialize ---
tokenizer = load_tokenizer()
model = load_sentiment_model()
max_tokens = 166
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

# --- Streamlit UI ---
st.title("Amazon Review Sentiment Analyzer")
user_input = st.text_area("Enter an Amazon product review:")

if st.button("Analyze"):
    if user_input.strip():
        # Predict
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_tokens)
        pred_prob = model.predict(padded)[0][0]
        sentiment = "🟢 Positive" if pred_prob < 0.5 else "🔴 Negative"

        # Show Result
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** {pred_prob:.2f}")

        # Explain with LIME
        with st.spinner("Explaining prediction..."):
            explanation = explainer.explain_instance(user_input, predict_proba, num_features=10)
            st.markdown("### 🔍 Why this prediction?")
            plot_explanation(explanation)
    else:
        st.warning("Please enter some text to analyze.")
