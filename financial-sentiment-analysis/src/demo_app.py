"""
Simple Streamlit demo for Financial Sentiment Analysis.

Run:
    streamlit run src/demo_app.py
"""

import streamlit as st
from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="centered"
)

st.title("Financial Sentiment Analysis")
st.write(
    "This demo classifies financial text as positive, neutral, or negative "
    "using a Transformer-based financial language model."
)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

classifier = load_model()

user_input = st.text_area(
    "Enter a financial headline or short financial text:",
    "The company reported stronger-than-expected quarterly earnings."
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = classifier(user_input)[0]
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {result['score']:.4f}")
    else:
        st.warning("Please enter some text.")
