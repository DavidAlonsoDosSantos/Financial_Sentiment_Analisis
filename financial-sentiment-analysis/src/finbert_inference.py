"""
FinBERT inference demo for Financial Sentiment Analysis.

This script uses a pretrained FinBERT model to classify financial text.
It is designed as a simple demo and does not require training.

Run:
    python src/finbert_inference.py
"""

from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"


def load_finbert_pipeline():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)


def predict_sentiment(text: str):
    classifier = load_finbert_pipeline()
    result = classifier(text)[0]
    return result


def main():
    examples = [
        "The company reported stronger-than-expected quarterly earnings.",
        "The firm warned investors about declining revenue and weaker demand.",
        "The board announced that the annual meeting will take place next month."
    ]

    classifier = load_finbert_pipeline()

    print("FinBERT Financial Sentiment Demo\n")

    for text in examples:
        result = classifier(text)[0]
        print("Text:", text)
        print("Prediction:", result["label"])
        print("Confidence:", round(result["score"], 4))
        print("-" * 70)


if __name__ == "__main__":
    main()
