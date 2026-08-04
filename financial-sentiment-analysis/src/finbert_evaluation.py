import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_PATH = "data/processed/financial_phrasebank_processed.csv"


def label_to_int(label):
    mapping = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
    }
    return mapping[label]


def main():

    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading FinBERT...")

    classifier = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

    predictions = []

    total = len(X_test)

    for i, text in enumerate(X_test):

        result = classifier(text[:512])[0]

        pred = label_to_int(result["label"].lower())

        predictions.append(pred)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{total}")

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print("\nFINBERT RESULTS")
    print("-" * 40)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "negative",
                "neutral",
                "positive"
            ]
        )
    )


if __name__ == "__main__":
    main()