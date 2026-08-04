"""
Baseline model for Financial Sentiment Analysis.

This script trains a simple TF-IDF + Logistic Regression model.
It provides a traditional Machine Learning baseline to compare against FinBERT.

Run:
    python src/baseline_model.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score


DATA_PATH = Path("data/processed/financial_phrasebank_processed.csv")
RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python src/data_preparation.py"
        )

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Baseline Model: TF-IDF + Logistic Regression")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    report_path = RESULTS_DIR / "baseline_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Baseline Model: TF-IDF + Logistic Regression\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Baseline Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
