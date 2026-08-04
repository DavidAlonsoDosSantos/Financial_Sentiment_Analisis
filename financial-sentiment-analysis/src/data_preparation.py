"""
Data preparation script for Financial Sentiment Analysis.

This script loads the Financial PhraseBank dataset from Hugging Face,
renames the columns, maps labels to readable sentiment names, and saves
a processed CSV file.

Run:
    python src/data_preparation.py
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset


OUTPUT_PATH = Path("data/processed/financial_phrasebank_processed.csv")


def load_financial_phrasebank() -> pd.DataFrame:
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    df = dataset["train"].to_pandas()

    # Expected columns usually include: sentence, label
    df = df.rename(columns={"sentence": "text"})

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    df["sentiment"] = df["label"].map(label_map)
    df = df[["text", "label", "sentiment"]]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates(subset=["text"])
    df["text"] = df["text"].str.strip()
    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_financial_phrasebank()
    df = clean_data(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Processed dataset saved to:", OUTPUT_PATH)
    print(df.head())
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
