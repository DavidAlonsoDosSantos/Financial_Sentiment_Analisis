"""
Utility functions for model evaluation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score


def evaluate_model(y_true, y_pred, model_name: str, output_dir: str = "results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred))

    with open(output_path / f"{model_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_confusion_matrix.png")
    plt.show()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }
