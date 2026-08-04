# Financial Sentiment Analysis with Transformer Models

## Project Overview

This project focuses on **Financial Sentiment Analysis** using Natural Language Processing (NLP) and Transformer-based models.  
The goal is to classify financial text, such as market headlines or short financial statements, into three sentiment categories:

- **Positive**
- **Neutral**
- **Negative**

The project compares a traditional Machine Learning baseline model with a Transformer-based financial language model.

---

## Author

- David Alonso Dos Santos

---

## Main Objective

To build and evaluate a financial sentiment classification system capable of analyzing financial text and predicting whether the sentiment is positive, neutral, or negative.

---

## Research Question

Can Transformer-based models such as FinBERT classify financial sentiment more accurately than traditional Machine Learning methods?

---

## Dataset

Recommended dataset:

**Financial PhraseBank**

This dataset contains financial sentences labeled as positive, neutral, or negative. It is commonly used for financial sentiment classification tasks.

Possible sources:
- Hugging Face Datasets
- Kaggle
- Academic NLP datasets

---

## Project Pipeline

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Baseline Model
        ↓
FinBERT / Transformer Model
        ↓
Evaluation
        ↓
Comparison
        ↓
Demo
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Transformers
- PyTorch
- Hugging Face Datasets
- Streamlit, optional for demo

---

## Project Structure

```text
financial-sentiment-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_finbert_model.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── baseline_model.py
│   ├── finbert_inference.py
│   ├── evaluation.py
│   └── demo_app.py
│
├── models/
├── results/
├── presentation/
│   └── presentation_outline.md
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the baseline model

```bash
python src/baseline_model.py
```

### 3. Run FinBERT inference demo

```bash
python src/finbert_inference.py
```

### 4. Optional: Run Streamlit demo

```bash
streamlit run src/demo_app.py
```

---

## Expected Results

The project should produce:

- Dataset summary
- Class distribution visualizations
- Baseline model performance
- FinBERT sentiment predictions
- Evaluation metrics
- Confusion matrix
- Final comparison between models

---

## Example Prediction

Input:

```text
The company reported stronger-than-expected quarterly earnings.
```

Output:

```text
Positive
```

---

## Use of AI Tools

AI tools such as GitHub Copilot, Cursor AI, or ChatGPT may be used to support coding, debugging, and documentation.  
Any relevant observations, such as hallucinations, incorrect suggestions, or debugging help, should be mentioned during the final presentation.

---

## Final Presentation

The final presentation will include:

1. Project motivation
2. Dataset description
3. Methodology
4. Baseline model
5. Transformer-based model
6. Results and evaluation
7. Demo
8. Limitations and future improvements
9. Use of AI tools
10. Conclusion
