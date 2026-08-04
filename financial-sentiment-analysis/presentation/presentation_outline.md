# Final Presentation Outline — 10 Minutes

## Slide 1 — Title
**Financial Sentiment Analysis Using Transformer Models**

Group members:
- Morich
- Toko
- Deif

---

## Slide 2 — Motivation

Financial markets react strongly to information.  
News, earnings reports, and company announcements can influence investor expectations.

Our project explores whether NLP models can automatically classify financial text as positive, neutral, or negative.

---

## Slide 3 — Research Question

**Can Transformer-based models classify financial sentiment more effectively than traditional Machine Learning approaches?**

---

## Slide 4 — Dataset

We use a financial sentiment dataset containing short financial sentences labeled as:

- Positive
- Neutral
- Negative

We analyze the class distribution, sentence length, and example texts before training the models.

---

## Slide 5 — Methodology

Pipeline:

1. Dataset collection
2. Data cleaning
3. Exploratory Data Analysis
4. Baseline model
5. FinBERT model
6. Evaluation
7. Demo

---

## Slide 6 — Baseline Model

We implement a traditional Machine Learning baseline:

- TF-IDF vectorization
- Logistic Regression classifier

This provides a reference point for evaluating Transformer-based models.

---

## Slide 7 — Transformer Model

We use FinBERT, a BERT-based model adapted to financial language.

FinBERT is useful because financial sentiment can be domain-specific.  
Words such as “liability”, “risk”, “growth”, or “guidance” may have different meanings in financial contexts.

---

## Slide 8 — Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

We compare the baseline model with FinBERT-based predictions.

---

## Slide 9 — Demo

The user enters a financial headline or short text.

The system returns:

- predicted sentiment
- confidence score

Example:

Input:
> The company reported stronger-than-expected quarterly earnings.

Output:
> Positive sentiment

---

## Slide 10 — Use of AI Tools

We used AI tools to support:

- code structure
- debugging
- documentation
- idea development

We also observed that AI tools may sometimes suggest incorrect imports, outdated libraries, or oversimplified explanations, so manual verification was necessary.

---

## Slide 11 — Limitations

- Dataset size is limited.
- Financial sentiment does not always directly predict stock price movement.
- Market reaction depends on many external factors.
- The model may struggle with sarcasm, context, or very technical statements.

---

## Slide 12 — Conclusion

The project shows how NLP and Transformer models can be applied to financial analysis.  
It provides a practical example of AI usage in finance, investment research, and market data interpretation.
