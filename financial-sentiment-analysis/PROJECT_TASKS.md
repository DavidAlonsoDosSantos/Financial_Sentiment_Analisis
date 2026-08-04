# Project Tasks and Work Distribution

## Group Members

- Morich
- Toko
- Deif

---

# Morich — Data & EDA

## Responsibilities

Morich is responsible for understanding, preparing, and explaining the dataset.

## Tasks

- Find and confirm the final dataset
- Load the dataset
- Clean missing values and duplicates
- Analyze class distribution
- Analyze text length
- Create EDA visualizations
- Prepare examples of positive, neutral, and negative texts
- Explain dataset limitations in the presentation

## Deliverables

- `notebooks/01_eda.ipynb`
- Dataset summary
- EDA plots
- Dataset explanation for presentation

---

# Toko — Model Development

## Responsibilities

Toko is responsible for implementing the classification models.

## Tasks

- Prepare the ML environment
- Implement TF-IDF + Logistic Regression baseline
- Implement FinBERT inference
- Test predictions on example financial headlines
- Compare model behavior
- Save relevant model outputs

## Deliverables

- `notebooks/02_baseline_model.ipynb`
- `notebooks/03_finbert_model.ipynb`
- `src/baseline_model.py`
- `src/finbert_inference.py`

---

# Deif — Evaluation, Integration & Demo

## Responsibilities

Deif is responsible for connecting the project parts and preparing the final presentation/demo.

## Tasks

- Define evaluation metrics
- Create confusion matrix
- Compare baseline and FinBERT results
- Prepare Streamlit demo
- Write README
- Prepare presentation structure
- Explain how AI tools were used
- Coordinate final GitHub repository

## Deliverables

- `src/evaluation.py`
- `src/demo_app.py`
- `README.md`
- `presentation/presentation_outline.md`
- Final demo

---

# First Milestone

The first working version should include:

- Dataset loaded
- EDA completed
- Baseline model working
- FinBERT demo working
- README explaining the project

---

# Final Milestone

The final version should include:

- Clean GitHub repository
- Model comparison
- Evaluation metrics
- Visualizations
- Demo
- Presentation
