# Breast Cancer Classifier

Compares three classic machine-learning models for classifying breast tumours as malignant or benign, using the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn (569 samples, 30 features).

## Results

5-fold cross-validation on the training set (80%):

| Model | Mean accuracy | Std |
|---|---|---|
| Logistic Regression | 0.980 | 0.013 |
| SVM (RBF) | 0.971 | 0.018 |
| Random Forest | 0.960 | 0.019 |

The best model (Logistic Regression) scores **98% accuracy** on the held-out test set (114 samples), with 0.98 recall on malignant cases.

## How it works
- Stratified 80/20 train/test split, so both sets keep the same class balance
- Feature scaling sits inside each scikit-learn `Pipeline`, so the test data never leaks into training
- Models are compared with cross-validation; only the winner is evaluated once on the test set
- Saves a confusion matrix to `confusion_matrix.png`

## Tech stack
Python · scikit-learn · pandas · matplotlib · pytest · GitHub Actions

## How to run
```bash
pip install -r requirements.txt
python classifier.py
pytest
```
