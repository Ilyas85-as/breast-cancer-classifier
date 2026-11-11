"""Compare classic ML models on the Breast Cancer Wisconsin dataset."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_data():
    """Return features X and labels y (0 = malignant, 1 = benign)."""
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def build_models():
    """Models to compare. Scaling is inside each pipeline to avoid data leakage."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def compare_models(X, y, cv=5):
    """5-fold cross-validated accuracy for each model, best first."""
    rows = []
    for name, model in build_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        rows.append({"model": name, "mean_accuracy": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = compare_models(X_train, y_train)
    print("Cross-validation (training set):")
    print(results.to_string(index=False, float_format="%.3f"))

    best_name = results.iloc[0]["model"]
    best_model = build_models()[best_name].fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nBest model on held-out test set: {best_name}")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["malignant", "benign"], cmap="Blues"
    )
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
