"""
SWYNEX Model Improvement
Dataset: Breast Cancer Wisconsin (Diagnostic), loaded from scikit-learn.
Compares a baseline Logistic Regression pipeline with a tuned Random Forest.
All preprocessing is fitted inside pipelines to avoid data leakage.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 1. Load data and inspect its shape/classes.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    print(f"Dataset shape: {X.shape}")
    print("Class counts (0=malignant, 1=benign):")
    print(y.value_counts().sort_index().to_string())

    # 2. Keep a holdout test set untouched until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Baseline model: scaling + Logistic Regression.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    baseline_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_cv_scores = cross_val_score(
        baseline, X_train, y_train, cv=baseline_cv, scoring="f1"
    )
    baseline.fit(X_train, y_train)

    # 4. Improved model: tune Random Forest hyperparameters using CV.
    improved_pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    param_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }
    search = GridSearchCV(
        estimator=improved_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=baseline_cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    improved = search.best_estimator_

    # 5. Evaluate both models on the same untouched test set.
    models = {"Baseline Logistic Regression": baseline, "Improved Random Forest": improved}
    rows = []
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        predictions[name] = (pred, proba)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        })

    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(OUTPUTS / "model_comparison.csv")
    print("\nTest-set comparison:")
    print(results.round(4).to_string())
    print("\nBest Random Forest parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print(f"\nBaseline 5-fold CV F1: {baseline_cv_scores.mean():.4f} "
          f"+/- {baseline_cv_scores.std():.4f}")
    print(f"Best tuned CV F1: {search.best_score_:.4f}")

    # 6. Save the improved model and a text report.
    joblib.dump(improved, OUTPUTS / "improved_model.joblib")
    best_pred = improved.predict(X_test)
    report = classification_report(
        y_test, best_pred, target_names=dataset.target_names, zero_division=0
    )
    (OUTPUTS / "classification_report.txt").write_text(report, encoding="utf-8")
    (OUTPUTS / "best_parameters.json").write_text(
        json.dumps(search.best_params_, indent=2), encoding="utf-8"
    )

    # 7. Visualizations for the repository and LinkedIn post.
    ax = results[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6), ylim=(0, 1.05)
    )
    ax.set_title("Baseline vs Improved Model — Test Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model_comparison.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (name, model), ax in zip(models.items(), axes):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=dataset.target_names,
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrices.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves — Holdout Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "roc_curves.png", dpi=160)
    plt.close()

    # Random Forest feature importance.
    importances = pd.Series(
        improved.named_steps["model"].feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(15).sort_values()
    ax = importances.plot(kind="barh", figsize=(9, 6))
    ax.set_title("Top 15 Feature Importances — Tuned Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=160)
    plt.close()

    print(f"\nSaved model, metrics, reports, and plots in: {OUTPUTS}")
    print("\nImportant: use the printed test metrics in your report; do not claim "
          "improvement unless the selected metric actually improves.")


if __name__ == "__main__":
    main()
