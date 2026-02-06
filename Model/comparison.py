import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef
)

from model.logistic import train_evaluate as log
from model.decision_tree import train_evaluate as dt
from model.knn import train_evaluate as knn
from model.naive_bayes import train_evaluate as nb
from model.random_forest import train_evaluate as rf
from model.xgboost_model import train_evaluate as xgb


def run_model_comparison(data: pd.DataFrame, show_roc: bool = True):
    """
    Runs comparison across all classifiers with proper preprocessing.
    Displays ROC curves only if show_roc=True.
    Returns results DataFrame and ROC data.
    """

    data = data.copy()

    # --------------------------------------------------
    # Data Preprocessing
    # --------------------------------------------------
    data = data.drop(columns=["student_id", "salary_package_lpa"], errors="ignore")

    data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
    data["extracurricular_activities"] = data["extracurricular_activities"].map(
        {"Yes": 1, "No": 0}
    )

    X = data.drop("placement_status", axis=1)
    y = data["placement_status"]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    models = {
        "Logistic Regression": log,
        "Decision Tree": dt,
        "KNN": knn,
        "Naive Bayes": nb,
        "Random Forest": rf,
        "XGBoost": xgb
    }

    results = []
    roc_data = {}

    # --------------------------------------------------
    # Model Evaluation
    # --------------------------------------------------
    for name, train_func in models.items():

        model, _ = train_func(X_train, X_test, y_train, y_test)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "ROC-AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

    # --------------------------------------------------
    # ROC Curve Plot
    # --------------------------------------------------
    if show_roc:
        st.subheader("ROC Curve Comparison")

        fig, ax = plt.subplots(figsize=(8, 6))

        for model_name, (fpr, tpr, auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves for All Models")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

    return results_df, roc_data
