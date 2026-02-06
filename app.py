import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

from model.logistic import train_evaluate as log_model
from model.decision_tree import train_evaluate as dt_model
from model.knn import train_evaluate as knn_model
from model.naive_bayes import train_evaluate as nb_model
from model.random_forest import train_evaluate as rf_model
from model.xgboost_model import train_evaluate as xgb_model

from model.comparison import run_model_comparison
from model.top_features import get_top_features
from model.prediction import predict_placement


# --------------------------------------------------
# Page Configuration and Header
# --------------------------------------------------
st.set_page_config(page_title="Student Placement Prediction", layout="wide")

st.markdown(
    """
    # Student Placement Prediction Model- Assignment 2
    **Name:** Milan Uniyal  
    **BITS ID:** 2024DC04145  

    **Dataset Source (Kaggle):**  
    🔗 [Student Academic Placement Performance Dataset](https://www.kaggle.com/datasets/suvidyasonawane/student-academic-placement-performance-dataset)
  
    ---
    """
)

# --------------------------------------------------
# Page Sidebar 
# --------------------------------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "",
    ["Single Model Evaluation", "All Model Comparison", "Placement Prediction"]
)

# --------------------------------------------------
# Loading of Dataset
# --------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("data/student_academic_placement_performance_dataset.csv")

# --------------------------------------------------
# Dataset Overview 
# --------------------------------------------------
st.subheader("Dataset Overview")
st.caption("This dataset contains academic, skill-based, and experiential attributes of students.It is used to predict whether a student is likely to be placed based on historical placement outcomes. Basic structural summary of the dataset")

rows, cols = data.shape
missing = data.isnull().sum().sum()

c1, c2, c3 = st.columns(3)
c1.markdown(f"**Rows:** {rows}")
c2.markdown(f"**Columns:** {cols}")
c3.markdown(f"**Missing Values:** {missing}")

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
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model_map = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

VALID_FEATURE_MODELS = ["Decision Tree", "Random Forest", "XGBoost"]

# ==================================================
#  Single Model Evaluation
# ==================================================
if section == "Single Model Evaluation":

    st.subheader("Select Model")
    model_name = st.selectbox("", list(model_map.keys()))

    show_features = st.checkbox("Show Feature Importance")

    model, metrics = model_map[model_name](X_train, X_test, y_train, y_test)

    st.markdown(f"Evaluation – {model_name}")

    y_pred = model.predict(X_test)

    for k, v in metrics.items():
        st.write(f"**{k}:** {v:.4f}")

    # MCC
    mcc = matthews_corrcoef(y_test, y_pred)
    st.write(f"**MCC Score:** {mcc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        cbar=False, annot_kws={"size": 10}, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title("Confusion Matrix", fontsize=10)
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    if show_features:
        if model_name in VALID_FEATURE_MODELS:
            st.markdown("---")
            st.subheader("Feature Importance (Top 5)")
            importance = get_top_features(
                data, target_col="placement_status", top_n=5
            )
            st.dataframe(importance)
        else:
            st.info(
                "Feature importance is available only for "
                "Decision Tree, Random Forest, and XGBoost."
            )

# ==================================================
# Model Comparison
# ==================================================
elif section == "All Model Comparison":

    st.subheader("Model Comparison")

    show_roc = st.checkbox("Show ROC Curves", value=True)

    results_df, _ = run_model_comparison(data, show_roc=show_roc)

    st.subheader("Model Comparison Summary")
    st.dataframe(results_df.round(4))

# ==================================================
# Placement Prediction
# ==================================================
elif section == "Placement Prediction":

    st.subheader("Placement Prediction")

    model_name = st.selectbox("Select Model", list(model_map.keys()))
    model, _ = model_map[model_name](X_train, X_test, y_train, y_test)

    with st.form("prediction_form"):

        st.subheader("Student Details")
        student_name = st.text_input("Student Name")
        college = st.text_input("College")
        branch = st.selectbox(
            "Branch", ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil", "Other"]
        )

        st.subheader("Academic & Skill Inputs")
        cgpa = st.slider("CGPA (0–10)", 0.0, 10.0, 5.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            entrance = st.slider("Entrance Exam Score", 0, 100, 50)
            internships = st.number_input("Internships", 0, 5, 1)
        with col2:
            technical = st.slider("Technical Skill Score", 0, 100, 50)
            projects = st.number_input("Live Projects", 0, 10, 2)
        with col3:
            soft = st.slider("Soft Skill Score", 0, 100, 50)
            attendance = st.slider("Attendance (%)", 50, 100, 70)

        backlogs = st.number_input("Backlogs", 0, 10, 0)

        submit = st.form_submit_button("Predict Placement")

    if submit:
        input_data = {
            "ssc_percentage": cgpa * 10,
            "hsc_percentage": cgpa * 10,
            "degree_percentage": cgpa * 10,
            "entrance_exam_score": entrance,
            "technical_skill_score": technical,
            "soft_skill_score": soft,
            "internship_count": internships,
            "live_projects": projects,
            "attendance_percentage": attendance,
            "backlogs": backlogs,
            "work_experience_months": 0,
            "gender": 1,
            "extracurricular_activities": 1
        }

        pred, prob = predict_placement(
            model, scaler, X.columns, input_data, X
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        st.write(f"**Name:** {student_name}")
        st.write(f"**College:** {college}")
        st.write(f"**Branch:** {branch}")

        if pred == 1:
            st.success("Likely to be PLACED")
        else:
            st.error("Likely to be NOT PLACED")

        if prob is not None:
            confidence_pct = round(prob * 100, 1)

            st.subheader("Prediction Confidence")
            st.write(f"**Probability of Placement:** {confidence_pct}%")
            st.progress(int(confidence_pct))
