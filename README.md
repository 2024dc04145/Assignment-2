# Assignment-2
This assignment uses the Student Academic Placement Performance Dataset from Kaggle to analyze and predict student placement outcomes.

**A.	Problem Description**

This project uses the Student Academic Placement Performance Dataset from Kaggle to analyze and predict student placement outcomes. The dataset contains information related to students’ academic scores, technical skills, internships, certifications, and extracurricular participation.
The model is designed as an interactive Streamlit web application that performs the following tasks:

**1.	Data Exploration**

- Displays dataset overview including number of rows, columns, and missing values.
- Ensures data quality before model training.
  
**2.	Single Model Evaluation**

- Allows the user to select a machine learning classifier (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost).
- Evaluates the selected model using metrics such as Accuracy, Precision, Recall, F1-Score, MCC and ROC-AUC.
- Displays confusion matrix and features for relevant classifiers.
  
**3.	Feature Importance Analysis**

- Identifies and highlights the top 5 most influential features affecting placement outcomes.
- Feature importance is shown only for tree-based models (Decision Tree, Random Forest, XGBoost) to ensure interpretability.
  
**4.	Model Comparison**

- Compares all classifiers on the same dataset using standard performance metrics.
  
**5.	Decision Support**

- Helps institutions understand which factors contribute most to placements.
- Assists in selecting the most effective predictive model.

**6.	Expected Outcome**

- Accurate prediction of student placement status.
- Identification of key academic and extracurricular factors influencing placement.
- Comparative performance analysis of multiple machine learning models.
- An interpretable and user-friendly application suitable for academic and real-world use.

This model predicts whether a student will be placed or not using multiple machine learning classification models and provides a detailed comparative analysis.

**B.	Dataset description**

This dataset contains academic records, skills, internships, and placement outcomes of 5,000 students. It includes educational performance, technical and soft skill scores, work experience, certifications, attendance, and backlogs to analyze factors influencing student placements. The dataset is designed for data analysis, visualization, and machine learning tasks such as placement prediction and salary estimation, making it suitable for EDA, classification, and regression models.

**C.	Observations on Model Performance**
1. Logistic Regression:
Achieves good accuracy (0.895) with high ROC-AUC (0.937), indicating strong class separability. However, lower recall shows it misses some positive cases, reflecting its limitation in modeling complex non-linear patterns.

2. Decision Tree:
Shows perfect scores across all metrics, which strongly indicates overfitting to the dataset. While training performance is excellent, such results may not generalize well to unseen data.

3. k-Nearest Neighbors (kNN):
Provides balanced performance with improved accuracy over Logistic Regression. However, moderate recall suggests sensitivity to local noise and dependence on distance-based decision boundaries.

4. Naive Bayes:
Delivers high precision (0.958) and strong ROC-AUC (0.985), meaning predictions are confident and well separated. Slightly lower recall reflects the restrictive feature independence assumption.


5. Random Forest (Ensemble):
Achieves perfect performance across all metrics, demonstrating the strength of ensemble learning in reducing variance and capturing complex feature interactions. More reliable than a single decision tree.
XG-Boost (Ensemble): Also attains perfect scores, confirming its ability to model non-linear relationships effectively using boosting and regularization. It is the most powerful and robust classifier among those evaluated.
