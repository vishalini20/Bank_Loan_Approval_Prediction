Bank Loan Approval Prediction
Project Overview

The Bank Loan Approval Prediction project aims to predict whether a loan application will be approved or rejected using Machine Learning techniques. The project includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and comparison of multiple classification algorithms.

Objective

To build a predictive model that helps financial institutions determine loan approval status based on applicant information such as income, credit history, loan amount, and other factors.

Dataset

The dataset contains applicant details including:

Gender
Married Status
Dependents
Education
Self Employed
Applicant Income
Coapplicant Income
Loan Amount
Loan Amount Term
Credit History
Property Area
Loan Status (Target Variable)
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Data Preprocessing
Handled missing values using mode and median imputation.
Removed unnecessary columns such as Loan_ID.
Encoded categorical variables using Label Encoding.
Split data into training and testing sets.
Exploratory Data Analysis (EDA)

The following visualizations were created:

Gender vs Loan Status
Loan Approval Percentage (Pie Chart)
Loan Amount Distribution
Income vs Loan Amount
Correlation Heatmap
Credit History vs Loan Approval
Machine Learning Models Used
Logistic Regression

A statistical classification algorithm used as a baseline model.

Decision Tree Classifier

A tree-based model that makes predictions using decision rules.

Random Forest Classifier

An ensemble learning technique that combines multiple decision trees.

K-Nearest Neighbors (KNN)

A distance-based classification algorithm.

Support Vector Machine (SVM)

A supervised learning algorithm used for classification tasks.

Model Evaluation

Models were evaluated using:

Accuracy Score
Classification Report
Confusion Matrix
Model Comparison
Model	Accuracy
SVM	79.67%
Logistic Regression	78.86%
Random Forest	75.61%
Decision Tree	69.11%
KNN	57.72%
Best Performing Model

Support Vector Machine (SVM) achieved the highest accuracy of approximately 79.67%.

Visualizations

Project screenshots are available in the screenshots folder:

gender_vs_loan_status.png
loan_approval_pie.png
loan_amount_distribution.png
income_vs_loan_amount.png
correlation_heatmap.png
credit_history_vs_loan.png
model_comparison.png
confusion_matrix.png
Project Structure
Bank_Loan_Approval_Prediction/
│
├── data/
│   ├── train.csv.csv
│   └── test.csv.csv
│
├── notebooks/
│   └── loan_prediction.py
│
├── screenshots/
│   ├── gender_vs_loan_status.png
│   ├── loan_approval_pie.png
│   ├── loan_amount_distribution.png
│   ├── income_vs_loan_amount.png
│   ├── correlation_heatmap.png
│   ├── credit_history_vs_loan.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── README.md
└── requirements.txt
Conclusion

This project demonstrates the complete Machine Learning workflow, including data cleaning, exploratory data analysis, feature preparation, model training, evaluation, and comparison. The results indicate that the Support Vector Machine (SVM) model provides the best performance for predicting loan approval status on this dataset.

Author

Vishalini O
B.Sc. Data Science
PSGR Krishnammal College for Women, Coimbatore
Aspiring Data Analyst | Excel | Power BI | Tableau | Python | SQL | Machine Learning 📊🚀
