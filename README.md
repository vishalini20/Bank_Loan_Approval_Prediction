# 🏦 BANK LOAN APPROVAL PREDICTION

## 📌 Project Overview

This project focuses on predicting whether a loan application will be approved or rejected using Machine Learning techniques.

The project involves data preprocessing, exploratory data analysis (EDA), model building, evaluation, and comparison of multiple classification algorithms to identify the best-performing model.

---

## 🎯 Objective

The objective of this project is to develop a predictive model that assists financial institutions in evaluating loan applications based on applicant information such as income, credit history, loan amount, and other relevant factors.

---

## 🛠️ Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- VS Code

---

## 📂 Dataset Features

The dataset contains the following applicant details:

- Gender
- Married Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (Target Variable)

---

## 🧹 Data Preprocessing

- Handled missing values using Mode and Median Imputation
- Removed unnecessary columns such as Loan_ID
- Encoded categorical variables using Label Encoding
- Performed train-test split for model training and evaluation
- Checked and treated missing values before model development

---

## 📊 Exploratory Data Analysis (EDA)

The following visualizations were created to gain insights from the data.

### 📌 Gender vs Loan Approval Status

- Analyzes loan approval distribution across genders
- Helps identify approval trends by gender

### 📌 Loan Approval Percentage

- Displays the percentage of approved and rejected loan applications
- Provides an overall view of approval rates

### 📌 Loan Amount Distribution

- Shows the distribution of loan amounts requested by applicants
- Helps identify common loan ranges

### 📌 Income vs Loan Amount

- Visualizes the relationship between applicant income and loan amount
- Useful for understanding applicant borrowing patterns

### 📌 Correlation Heatmap

- Displays relationships between numerical variables
- Helps identify important features affecting loan approval

### 📌 Credit History vs Loan Approval

- Examines the impact of credit history on loan approval
- Highlights one of the most influential factors in decision-making

---

## 🤖 Machine Learning Models Used

### 🔹 Logistic Regression

- Used as a baseline classification model
- Provides interpretable prediction results

### 🔹 Decision Tree Classifier

- Uses decision rules to classify loan applications
- Easy to visualize and interpret

### 🔹 Random Forest Classifier

- Ensemble model combining multiple decision trees
- Improves prediction accuracy and reduces overfitting

### 🔹 K-Nearest Neighbors (KNN)

- Classifies applicants based on similarity with neighboring data points
- Distance-based learning algorithm

### 🔹 Support Vector Machine (SVM)

- Powerful classification algorithm for binary prediction tasks
- Achieved the highest accuracy in this project

---

## 📈 Model Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Model Comparison Analysis

---

## 🏆 Model Performance Comparison

| Model | Accuracy |
|---------|---------|
| SVM | 79.67% |
| Logistic Regression | 78.86% |
| Random Forest | 75.61% |
| Decision Tree | 69.11% |
| KNN | 57.72% |

---

## 🥇 Best Performing Model

### Support Vector Machine (SVM)

- Achieved the highest accuracy of **79.67%**
- Delivered the best classification performance among all tested models
- Selected as the final model for loan approval prediction

---

## 📊 Visualizations Included

- Gender vs Loan Approval Status
- Loan Approval Percentage
- Loan Amount Distribution
- Income vs Loan Amount
- Correlation Heatmap
- Credit History vs Loan Approval
- Machine Learning Model Comparison
- Random Forest Confusion Matrix

---

## 💡 Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning Classification Algorithms
- Model Evaluation
- Feature Engineering
- Predictive Analytics
- Python Programming
- Problem Solving

---

## 📁 Project Structure

```text
Bank_Loan_Approval_Prediction/
│
├── data/
├── notebooks/
├── screenshots/
├── README.md
├── requirements.txt
└── loan_prediction.py
```

---

## ✅ Conclusion

This project demonstrates the complete Machine Learning workflow from data preprocessing and exploratory analysis to model training, evaluation, and comparison.

The results show that the Support Vector Machine (SVM) model achieved the highest accuracy and is the most effective model for predicting loan approval status in this dataset.

---

## 👩‍💻 Author

**Vishalini O**

**B.Sc. Data Science**  
PSGR Krishnammal College for Women, Coimbatore

**Aspiring Data Analyst | Excel | Power BI | Tableau | Python | SQL | Machine Learning | Data Visualization 📊🚀**
