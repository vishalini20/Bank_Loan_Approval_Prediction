import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df =pd.read_csv(r"C:\Users\Admin\Desktop\Bank_Loan_Approval_Prediction\data\train.csv.csv")

df.head()
print("Shape:", df.shape)

df.info()
df.describe()
df.isnull().sum()
missing = df.isnull().sum()

missing[missing > 0]


df.isnull().sum()



df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

df['Married'] = df['Married'].fillna(df['Married'].mode()[0])

df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])


print(df.isnull().sum())



sns.countplot(
    x='Gender',
    hue='Loan_Status',
    data=df,
    palette=['#1f77b4', '#ff7f0e']
)

plt.title('Gender vs Loan Approval Status')
plt.xlabel('Gender')
plt.ylabel('Number of Applicants')
plt.tight_layout()
plt.savefig("screenshots/gender_vs_loan_status.png")
plt.show()

loan_status = df['Loan_Status'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    loan_status,
    labels=loan_status.index,
    autopct='%1.1f%%',
    colors=['#1f77b4', '#ff7f0e']
)

plt.title('Loan Approval Percentage')
plt.tight_layout()
plt.savefig("screenshots/loan_approval_pie.png")
plt.show()



plt.figure(figsize=(8,5))

plt.hist(
    df['LoanAmount'],
    bins=20
)

plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.tight_layout()
plt.savefig("screenshots/loan_amount_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='ApplicantIncome',
    y='LoanAmount',
    hue='Loan_Status',
    data=df
)

plt.title('Income vs Loan Amount')
plt.tight_layout()
plt.savefig("screenshots/income_vs_loan_amount.png")
plt.show()

plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='Blues'
)

plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("screenshots/correlation_heatmap.png")
plt.show()


plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='Credit_History',
    hue='Loan_Status',
    palette=['#1f77b4', '#ff7f0e']
)

plt.title('Credit History vs Loan Approval')
plt.xlabel('Credit History')
plt.ylabel('Number of Applicants')
plt.tight_layout()
plt.savefig("screenshots/credit_history_vs_loan.png")
plt.show()




df.drop('Loan_ID', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])





X = df.drop('Loan_Status', axis=1)

y = df['Loan_Status']
print(X.columns.tolist())

print(X.isnull().sum())
print("Total Missing:", X.isnull().sum().sum())





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))




from sklearn.metrics import classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))




from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("KNN Accuracy:",
      accuracy_score(y_test, knn_pred))

from sklearn.svm import SVC

svm = SVC(kernel='linear')

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("SVM Accuracy:",
      accuracy_score(y_test, svm_pred))







results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'KNN',
        'SVM'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, svm_pred)
    ]
})

print(results.sort_values(
    by='Accuracy',
    ascending=False
))



best_model = results.loc[
    results['Accuracy'].idxmax()
]

print("Best Model:")
print(best_model)


plt.figure(figsize=(8,5))

sns.barplot(
    data=results,
    x='Model',
    y='Accuracy'
)

plt.title('Machine Learning Model Comparison')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("screenshots/model_comparison.png")
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Rejected','Approved'],
    yticklabels=['Rejected','Approved']
)

plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png")
plt.show()





