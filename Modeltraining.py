import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_academic_performance_dataset.csv")

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col].astype(str))

# Features (X) and Target (y) - predicting Grade
X = df.drop(["Grade", "GPA"], axis=1)
y = df["Grade"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- KNN Model -----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ----------------- Naive Bayes Model -----------------
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# ----------------- Evaluation -----------------
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# ----------------- Confusion Matrices -----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Naive Bayes Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.show()

# ----------------- Accuracy Comparison -----------------
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_nb = accuracy_score(y_test, y_pred_nb)

plt.bar(["KNN", "Naive Bayes"], [acc_knn, acc_nb], color=["blue", "green"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
