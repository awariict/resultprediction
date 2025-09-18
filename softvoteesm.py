import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. Load Dataset
# =======================
df = pd.read_csv("student_academic_performance_dataset.csv")

# =======================
# 2. Encode Target
# =======================
le = LabelEncoder()
df["Grade_encoded"] = le.fit_transform(df["Grade"].astype(str))  # A–F → 0–5
y = df["Grade_encoded"]

# =======================
# 3. Prepare Features
# =======================
# Drop non-feature columns (IDs, Name)
non_features = ["Grade", "RegistrationNumber", "Name"]
X = df.drop(columns=[c for c in non_features if c in df.columns])

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# =======================
# 4. Define Models
# =======================
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
voting_clf = VotingClassifier(
    estimators=[("knn", knn), ("nb", nb)],
    voting="soft"
)

# =======================
# 5. Train Models
# =======================
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
voting_clf.fit(X_train, y_train)

# =======================
# 6. Predictions
# =======================
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)
y_pred_vote = voting_clf.predict(X_test)

# =======================
# 7. Evaluation
# =======================
print("=== Train/Test Split Accuracies ===")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Voting Ensemble Accuracy:", accuracy_score(y_test, y_pred_vote))

print("\nClassification Report (Voting Ensemble):\n")
print(classification_report(y_test, y_pred_vote, target_names=le.classes_))

# =======================
# 8. Cross-Validation (5-Fold)
# =======================
print("\n=== Cross-Validation Accuracies (5-Fold) ===")
for model_name, model in [("KNN", knn), ("Naive Bayes", nb), ("Voting Ensemble", voting_clf)]:
    scores = cross_val_score(model, X_imputed, y, cv=5)
    print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# =======================
# 9. Confusion Matrix
# =======================
cm_vote = confusion_matrix(y_test, y_pred_vote)
plt.figure(figsize=(6,5))
sns.heatmap(cm_vote, annot=True, fmt="d", cmap="Purples",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Voting Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =======================
# 10. Accuracy Comparison Bar Chart
# =======================
acc_df = pd.DataFrame({
    "Model": ["KNN", "Naive Bayes", "Voting Ensemble"],
    "Accuracy": [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_vote)]
})
plt.figure(figsize=(6,4))
sns.barplot(x="Model", y="Accuracy", data=acc_df, palette="viridis")
plt.title("Model Accuracy Comparison (Train/Test Split)")
plt.show()
