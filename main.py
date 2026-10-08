import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load features
df = pd.read_csv("results/features.csv")


# Same balanced dataset as Quantum SVM
positive = df[df["label"] == 1].sample(n=400, random_state=42)
negative = df[df["label"] == 0].sample(n=400, random_state=42)

data = pd.concat([positive, negative]).sample(
    frac=1,
    random_state=42
)


# Features and labels
X = data[["aa_code", "num_atoms"]]
y = data["label"]
groups = data["protein_id"]


# Split by protein
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# Classical SVM
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    ))
])


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n===== FAIR CLASSICAL SVM RESULTS =====")
print("Samples   :", len(data))
print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-Score  :", round(f1, 4))
print("======================================")