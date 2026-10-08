import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# Load dataset
df = pd.read_csv("results/features.csv")

# Balanced sample
positive = df[df["label"] == 1].sample(n=400, random_state=42)
negative = df[df["label"] == 0].sample(n=400, random_state=42)

data = pd.concat([positive, negative]).sample(frac=1, random_state=42)

X = data[["aa_code", "num_atoms"]]
y = data["label"]
groups = data["protein_id"]


# Train/test split by protein
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


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Quantum feature map
feature_map = ZZFeatureMap(
    feature_dimension=2,
    reps=2
)


# Quantum kernel
quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)


print("\nCalculating quantum kernel...")

K_train = quantum_kernel.evaluate(
    x_vec=X_train
)

K_test = quantum_kernel.evaluate(
    x_vec=X_test,
    y_vec=X_train
)


# Quantum-kernel SVM
model = SVC(
    kernel="precomputed",
    class_weight="balanced"
)

model.fit(K_train, y_train)

y_pred = model.predict(K_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n===== QUANTUM SVM RESULTS =====")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("===============================")