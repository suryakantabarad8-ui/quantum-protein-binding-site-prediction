import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# FAIR RESULTS
results = pd.DataFrame({
    "Model": ["Classical SVM", "Quantum SVM"],
    "Accuracy": [0.5824, 0.5353],
    "Precision": [0.6000, 0.5156],
    "Recall": [0.3704, 0.4074],
    "F1-Score": [0.4580, 0.4552]
})

# Save result table
results.to_csv("results/results.csv", index=False)

print("\n===== FINAL FAIR COMPARISON =====")
print(results.to_string(index=False))
print("\nSaved: results/results.csv")


# Accuracy graph
plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Classical SVM vs Quantum SVM - Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=300)
plt.close()


# F1 graph
plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["F1-Score"])
plt.ylabel("F1-Score")
plt.title("Classical SVM vs Quantum SVM - F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/f1_comparison.png", dpi=300)
plt.close()

print("Saved: results/accuracy_comparison.png")
print("Saved: results/f1_comparison.png")