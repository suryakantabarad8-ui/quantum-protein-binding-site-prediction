import pandas as pd
import os

INPUT = "results/dataset.csv"
OUTPUT = "results/features.csv"

df = pd.read_csv(INPUT)

# Number of atoms/features based on amino-acid type
atom_count = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8,
    "CYS": 6, "GLN": 9, "GLU": 9, "GLY": 4,
    "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9,
    "MET": 8, "PHE": 11, "PRO": 7, "SER": 6,
    "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7
}

# Convert amino-acid type to numerical value
amino_acids = sorted(atom_count.keys())
aa_map = {aa: i for i, aa in enumerate(amino_acids)}

df["aa_code"] = df["residue"].map(aa_map)
df["num_atoms"] = df["residue"].map(atom_count)

# Keep only non-leaky ML features
features = df[
    ["protein_id", "residue", "aa_code", "num_atoms", "label"]
]

os.makedirs("results", exist_ok=True)
features.to_csv(OUTPUT, index=False)

print("Features created successfully!")
print("Rows:", len(features))
print("Columns:", features.columns.tolist())
print("Saved:", OUTPUT)