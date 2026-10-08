import os
import glob
import numpy as np
import pandas as pd

PDB_DIR = "data/coach420"
OUTPUT_FILE = "results/dataset.csv"

rows = []

pdb_files = glob.glob(os.path.join(PDB_DIR, "*.pdb"))

print("PDB files found:", len(pdb_files))

for pdb_file in pdb_files:
    protein_id = os.path.basename(pdb_file).replace(".pdb", "")

    protein_atoms = []
    ligand_atoms = []

    with open(pdb_file, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain = line[21].strip()
                    residue_id = line[22:26].strip()

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    protein_atoms.append(
                        (residue_name, chain, residue_id,
                         atom_name, x, y, z)
                    )
                except:
                    pass

            elif line.startswith("HETATM"):
                try:
                    residue_name = line[17:20].strip()

                    # Ignore water
                    if residue_name in ["HOH", "WAT"]:
                        continue

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    ligand_atoms.append((x, y, z))
                except:
                    pass

    if not protein_atoms or not ligand_atoms:
        continue

    # Group protein atoms by residue
    residues = {}

    for res_name, chain, res_id, atom_name, x, y, z in protein_atoms:
        key = (chain, res_id, res_name)

        if key not in residues:
            residues[key] = []

        residues[key].append((x, y, z))

    for (chain, res_id, res_name), atoms in residues.items():

        # Minimum atom-to-ligand distance
        min_distance = min(
            np.linalg.norm(
                np.array(atom) - np.array(ligand)
            )
            for atom in atoms
            for ligand in ligand_atoms
        )

        label = 1 if min_distance <= 8.0 else 0

        rows.append({
            "protein_id": protein_id,
            "chain": chain,
            "residue_id": res_id,
            "residue": res_name,
            "min_distance": min_distance,
            "label": label
        })

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset created successfully!")
print("Rows:", len(df))
print("Binding residues:", (df["label"] == 1).sum())
print("Non-binding residues:", (df["label"] == 0).sum())
print("Saved at:", OUTPUT_FILE)