"""Prepare inputs for Uni-Mol Docking V2: protein-only PDBs, docking-grid
JSONs (from the real reference-ligand position), and ligand SDFs, for a
small 6-receptor x 6-ligand test panel."""
import json
import os
import pickle

import lmdb
import numpy as np
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# 3 receptors BPA binds well (highest raw score) + 3 it doesn't (lowest), per score_matrix.csv
RECEPTORS = {
    "ERRgamma": ("2E2R", "2OH"),   # BPA score 0.8267 (best)
    "AR": ("2AMA", "DHT"),          # BPA score 0.5887
    "ERalpha": ("3UU7", "2OH"),     # BPA score 0.5711
    "PPARgamma": ("9F7W", "2OH"),   # BPA score 0.0148 (worst)
    "PR": ("1A28", "STR"),          # BPA score 0.3497
    "THRbeta": ("3GWS", "T3"),      # BPA score 0.3707
}

PDB_DIR = "../pdb"
OUT_DIR = "inputs"
PADDING = 8.0  # Angstrom around the reference ligand's bounding box


def strip_and_grid(receptor, pdb_id, lig_code):
    ppdb = PandasPdb().read_pdb(f"{PDB_DIR}/{pdb_id}.pdb")

    protein_only = PandasPdb()
    protein_only.df["ATOM"] = ppdb.df["ATOM"]
    protein_only.df["HETATM"] = ppdb.df["HETATM"].iloc[0:0]  # empty
    out_pdb = f"{OUT_DIR}/{receptor}.pdb"
    protein_only.to_pdb(out_pdb, records=["ATOM"])

    lig = ppdb.df["HETATM"]
    lig = lig[lig["residue_name"] == lig_code]
    # some entries have multiple copies of the reference ligand (different
    # chains/asymmetric-unit copies) -- pooling them would average across
    # distinct binding sites into a nonsensical oversized box. Keep just the
    # first (chain_id, residue_number) instance.
    first_instance = lig[["chain_id", "residue_number"]].iloc[0]
    lig = lig[
        (lig["chain_id"] == first_instance["chain_id"])
        & (lig["residue_number"] == first_instance["residue_number"])
    ]
    coords = lig[["x_coord", "y_coord", "z_coord"]].to_numpy()
    center = coords.mean(axis=0)
    size = (coords.max(axis=0) - coords.min(axis=0)) + 2 * PADDING

    grid = {
        "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
        "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2]),
    }
    grid_path = f"{OUT_DIR}/{receptor}_grid.json"
    with open(grid_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(receptor, "->", out_pdb, grid_path, "center=", center.round(1), "size=", size.round(1))
    return out_pdb, grid_path


# 3 bisphenol-family compounds (diverse per prior findings) + 3 unrelated random compounds
BISPHENOL_LIGANDS = {
    "BPS": "O=S(=O)(c1ccc(O)cc1)c1ccc(O)cc1",
    "BPAF": "Oc1ccc(C(c2ccc(O)cc2)(C(F)(F)F)C(F)(F)F)cc1",
    "BPZ": "Oc1ccc(C2(c3ccc(O)cc3)CCCCC2)cc1",
}


def build_ligand_sdfs():
    os.makedirs(OUT_DIR, exist_ok=True)
    ligand_paths = {}

    for name, smi in BISPHENOL_LIGANDS.items():
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        path = f"{OUT_DIR}/{name}.sdf"
        Chem.MolToMolFile(mol, path)
        ligand_paths[name] = path

    # 3 random, non-bisphenol compounds from the earlier 20k library sample
    env = lmdb.open("../mols_library_sample.lmdb", readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        for i, idx in enumerate([0, 1, 2]):
            v = txn.get(str(idx).encode())
            d = pickle.loads(v)
            mol = Chem.MolFromSmiles(d["smi"])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
            name = f"RANDOM{i+1}"
            path = f"{OUT_DIR}/{name}.sdf"
            Chem.MolToMolFile(mol, path)
            ligand_paths[name] = path
            print(name, "->", d["smi"])
    env.close()
    return ligand_paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    receptor_files = {r: strip_and_grid(r, pdb_id, lig) for r, (pdb_id, lig) in RECEPTORS.items()}
    ligand_files = build_ligand_sdfs()

    rows = []
    for receptor, (protein_pdb, grid_json) in receptor_files.items():
        for ligand_name, ligand_sdf in ligand_files.items():
            rows.append({
                # absolute paths: demo.py's cwd is unimol_docking_v2/interface, not here
                "input_protein": os.path.abspath(protein_pdb),
                "input_ligand": os.path.abspath(ligand_sdf),
                "input_docking_grid": os.path.abspath(grid_json),
                "output_ligand_name": f"{receptor}__{ligand_name}",
            })

    import csv
    with open(f"{OUT_DIR}/batch_one2one.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input_protein", "input_ligand", "input_docking_grid", "output_ligand_name"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} pairs to {OUT_DIR}/batch_one2one.csv")


if __name__ == "__main__":
    main()
