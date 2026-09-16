"""Extract DrugCLIP-format pockets from real PDB structures.

Pocket definition mirrors DrugCLIP/py_scripts/write_dude_multi.py: any residue
with at least one atom within `radius` Angstrom of any reference-ligand atom
is included, in full (all its atoms, not just the close ones).
"""
import pickle
import sys

import lmdb
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

RADIUS = 6.0

TARGETS = [
    # receptor, pdb file, reference ligand 3-letter code
    ("ERalpha", "3UU7", "2OH"),
    ("ERbeta", "5TOA", "EST"),
    ("ERRgamma", "2E2R", "2OH"),
    ("PPARgamma", "9F7W", "2OH"),
    ("AR", "2AMA", "DHT"),
    ("THRbeta", "3GWS", "T3"),
    ("PR", "1A28", "STR"),
]


def extract_pocket(pdb_path, ligand_code, radius=RADIUS):
    ppdb = PandasPdb().read_pdb(pdb_path)

    protein = ppdb.df["ATOM"]
    hetatm = ppdb.df["HETATM"]
    ligand = hetatm[hetatm["residue_name"] == ligand_code]
    if len(ligand) == 0:
        raise ValueError(f"ligand {ligand_code} not found in {pdb_path}")

    protein_coord = protein[["x_coord", "y_coord", "z_coord"]].to_numpy()
    ligand_coord = ligand[["x_coord", "y_coord", "z_coord"]].to_numpy()

    # residue key = chain + residue number (matches write_dude_multi.py)
    residue_key = protein["chain_id"].astype(str) + protein["residue_number"].astype(str)

    # any protein atom within radius of any ligand atom -> whole residue included
    dists = np.linalg.norm(
        protein_coord[:, None, :] - ligand_coord[None, :, :], axis=-1
    )
    close_atom_mask = (dists < radius).any(axis=1)
    close_residues = set(residue_key[close_atom_mask])

    pocket_mask = residue_key.isin(close_residues)
    pocket_atoms = protein.loc[pocket_mask, "atom_name"].tolist()
    pocket_coords = [
        row for row in protein.loc[pocket_mask, ["x_coord", "y_coord", "z_coord"]].to_numpy()
    ]
    return pocket_atoms, pocket_coords


def main():
    env = lmdb.open(
        "pocket_bpa_panel.lmdb", subdir=False, map_size=50 * 1024 * 1024
    )
    with env.begin(write=True) as txn:
        for i, (receptor, pdb_id, lig) in enumerate(TARGETS):
            atoms, coords = extract_pocket(f"pdb/{pdb_id}.pdb", lig)
            print(f"{receptor} ({pdb_id}, ref ligand {lig}): {len(atoms)} pocket atoms")
            entry = {
                "pocket": receptor,
                "pocket_atoms": atoms,
                "pocket_coordinates": coords,
            }
            txn.put(str(i).encode(), pickle.dumps(entry))
    env.close()
    print("wrote", len(TARGETS), "pockets to pocket_bpa_panel.lmdb")


if __name__ == "__main__":
    main()
