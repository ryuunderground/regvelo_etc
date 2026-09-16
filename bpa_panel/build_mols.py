"""Build a DrugCLIP-format mols.lmdb from bpa_panel/compounds.csv."""
import csv
import pickle

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

NUM_CONF = 10


def gen_conformers(mol, num_conf=NUM_CONF):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(
        mol, numConfs=num_conf, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=True
    )
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return mol


def main():
    rows = list(csv.DictReader(open("compounds.csv")))

    env = lmdb.open("mols_bpa_panel.lmdb", subdir=False, map_size=50 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, row in enumerate(rows):
            smi = row["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print("FAILED to parse:", row["abbreviation"], smi)
                continue
            mol = gen_conformers(mol)
            if mol.GetNumConformers() == 0:
                print("FAILED to embed conformers:", row["abbreviation"])
                continue

            coords = [c.GetPositions() for c in mol.GetConformers()]
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]

            entry = {
                "atoms": atoms,
                "coordinates": coords,
                "smi": Chem.MolToSmiles(mol),
                "abbreviation": row["abbreviation"],
                "name": row["name"],
            }
            txn.put(str(i).encode(), pickle.dumps(entry))
            print(row["abbreviation"], "->", len(atoms), "atoms,", len(coords), "conformers")
    env.close()
    print("wrote", len(rows), "compounds to mols_bpa_panel.lmdb")


if __name__ == "__main__":
    main()
