"""Build a DrugCLIP-format mols.lmdb from compounds_smiles.csv (1,555 EDC
panel compounds from Tox21, BPA/bisphenols already excluded upstream)."""
import csv
import pickle

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

NUM_CONF = 5


def gen_conformers(mol, num_conf=NUM_CONF):
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_conf, pruneRmsThresh=1, maxAttempts=50, useRandomCoords=True
    )
    if len(ids) == 0:
        return None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return mol


def main():
    rows = list(csv.DictReader(open("compounds_smiles.csv")))
    print(f"{len(rows)} compounds to process")

    env = lmdb.open("mols_edc_panel.lmdb", subdir=False, map_size=500 * 1024 * 1024)
    n_ok, n_fail = 0, 0
    fail_cids = []
    with env.begin(write=True) as txn:
        for i, row in enumerate(rows):
            cid, smi = row["cid"], row["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_fail += 1
                fail_cids.append(cid)
                continue
            mol = gen_conformers(mol)
            if mol is None or mol.GetNumConformers() == 0:
                n_fail += 1
                fail_cids.append(cid)
                continue

            coords = [c.GetPositions() for c in mol.GetConformers()]
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]

            entry = {
                "atoms": atoms,
                "coordinates": coords,
                "smi": Chem.MolToSmiles(mol),
                "cid": cid,
            }
            txn.put(cid.encode(), pickle.dumps(entry))
            n_ok += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)} processed ({n_ok} ok, {n_fail} failed)")
    env.close()

    print(f"\nwrote {n_ok} compounds to mols_edc_panel.lmdb ({n_fail} failed)")
    if fail_cids:
        with open("failed_cids.txt", "w") as f:
            f.write("\n".join(fail_cids))
        print(f"failed CIDs written to failed_cids.txt")


if __name__ == "__main__":
    main()
