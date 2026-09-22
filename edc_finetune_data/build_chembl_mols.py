"""3D conformers for every compound in the ChEMBL training set.

This is the only real work standing between the step-14 dataset and a retrain --
13,661 compounds across the six trainable receptors. Same generation settings as
build_mols.py so the two panels stay comparable, plus the salt stripping the
Vina runs needed: ChEMBL ships hydrochlorides and hydrates, and a multi-fragment
molecule embeds badly.

Resumable -- anything already in the output LMDB is skipped.
"""
import csv
import glob
import pickle
import sys

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

TRAIN_RECEPTORS = {"ERalpha", "ERbeta", "AR", "PR", "GR", "PPARgamma"}
OUT = "mols_chembl_all.lmdb"
NUM_CONF = 5


def compounds():
    out = {}
    for f in glob.glob("chembl_*_dataset.csv"):
        for r in csv.DictReader(open(f)):
            if r["receptor"] in TRAIN_RECEPTORS and r["label"] in ("Active", "Inactive"):
                out[r["chembl_id"]] = r["smiles"]
    return out


def main():
    comp = compounds()
    print(f"{len(comp)}개 화합물", flush=True)

    env = lmdb.open(OUT, subdir=False, map_size=8 * 1024 ** 3)
    with env.begin() as txn:
        done = {k.decode() for k, _ in txn.cursor()}
    todo = [(c, s) for c, s in comp.items() if c not in done]
    print(f"{len(done)}개 완료, {len(todo)}개 남음", flush=True)

    ok, fail = 0, 0
    with env.begin(write=True) as txn:
        for i, (cid, smi) in enumerate(todo, 1):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
            if mol is None or mol.GetNumAtoms() < 4:
                fail += 1
                continue
            m = Chem.AddHs(mol)
            if len(AllChem.EmbedMultipleConfs(
                    m, numConfs=NUM_CONF, pruneRmsThresh=1,
                    maxAttempts=50, useRandomCoords=True)) == 0:
                fail += 1
                continue
            try:
                AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
            except Exception:                              # noqa: BLE001
                pass
            m = Chem.RemoveHs(m)
            txn.put(cid.encode(), pickle.dumps({
                "atoms": [a.GetSymbol() for a in m.GetAtoms()],
                "coordinates": [c.GetPositions() for c in m.GetConformers()],
                "smi": Chem.MolToSmiles(m),
                "cid": cid,
            }))
            ok += 1
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}  성공 {ok} 실패 {fail}", flush=True)
    env.close()
    print(f"\n{ok}개 생성 / {fail}개 실패 -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
