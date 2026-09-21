"""Build a DrugCLIP mols LMDB for the compounds the lab already docked in Glide.

scores/<RECEPTOR>/flexible_sp.csv is the 9/13 flexible_five_v1 Glide run: 30
compounds across ESR1/ESR2/ESRRG/AR/PXR/VDR. Those Glide scores are an
independent ruler -- a different method on a differently prepared receptor --
so scoring the same compounds through DrugCLIP lets us ask which one tracks
the measured Tox21 labels. That is what separates "our ERalpha pocket is
over-fit to BPA" from "DrugCLIP cannot use this pocket".

SMILES come from PubChem, not from the Glide CSV: those rows carry Epik-
prepared protonation states, and the rest of this project feeds DrugCLIP
neutral PubChem structures.
"""
import csv
import json
import os
import pickle
import re
import sys
import time
import urllib.request

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

SCORES_DIR = "../scores"
RECEPTORS = ["ESR1", "ESR2", "ESRRG", "AR", "PXR", "VDR"]
OUT_LMDB = "mols_glide_panel.lmdb"
NUM_CONF = 5


def glide_cids():
    cids = set()
    for rec in RECEPTORS:
        path = os.path.join(SCORES_DIR, rec, "flexible_sp.csv")
        for row in csv.DictReader(open(path)):
            m = re.search(r"CID(\d+)", row["title"])
            if m:
                cids.add(m.group(1))
    return sorted(cids, key=int)


def fetch_smiles(cids):
    out = {}
    for i in range(0, len(cids), 100):
        batch = cids[i:i + 100]
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               + ",".join(batch) + "/property/IsomericSMILES/JSON")
        with urllib.request.urlopen(url, timeout=40) as r:
            for p in json.load(r)["PropertyTable"]["Properties"]:
                smi = p.get("IsomericSMILES") or p.get("SMILES") or p.get("ConnectivitySMILES")
                if smi:
                    out[str(p["CID"])] = smi
        time.sleep(0.3)
    return out


def gen_conformers(mol, num_conf=NUM_CONF):
    """Same settings as build_mols.py, so the two panels stay comparable."""
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
    return Chem.RemoveHs(mol)


def main():
    cids = glide_cids()
    print(f"{len(cids)} compounds in the Glide run")
    smiles = fetch_smiles(cids)
    print(f"{len(smiles)}/{len(cids)} SMILES from PubChem")

    env = lmdb.open(OUT_LMDB, subdir=False, map_size=200 * 1024 * 1024)
    ok, failed = 0, []
    with env.begin(write=True) as txn:
        for cid in cids:
            smi = smiles.get(cid)
            mol = Chem.MolFromSmiles(smi) if smi else None
            mol = gen_conformers(mol) if mol else None
            if mol is None or mol.GetNumConformers() == 0:
                failed.append(cid)
                continue
            txn.put(cid.encode(), pickle.dumps({
                "atoms": [a.GetSymbol() for a in mol.GetAtoms()],
                "coordinates": [c.GetPositions() for c in mol.GetConformers()],
                "smi": Chem.MolToSmiles(mol),
                "cid": cid,
            }))
            ok += 1
    env.close()
    print(f"wrote {ok} compounds to {OUT_LMDB}" + (f" ({len(failed)} failed: {failed})" if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
