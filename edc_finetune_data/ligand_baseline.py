"""Does DrugCLIP beat a trivial 2D-similarity baseline at all?

DrugCLIP reads the pocket structure. The cheapest thing that ignores structure
entirely is "this compound looks like a known binder": max Tanimoto to the
training actives of that receptor. If the pocket-based model cannot beat that,
the pocket is contributing nothing, whatever the pocket turns out to be.

Same leak-free split the 0.409 comes from, so the two numbers are comparable.
"""
import csv
import json
import pickle
from collections import defaultdict

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from eval_auroc import auroc, load_smiles_to_cid, load_matrix, RECEPTORS

RDLogger.DisableLog("rdApp.*")
_fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def main():
    smiles = {r["cid"]: r["smiles"] for r in csv.DictReader(open("library_smiles.csv"))}
    fps = {}
    for cid, smi in smiles.items():
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps[cid] = _fp(m)

    valid = set(json.load(open("valid_cids.json")))
    pairs = [(r["cid"], r["receptor"], r["label"]) for r in csv.DictReader(open("pairs.csv"))]
    M = load_matrix("eval_v2/score_matrix.csv", load_smiles_to_cid())

    # training actives per receptor -- the only thing the baseline may look at
    train_actives = defaultdict(list)
    for cid, rec, lab in pairs:
        if lab == "positive" and cid not in valid and cid in fps:
            train_actives[rec].append(fps[cid])

    print("leak-free split: DrugCLIP (pocket-based) vs max-Tanimoto to training actives")
    print(f"{'receptor':<11}{'n':>10}{'DrugCLIP v2':>13}{'2D kNN':>9}{'diff':>8}"
          f"{'train act':>11}")
    for rec in RECEPTORS:
        d_pos, d_neg, b_pos, b_neg = [], [], [], []
        refs = train_actives[rec]
        if not refs:
            continue
        for cid, r, lab in pairs:
            if r != rec or cid not in valid or cid not in fps or cid not in M.get(rec, {}):
                continue
            sim = max(DataStructs.BulkTanimotoSimilarity(fps[cid], refs))
            if lab == "positive":
                d_pos.append(M[rec][cid]); b_pos.append(sim)
            else:
                d_neg.append(M[rec][cid]); b_neg.append(sim)
        if not d_pos or not d_neg:
            continue
        a_dc, a_bl = auroc(d_pos, d_neg), auroc(b_pos, b_neg)
        print(f"{rec:<11}{f'{len(d_pos)}+/{len(d_neg)}-':>10}{a_dc:>13.3f}"
              f"{a_bl:>9.3f}{a_dc - a_bl:>+8.3f}{len(refs):>11}")

    print("\nThe baseline never sees a pocket. Where it wins, the structure side")
    print("of the model is not earning its place on that receptor.")


if __name__ == "__main__":
    main()
