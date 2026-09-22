"""Training LMDBs from the ChEMBL dataset, split by scaffold.

Three things the Tox21 build got wrong and this one fixes.

Split by Murcko scaffold, not by CID. Splitting on compound id let 54% of the
ERalpha test compounds share a scaffold with a training active, with a maximum
Tanimoto of 1.00 -- the same molecule on both sides. Whole scaffold groups go to
one side or the other.

Hold out the bisphenol panel by name, not the whole chemotype. 8 of the 11 panel
compounds share BPA's Murcko scaffold, so a scaffold split would drag the entire
BPA chemotype out of training -- and the goal is a model that is good at exactly
that chemistry. The 11 named compounds stay out as the flagship test set; the
rest of the chemotype stays in and teaches it.

That makes the BPA evaluation an interpolation within a chemotype, not
generalisation to a new one, and it has to be reported that way. Inside a tight
chemotype logP alone reached 0.956 on the lab's Glide set, so the descriptor
baselines are not optional here.

  python build_chembl_train.py
"""
import csv
import glob
import json
import pickle
import random

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TRAIN_RECEPTORS = {"ERalpha", "ERbeta", "AR", "PR", "GR", "PPARgamma"}
MOLS = "mols_chembl_all.lmdb"
POCKETS = "../bpa_panel/pocket_nr16.lmdb"
OUT_DIR = "chembl_train_data"

# the flagship panel, held out entirely -- ChEMBL ids resolved by exact structure
PANEL = {
    "CHEMBL418971": "BPA", "CHEMBL384441": "BPS", "CHEMBL138061": "BPF",
    "CHEMBL1900054": "BPAF", "CHEMBL371077": "BPB", "CHEMBL2392656": "BPE",
    "CHEMBL1231453": "BPZ", "CHEMBL2392777": "BPC", "CHEMBL490942": "BPAP",
    "CHEMBL184450": "TBBPA", "CHEMBL1738928": "TCBPA",
}
VALID_FRACTION = 0.12


def scaffold_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:                                      # noqa: BLE001
        return None


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for f in glob.glob("chembl_*_dataset.csv"):
        for r in csv.DictReader(open(f)):
            if r["receptor"] in TRAIN_RECEPTORS and r["label"] in ("Active", "Inactive"):
                rows.append(r)
    print(f"{len(rows)}쌍 / 화합물 {len({r['chembl_id'] for r in rows})}개")

    env = lmdb.open(MOLS, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        mols = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
    env.close()
    penv = lmdb.open(POCKETS, readonly=True, lock=False, subdir=False)
    with penv.begin() as txn:
        pockets = {}
        for _, v in txn.cursor():
            d = pickle.loads(v)
            pockets[d["pocket"]] = d
    penv.close()
    print(f"컨포머 {len(mols)}개 / 포켓 {len(pockets)}개")

    usable = [r for r in rows if r["chembl_id"] in mols and r["receptor"] in pockets]
    held = [r for r in usable if r["chembl_id"] in PANEL]
    rest = [r for r in usable if r["chembl_id"] not in PANEL]
    print(f"사용 가능 {len(usable)}쌍 | 플래그십 홀드아웃 {len(held)}쌍 "
          f"({len({r['chembl_id'] for r in held})} 화합물)")

    # scaffold groups, assigned whole
    groups = {}
    for r in rest:
        groups.setdefault(r["chembl_id"], scaffold_of(r["smiles"]))
    by_scaffold = {}
    for cid, sc in groups.items():
        by_scaffold.setdefault(sc or f"__none_{cid}", []).append(cid)
    scaffolds = sorted(by_scaffold)
    random.seed(1)
    random.shuffle(scaffolds)

    n_valid_target = int(len(groups) * VALID_FRACTION)
    valid_cids, n = set(), 0
    for sc in scaffolds:
        if n >= n_valid_target:
            break
        valid_cids.update(by_scaffold[sc])
        n += len(by_scaffold[sc])
    print(f"scaffold {len(by_scaffold)}종 → valid {len(valid_cids)} 화합물 "
          f"({len(valid_cids)/len(groups):.0%})")

    # precomputed: compound -> receptors it is Active at. Doing this inside the
    # row loop would be 19k x 19k scans.
    actives_of = {}
    for r in usable:
        if r["label"] == "Active":
            actives_of.setdefault(r["chembl_id"], set()).add(r["receptor"])

    def write(name, subset):
        """Positives only -- in_batch_softmax takes the other rows as negatives."""
        pos = [r for r in subset if r["label"] == "Active"]
        env = lmdb.open(f"{OUT_DIR}/{name}.lmdb", subdir=False, map_size=8 * 1024 ** 3)
        with env.begin(write=True) as txn:
            for i, r in enumerate(pos):
                m, p = mols[r["chembl_id"]], pockets[r["receptor"]]
                txn.put(str(i).encode(), pickle.dumps({
                    "atoms": m["atoms"], "coordinates": m["coordinates"],
                    "smi": m["smi"],
                    "pocket_atoms": p["pocket_atoms"],
                    "pocket_coordinates": p["pocket_coordinates"],
                    "pocket": p["pocket"],
                    # every receptor this compound is Active at, for the
                    # false-negative mask added in step 9
                    "actives": "|".join(sorted(actives_of.get(r["chembl_id"], ()))),
                    "label": 1,
                }))
        env.close()
        print(f"  {name}.lmdb  {len(pos)} positive 쌍")
        return len(pos)

    train_rows = [r for r in rest if r["chembl_id"] not in valid_cids]
    valid_rows = [r for r in rest if r["chembl_id"] in valid_cids]
    write("train", train_rows)
    write("valid", valid_rows)

    json.dump(sorted(valid_cids), open(f"{OUT_DIR}/valid_ids.json", "w"))
    with open(f"{OUT_DIR}/holdout_panel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "name", "receptor", "label", "pchembl_median"])
        for r in held:
            w.writerow([r["chembl_id"], PANEL[r["chembl_id"]], r["receptor"],
                        r["label"], r["pchembl_median"]])

    for d in ("dict_mol.txt", "dict_pkt.txt"):
        if not os.path.exists(f"{OUT_DIR}/{d}"):
            os.link(f"train_data/{d}", f"{OUT_DIR}/{d}")
    print(f"\n{OUT_DIR}/ 준비 완료")


if __name__ == "__main__":
    main()
