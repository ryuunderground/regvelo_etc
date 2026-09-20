"""Assemble DrugCLIP-format train/valid LMDBs from pairs.csv positive pairs.

Only positive (compound, receptor) pairs go in -- DrugCLIP's in_batch_softmax
loss treats every other item in a training batch as an automatic negative for
each positive pair, so explicit hard-negative labels aren't consumed by the
loss itself. The "hard negative" effect comes from batch composition: since
all 5 receptors here are a tight nuclear-receptor family, random batches
drawn from this positive-only pool naturally mix receptors that our
pairs.csv already confirmed have real differential specificity.

Split: by CID (not by pair) so the same compound never appears in both train
and valid, 90/10.
"""
import csv
import pickle
import random

import lmdb

random.seed(1)

POCKET_LMDB = "../bpa_panel/pocket_bpa_panel.lmdb"
MOL_LMDB = "mols_edc_panel.lmdb"


def load_pockets():
    env = lmdb.open(POCKET_LMDB, readonly=True, lock=False, subdir=False)
    pockets = {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            d = pickle.loads(v)
            pockets[d["pocket"]] = d
    env.close()
    return pockets


def load_mols():
    env = lmdb.open(MOL_LMDB, readonly=True, lock=False, subdir=False)
    mols = {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            mols[k.decode()] = pickle.loads(v)
    env.close()
    return mols


def main():
    pockets = load_pockets()
    mols = load_mols()
    print("pockets available:", list(pockets.keys()))

    positives = [
        row for row in csv.DictReader(open("pairs.csv"))
        if row["label"] == "positive" and row["cid"] in mols and row["receptor"] in pockets
    ]
    print(f"{len(positives)} usable positive pairs "
          f"(of {sum(1 for r in csv.DictReader(open('pairs.csv')) if r['label']=='positive')} total positive rows)")

    cids = sorted(set(r["cid"] for r in positives))
    random.shuffle(cids)
    n_valid = max(1, int(len(cids) * 0.1))
    valid_cids = set(cids[:n_valid])
    train_cids = set(cids[n_valid:])
    print(f"{len(train_cids)} train compounds, {len(valid_cids)} valid compounds")

    def write_split(name, split_cids):
        rows = [r for r in positives if r["cid"] in split_cids]
        env = lmdb.open(f"{name}.lmdb", subdir=False, map_size=200 * 1024 * 1024)
        with env.begin(write=True) as txn:
            for i, row in enumerate(rows):
                mol = mols[row["cid"]]
                pocket = pockets[row["receptor"]]
                entry = {
                    "atoms": mol["atoms"],
                    "coordinates": mol["coordinates"],
                    "smi": mol["smi"],
                    "pocket_atoms": pocket["pocket_atoms"],
                    "pocket_coordinates": pocket["pocket_coordinates"],
                    "pocket": pocket["pocket"],
                    "label": 1,
                }
                txn.put(str(i).encode(), pickle.dumps(entry))
        env.close()
        print(f"wrote {len(rows)} entries to {name}.lmdb")
        return len(rows)

    write_split("train", train_cids)
    write_split("valid", valid_cids)


if __name__ == "__main__":
    main()
