"""Vina vs DrugCLIP on the same ERalpha compounds, against Tox21 labels.

The question this run exists to answer: is the binding signal present in the
3UU7 pocket DrugCLIP was handed? Vina is zero-shot and physics-based, so its
number is a property of the pocket and the labels, not of any training set.

  Vina near 0.5  -> the pocket or the labels are the problem
  Vina well above -> the signal is extractable and DrugCLIP is the weak link

Vina affinity is negative-is-better, so it is negated before ranking.
"""
import csv
import glob
import json
import os
import pickle
import random
import sys

sys.path.insert(0, "../edc_finetune_data")
import lmdb
from eval_auroc import auroc


def load_results():
    rows = []
    for f in glob.glob("results_*.csv"):
        for r in csv.DictReader(open(f)):
            if r["vina_affinity"]:
                rows.append((r["cid"], r["tox21_outcome"], float(r["vina_affinity"])))
    return rows


def drugclip_scores():
    """ERalpha column of the v2 matrix, keyed by CID."""
    env = lmdb.open("../edc_finetune_data/mols_edc_panel.lmdb",
                    readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        smi2cid = {pickle.loads(v)["smi"]: k.decode() for k, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open("../edc_finetune_data/eval_v2/score_matrix.csv")))
    cids = [smi2cid.get(s) for s in rows[0][1:]]
    for r in rows[1:]:
        if r[0] == "ERalpha":
            return {c: float(v) for c, v in zip(cids, r[1:]) if c}
    return {}


def boot(pos, neg, n=2000):
    random.seed(0)
    vals = sorted(auroc([random.choice(pos) for _ in pos],
                        [random.choice(neg) for _ in neg]) for _ in range(n))
    return vals[50], vals[1949]


def report(name, pos, neg):
    if not pos or not neg:
        print(f"{name:<34}{'--':>9}   {len(pos)}+/{len(neg)}- 부족")
        return
    a = auroc(pos, neg)
    lo, hi = boot(pos, neg)
    print(f"{name:<34}{a:>9.3f}   [{lo:.3f}, {hi:.3f}]   {len(pos)}+/{len(neg)}-")


def main():
    rows = load_results()
    dc = drugclip_scores()
    clean = set(json.load(open("clean_actives.json")))
    trained = set()
    valid = set(json.load(open("../edc_finetune_data/valid_cids.json")))
    for r in csv.DictReader(open("../edc_finetune_data/pairs.csv")):
        if r["label"] == "positive" and r["cid"] not in valid:
            trained.add(r["cid"])

    print(f"docked {len(rows)} compounds\n")
    print(f"{'':<34}{'AUROC':>9}   {'95% CI':<16} n")

    # Vina on everything it docked -- its own number, no contamination concept
    pos = [-a for c, o, a in rows if o == "Active"]
    neg = [-a for c, o, a in rows if o == "Inactive"]
    report("Vina  (전체)", pos, neg)

    # the subset where DrugCLIP is also clean, so the two are comparable
    sub = [(c, o, a) for c, o, a in rows
           if c in dc and (o == "Inactive" or c in clean) and c not in trained]
    vp = [-a for c, o, a in sub if o == "Active"]
    vn = [-a for c, o, a in sub if o == "Inactive"]
    dp = [dc[c] for c, o, a in sub if o == "Active"]
    dn = [dc[c] for c, o, a in sub if o == "Inactive"]
    print()
    report("Vina      (DrugCLIP-clean 부분집합)", vp, vn)
    report("DrugCLIP  (같은 화합물)", dp, dn)


if __name__ == "__main__":
    main()
