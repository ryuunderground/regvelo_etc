"""Score the bisphenol panel against its held-out Tox21 outcomes.

These 11 compounds were excluded from the fine-tuning pairs from the start, so
unlike the valid split -- where the negatives are receptors of compounds the
model trained on -- nothing here was seen in any form. It is the cleanest test
the project has, and also the smallest: 39 usable pairs.

Inconclusive outcomes are dropped, not folded into either class.

Usage: python bisphenol_auroc.py ../bpa_panel/eval_v2/score_matrix.csv ...
"""
import csv
import pickle
import sys
from collections import defaultdict

import lmdb

from eval_auroc import auroc

RECEPTORS = ["ERalpha", "AR", "PPARgamma", "PR", "THRbeta"]


def load_matrix(path):
    """Panel LMDB keys are row indices, so map columns back via 'abbreviation'."""
    env = lmdb.open("../bpa_panel/mols_bpa_panel.lmdb", readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        abbr = {pickle.loads(v)["smi"]: pickle.loads(v)["abbreviation"] for _, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open(path)))
    names = [abbr.get(s) for s in rows[0][1:]]
    return {r[0]: {n: float(v) for n, v in zip(names, r[1:]) if n} for r in rows[1:]}


def main(paths):
    labels = [r for r in csv.DictReader(open("bisphenol_holdout.csv"))
              if r["tox21_outcome"] in ("Active", "Inactive")]

    print(f"{'matrix':<42}{'pooled':>9}" + "".join(f"{r:>12}" for r in RECEPTORS))
    for path in paths:
        m = load_matrix(path)
        pooled = ([], [])
        per = defaultdict(lambda: ([], []))
        for row in labels:
            rec, name = row["receptor"], row["abbreviation"]
            if rec not in m or name not in m[rec]:
                continue
            side = 0 if row["tox21_outcome"] == "Active" else 1
            pooled[side].append(m[rec][name])
            per[rec][side].append(m[rec][name])
        cells = "".join(f"{auroc(*per[r]):>12.3f}" for r in RECEPTORS)
        print(f"{path:<42}{auroc(*pooled):>9.3f}{cells}")

    counts = defaultdict(lambda: [0, 0])
    for row in labels:
        counts[row["receptor"]][0 if row["tox21_outcome"] == "Active" else 1] += 1
    n = "  ".join(f"{r} {counts[r][0]}+/{counts[r][1]}-" for r in RECEPTORS)
    print(f"{'  n:':<42}{len(labels):>9}  " + n)


if __name__ == "__main__":
    main(sys.argv[1:] or ["../bpa_panel/eval_base/score_matrix.csv",
                          "../bpa_panel/eval_v1/score_matrix.csv",
                          "../bpa_panel/eval_v2/score_matrix.csv"])
