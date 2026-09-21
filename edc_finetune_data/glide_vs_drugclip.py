"""Head-to-head: Glide gscore vs DrugCLIP cosine, same pairs, same labels.

Glide is an independent ruler -- different method, differently prepared
receptor -- so where the two disagree against measured Tox21 outcomes tells us
which side the ERalpha failure sits on:

  both low   -> the labels or the pocket are the problem, not DrugCLIP
  Glide fine -> DrugCLIP cannot use a pocket that does carry the signal

Glide gscore is negative-is-better, so it is negated before ranking.
"""
import csv
import json
import os
import pickle
import re
from collections import defaultdict

import lmdb

from eval_auroc import auroc

SCORES_DIR = "../scores"
# Glide target name -> the pocket name used everywhere else in this project
GLIDE_TO_POCKET = {"ESR1": "ERalpha", "ESR2": "ERbeta", "ESRRG": "ERRgamma", "AR": "AR"}
ASSAYS = {"ERalpha": ["743078"], "AR": ["743053", "743063"],
          "PPARgamma": ["743191"], "PR": ["1347031"], "THRbeta": ["743066"]}
RANK = {"Active": 2, "Inactive": 1, "Inconclusive": 0}


def best_glide():
    """Most negative gscore per (target, CID) -- Glide emits many poses per row."""
    out = defaultdict(dict)
    for target in GLIDE_TO_POCKET:
        path = os.path.join(SCORES_DIR, target, "flexible_sp.csv")
        for row in csv.DictReader(open(path)):
            m = re.search(r"CID(\d+)", row["title"])
            if not m or row["docking_status"] != "Done":
                continue
            try:
                g = float(row["r_i_glide_gscore"])
            except ValueError:
                continue
            cid = m.group(1)
            if cid not in out[target] or g < out[target][cid]:
                out[target][cid] = g
    return out


def tox21():
    out = defaultdict(dict)
    for receptor, aids in ASSAYS.items():
        for aid in aids:
            for row in json.load(open(f"raw/aid_{aid}.json"))["Table"]["Row"]:
                cid, o = row["Cell"][2], row["Cell"][3]
                if cid and (cid not in out[receptor] or RANK[o] > RANK[out[receptor][cid]]):
                    out[receptor][cid] = o
    return out


def drugclip(path):
    env = lmdb.open("mols_glide_panel.lmdb", readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        smi2cid = {pickle.loads(v)["smi"]: k.decode() for k, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open(path)))
    cids = [smi2cid.get(s) for s in rows[0][1:]]
    return {r[0]: {c: float(v) for c, v in zip(cids, r[1:]) if c} for r in rows[1:]}


def main():
    G, T = best_glide(), tox21()
    dc = {"v2": drugclip("glide_v2/score_matrix.csv"),
          "pretrained": drugclip("glide_base/score_matrix.csv")}

    print("Same compounds, same Tox21 labels, two rulers.  AUROC, higher is better.\n")
    print(f"{'receptor':<11}{'n':>10}{'Glide':>9}{'DrugCLIP v2':>14}{'DrugCLIP pre':>14}")
    for target, pocket in GLIDE_TO_POCKET.items():
        if pocket not in ASSAYS:
            print(f"{pocket:<11}{'--':>10}   no Tox21 assay pulled for this receptor")
            continue
        pos_g, neg_g, pos_2, neg_2, pos_b, neg_b = [], [], [], [], [], []
        for cid, g in G[target].items():
            outcome = T[pocket].get(cid)
            if outcome not in ("Active", "Inactive"):
                continue
            if cid not in dc["v2"].get(pocket, {}):
                continue
            active = outcome == "Active"
            (pos_g if active else neg_g).append(-g)          # negate: lower gscore binds better
            (pos_2 if active else neg_2).append(dc["v2"][pocket][cid])
            (pos_b if active else neg_b).append(dc["pretrained"][pocket][cid])
        n = f"{len(pos_g)}+/{len(neg_g)}-"
        if not pos_g or not neg_g:
            print(f"{pocket:<11}{n:>10}   not enough of both classes")
            continue
        print(f"{pocket:<11}{n:>10}{auroc(pos_g, neg_g):>9.3f}"
              f"{auroc(pos_2, neg_2):>14.3f}{auroc(pos_b, neg_b):>14.3f}")

    print("\nGlide targets with no Tox21 assay on our side:",
          ", ".join(f"{t}({p})" for t, p in GLIDE_TO_POCKET.items() if p not in ASSAYS)
          or "none")


if __name__ == "__main__":
    main()
