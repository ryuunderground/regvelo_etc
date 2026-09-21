"""Extract the Tox21 outcomes for the 11-compound bisphenol panel.

build_pairs.py drops these CIDs from the fine-tuning pairs on purpose, but the
raw assay pulls still carry their measured outcomes -- so they are a genuinely
held-out validation set across every receptor we pulled, and the one the whole
BPA case study is about. Writes bisphenol_holdout.csv.
"""
import csv
import json
import os

RAW_DIR = "raw"
ASSAYS = {  # same mapping as build_pairs.py
    "ERalpha": ["743078"],
    "AR": ["743053", "743063"],
    "PPARgamma": ["743191"],
    "PR": ["1347031"],
    "THRbeta": ["743066"],
}
# bpa_panel/compounds.csv, CID -> abbreviation
PANEL = {
    "6623": "BPA", "6626": "BPS", "12111": "BPF", "73864": "BPAF",
    "66166": "BPB", "608116": "BPE", "232446": "BPZ", "6620": "BPC",
    "623849": "BPAP", "6618": "TBBPA", "6619": "TCBPA",
}
RANK = {"Active": 2, "Inactive": 1, "Inconclusive": 0}


def outcomes_for(aid):
    path = os.path.join(RAW_DIR, f"aid_{aid}.json")
    found = {}
    for row in json.load(open(path))["Table"]["Row"]:
        cid, outcome = row["Cell"][2], row["Cell"][3]
        if cid in PANEL and (cid not in found or RANK[outcome] > RANK[found[cid]]):
            found[cid] = outcome
    return found


def main():
    rows = []
    for receptor, aids in ASSAYS.items():
        combined = {}
        for aid in aids:
            for cid, outcome in outcomes_for(aid).items():
                if cid not in combined or RANK[outcome] > RANK[combined[cid]]:
                    combined[cid] = outcome
        for cid, outcome in sorted(combined.items(), key=lambda kv: PANEL[kv[1 - 1]]):
            rows.append((PANEL[cid], cid, receptor, outcome))

    with open("bisphenol_holdout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["abbreviation", "cid", "receptor", "tox21_outcome"])
        w.writerows(sorted(rows))

    print(f"wrote {len(rows)} measured (compound, receptor) pairs\n")
    by_rec = {}
    for _, _, receptor, outcome in rows:
        by_rec.setdefault(receptor, []).append(outcome)
    for receptor, outs in sorted(by_rec.items()):
        n = {k: outs.count(k) for k in RANK}
        print(f"  {receptor:<11} {len(outs):>3} tested  "
              f"Active {n['Active']:>2} / Inactive {n['Inactive']:>2} / Inconclusive {n['Inconclusive']:>2}")
    tested = {r[0] for r in rows}
    print(f"\n  panel compounds with at least one measurement: {len(tested)}/11")
    print(f"  never tested anywhere: {sorted(set(PANEL.values()) - tested) or 'none'}")


if __name__ == "__main__":
    main()
