"""Merge Tox21 qHTS assay pulls into (compound, receptor, label) pairs for
hard-negative fine-tuning. BPA/bisphenol-panel compounds are excluded from
here entirely -- they're reserved for final held-out validation, per plan.
"""
import csv
import json
import os

RAW_DIR = "raw"

# receptor -> list of (AID, positive-outcome-meaning)
# AR has separate agonist/antagonist assays; either counts as "this compound
# is active at this receptor" for our purposes (we care about "does it touch
# this receptor" for hard-negative construction, not agonist vs antagonist).
ASSAYS = {
    "ERalpha": ["743078"],
    "AR": ["743053", "743063"],
    "PPARgamma": ["743191"],
    "PR": ["1347031"],
    "THRbeta": ["743066"],
}

# our 11-compound bisphenol panel (bpa_panel/compounds.csv) -- excluded from
# fine-tuning data entirely, reserved for final validation only.
BISPHENOL_CIDS = {
    "6623", "6626", "12111", "73864", "66166", "608116",
    "232446", "6620", "623849", "6618", "6619",
}


def load_outcomes(aid):
    path = os.path.join(RAW_DIR, f"aid_{aid}.json")
    d = json.load(open(path))
    cid_to_outcome = {}
    for row in d["Table"]["Row"]:
        c = row["Cell"]
        cid, outcome = c[2], c[3]
        if not cid or cid in BISPHENOL_CIDS:
            continue
        # if a compound has multiple records in one assay (rare), keep the
        # more informative outcome (Active > Inactive > Inconclusive)
        rank = {"Active": 2, "Inactive": 1, "Inconclusive": 0}
        if cid not in cid_to_outcome or rank[outcome] > rank[cid_to_outcome[cid]]:
            cid_to_outcome[cid] = outcome
    return cid_to_outcome


def main():
    # receptor -> {cid: outcome}, combining multiple assays per receptor
    per_receptor = {}
    for receptor, aids in ASSAYS.items():
        combined = {}
        for aid in aids:
            outcomes = load_outcomes(aid)
            for cid, outcome in outcomes.items():
                rank = {"Active": 2, "Inactive": 1, "Inconclusive": 0}
                if cid not in combined or rank[outcome] > rank[combined[cid]]:
                    combined[cid] = outcome
        per_receptor[receptor] = combined
        print(f"{receptor}: {len(combined)} compounds "
              f"({sum(v == 'Active' for v in combined.values())} active)")

    # all compounds active at >=1 receptor in the panel
    active_anywhere = set()
    for combined in per_receptor.values():
        active_anywhere |= {cid for cid, o in combined.items() if o == "Active"}
    print(f"\ncompounds active at >=1 panel receptor: {len(active_anywhere)}")

    rows = []
    for receptor, combined in per_receptor.items():
        for cid, outcome in combined.items():
            if outcome == "Active":
                rows.append((cid, receptor, "positive"))
            elif outcome == "Inactive" and cid in active_anywhere:
                # hard negative: a real binder (somewhere in the panel) that
                # is confirmed NOT active at this specific receptor
                rows.append((cid, receptor, "hard_negative"))
            # plain Inactive (never active anywhere) and Inconclusive: skip
            # for now -- not useful for hard-negative contrastive pairs

    with open("pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", "receptor", "label"])
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r[2] == "positive")
    n_hard_neg = sum(1 for r in rows if r[2] == "hard_negative")
    print(f"\nwrote {len(rows)} pairs to pairs.csv ({n_pos} positive, {n_hard_neg} hard_negative)")
    print(f"unique compounds involved: {len(set(r[0] for r in rows))}")


if __name__ == "__main__":
    main()
