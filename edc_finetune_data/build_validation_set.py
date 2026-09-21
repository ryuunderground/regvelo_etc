"""Build the validation set defined in VALIDATION.md.

Reads library_smiles.csv (the whole Tox21 library, not just the fine-tuning
pool) and emits validation_set.csv: one row per compound with its tier, its
measured outcome at each receptor, and whether it is contaminated by training.

Tiers (see VALIDATION.md for the reasoning):
  T1  the 11-compound bisphenol panel
  T2  BPA chemotype -- bisphenol core, two phenols, or Tanimoto(BPA) >= 0.4
  T3  MW 228.29 +-30, NOT chemotype, definite call at all four profile
      receptors, and Active at >=1 of them -- it binds something, just not
      the way BPA does
      T3a  Hamming distance from BPA's profile >= 3
      T3b  distance <= 2
  T4  same MW/chemotype filter but Inactive at all four -- binds nothing

The "Active somewhere" condition on T3 is load-bearing. BPA is Inactive at
THRbeta, so a compound inactive everywhere sits at distance 3 automatically:
without the condition T3a fills with 842 non-binders and stops meaning
"different binding pattern". Those belong in T4, where they are useful for a
different reason -- pairs.csv only ever held compounds active somewhere, so
T4 is almost entirely free of training contamination.
"""
import csv
import json
import os
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

RDLogger.DisableLog("rdApp.*")

RAW_DIR = "raw"
ASSAYS = {"ERalpha": ["743078"], "AR": ["743053", "743063"],
          "PPARgamma": ["743191"], "PR": ["1347031"], "THRbeta": ["743066"]}
# PPARgamma is excluded from the profile: BPA is Inconclusive there.
PROFILE = ["ERalpha", "AR", "PR", "THRbeta"]
BPA_PROFILE = ("Active", "Active", "Active", "Inactive")
BPA_CID = "6623"
BPA_MW = 228.29
MW_BAND = 30.0
RANK = {"Active": 2, "Inactive": 1, "Inconclusive": 0}

PANEL = {"6623": "BPA", "6626": "BPS", "12111": "BPF", "73864": "BPAF",
         "66166": "BPB", "608116": "BPE", "232446": "BPZ", "6620": "BPC",
         "623849": "BPAP", "6618": "TBBPA", "6619": "TCBPA"}

CORE = Chem.MolFromSmarts("Oc1ccc(cc1)[*]c1ccc(O)cc1")
DIOL = Chem.MolFromSmarts("[OX2H]c1ccccc1.[OX2H]c1ccccc1")
_fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
BPA_FP = _fp(Chem.MolFromSmiles("CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"))


def outcomes():
    out = defaultdict(dict)
    for receptor, aids in ASSAYS.items():
        for aid in aids:
            for row in json.load(open(f"{RAW_DIR}/aid_{aid}.json"))["Table"]["Row"]:
                cid, o = row["Cell"][2], row["Cell"][3]
                if cid and (cid not in out[receptor] or RANK[o] > RANK[out[receptor][cid]]):
                    out[receptor][cid] = o
    return out


def main():
    T = outcomes()
    lib = list(csv.DictReader(open("library_smiles.csv")))
    print(f"{len(lib)} compounds in library_smiles.csv")

    # contamination: compounds that went into train.lmdb
    valid = set(json.load(open("valid_cids.json")))
    pairs = [(r["cid"], r["label"]) for r in csv.DictReader(open("pairs.csv"))]
    trained = {c for c, l in pairs if l == "positive" and c not in valid}

    rows, bad = [], 0
    for r in lib:
        cid, smi = r["cid"], r["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            continue
        mw = Descriptors.MolWt(mol)
        chemotype = (mol.HasSubstructMatch(CORE) or mol.HasSubstructMatch(DIOL)
                     or DataStructs.TanimotoSimilarity(BPA_FP, _fp(mol)) >= 0.4)
        prof = tuple(T[x].get(cid) for x in PROFILE)
        complete = all(p in ("Active", "Inactive") for p in prof)
        dist = sum(a != b for a, b in zip(prof, BPA_PROFILE)) if complete else ""

        if cid in PANEL:
            tier = "T1"
        elif chemotype:
            tier = "T2"
        elif abs(mw - BPA_MW) <= MW_BAND and complete:
            binds = any(p == "Active" for p in prof)
            tier = ("T3a" if dist >= 3 else "T3b") if binds else "T4"
        else:
            continue

        rows.append({
            "cid": cid, "tier": tier, "abbreviation": PANEL.get(cid, ""),
            "molecular_weight": f"{mw:.2f}",
            "tanimoto_bpa": f"{DataStructs.TanimotoSimilarity(BPA_FP, _fp(mol)):.3f}",
            "profile_distance": dist, "in_training": int(cid in trained),
            "smiles": smi,
            **{f"tox21_{x}": T[x].get(cid, "") for x in ASSAYS},
        })

    fields = ["cid", "tier", "abbreviation", "molecular_weight", "tanimoto_bpa",
              "profile_distance", "in_training", "smiles"] + [f"tox21_{x}" for x in ASSAYS]
    with open("validation_set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["tier"], -float(r["tanimoto_bpa"]))))
    if bad:
        print(f"{bad} SMILES failed to parse")
    print(f"wrote {len(rows)} compounds to validation_set.csv\n")

    counts = Counter(r["tier"] for r in rows)
    clean = Counter(r["tier"] for r in rows if not r["in_training"])
    print(f"{'tier':<6}{'total':>8}{'clean':>8}")
    for t in ["T1", "T2", "T3a", "T3b", "T4"]:
        print(f"{t:<6}{counts[t]:>8}{clean[t]:>8}")

    print(f"\nprofile distance among T3 (BPA = {BPA_PROFILE} over {PROFILE}):")
    d = Counter(r["profile_distance"] for r in rows if r["tier"].startswith("T3"))
    for k in sorted(d):
        print(f"  d={k}  {d[k]:>4}")

    print("\nmeasured outcomes per receptor, by tier  (A=Active / I=Inactive)")
    print(f"{'receptor':<11}" + "".join(f"{t+' A':>9}{t+' I':>9}" for t in ["T2", "T3a", "T3b", "T4"]))
    for rec in ASSAYS:
        cells = []
        for t in ["T2", "T3a", "T3b", "T4"]:
            sub = [r for r in rows if r["tier"] == t]
            cells += [sum(1 for r in sub if r[f"tox21_{rec}"] == "Active"),
                      sum(1 for r in sub if r[f"tox21_{rec}"] == "Inactive")]
        print(f"{rec:<11}" + "".join(f"{c:>9}" for c in cells))


if __name__ == "__main__":
    main()
