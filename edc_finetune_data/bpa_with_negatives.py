"""Where does BPA land once real negatives are in the list?

Every BPA view so far ranked it among 11 structural cousins that are mostly
Active, or against 3 unverified randoms (bpa_panel/docking/igmodel_panel.csv).
Neither can say whether a high score means anything. This mixes in compounds
with MEASURED Tox21 outcomes at the same receptor, in two tiers:

  look-alike negative   BPA-like by substructure or Tanimoto, measured Inactive
  plain negative        measured Inactive, not BPA-like

Scores are merged from the two v2 matrices -- same checkpoint, same pockets,
same --max-pocket-atoms -- so they are directly comparable.

Compounds used in fine-tuning are marked: their scores were pushed down at
non-matching pockets, which flatters BPA's rank. The leak-free block repeats
the ranking with those dropped.
"""
import csv
import json
import pickle
from collections import defaultdict

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from eval_auroc import auroc

RDLogger.DisableLog("rdApp.*")

RECEPTORS = ["ERalpha", "AR", "PPARgamma", "PR", "THRbeta"]
ASSAYS = {"ERalpha": ["743078"], "AR": ["743053", "743063"],
          "PPARgamma": ["743191"], "PR": ["1347031"], "THRbeta": ["743066"]}
RANK = {"Active": 2, "Inactive": 1, "Inconclusive": 0}

BPA_SMI = "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"
CORE = Chem.MolFromSmarts("Oc1ccc(cc1)[*]c1ccc(O)cc1")
DIOL = Chem.MolFromSmarts("[OX2H]c1ccccc1.[OX2H]c1ccccc1")
_fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
BPA_FP = _fp(Chem.MolFromSmiles(BPA_SMI))


def load_panel(lmdb_path, matrix_path, key):
    """key: what to call each compound -- 'cid' or 'abbreviation'."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        meta = {}
        for k, v in txn.cursor():
            d = pickle.loads(v)
            meta[d["smi"]] = (d.get(key) or k.decode(), d["smi"])
    env.close()
    rows = list(csv.reader(open(matrix_path)))
    cols = [meta.get(s) for s in rows[0][1:]]
    scores = defaultdict(dict)
    for r in rows[1:]:
        for c, v in zip(cols, r[1:]):
            if c:
                scores[r[0]][c[0]] = float(v)
    return scores, {c[0]: c[1] for c in cols if c}


def tox21():
    out = defaultdict(dict)
    for receptor, aids in ASSAYS.items():
        for aid in aids:
            for row in json.load(open(f"raw/aid_{aid}.json"))["Table"]["Row"]:
                cid, o = row["Cell"][2], row["Cell"][3]
                if cid and (cid not in out[receptor] or RANK[o] > RANK[out[receptor][cid]]):
                    out[receptor][cid] = o
    return out


def bpa_like(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return (m.HasSubstructMatch(CORE) or m.HasSubstructMatch(DIOL)
            or DataStructs.TanimotoSimilarity(BPA_FP, _fp(m)) >= 0.4)


def main():
    edc, edc_smi = load_panel("mols_edc_panel.lmdb", "eval_v2/score_matrix.csv", "cid")
    bis, bis_smi = load_panel("../bpa_panel/mols_bpa_panel.lmdb",
                              "../bpa_panel/eval_v2/score_matrix.csv", "abbreviation")
    T = tox21()
    valid = set(json.load(open("valid_cids.json")))
    pairs = [(r["cid"], r["receptor"], r["label"]) for r in csv.DictReader(open("pairs.csv"))]
    trained = {c for c, _, l in pairs if l == "positive" and c not in valid}
    like = {cid for cid, smi in edc_smi.items() if bpa_like(smi)}

    for leak_free in (False, True):
        tag = "held-out compounds only" if leak_free else "all compounds (training ones marked *)"
        print(f"\n{'='*74}\nBPA vs measured negatives -- {tag}\n{'='*74}")
        print(f"{'receptor':<11}{'BPA':>8}{'rank':>12}{'%ile':>7}"
              f"{'look-alike neg':>16}{'plain neg':>11}{'AUROC':>8}")
        for rec in RECEPTORS:
            if rec not in bis or "BPA" not in bis[rec]:
                continue
            bpa_score = bis[rec]["BPA"]
            pos, neg_like, neg_plain = [], [], []
            for cid, outcome in T[rec].items():
                if cid not in edc.get(rec, {}):
                    continue
                if leak_free and cid in trained:
                    continue
                s = edc[rec][cid]
                if outcome == "Active":
                    pos.append(s)
                elif outcome == "Inactive":
                    (neg_like if cid in like else neg_plain).append(s)
            negs = neg_like + neg_plain
            if not negs:
                continue
            above = sum(1 for s in negs if s >= bpa_score)
            pct = 100.0 * (1 - above / len(negs))
            a = auroc(pos, negs) if pos else float("nan")
            star = "" if leak_free else "*"
            print(f"{rec:<11}{bpa_score:>8.3f}{f'{above+1}/{len(negs)+1}':>12}{pct:>6.1f}%"
                  f"{len(neg_like):>16}{len(neg_plain):>11}{a:>8.3f}")

    print("\nrank/%ile are BPA against the measured NEGATIVES only -- 'how many")
    print("confirmed non-binders outscore BPA'. AUROC is the measured actives")
    print("vs those same negatives, i.e. whether the ordering means anything at all.")

    # ---- look-alikes only -------------------------------------------------
    # Dropping the plain negatives leaves the middle difficulty: not BPA among
    # its mostly-Active cousins, and not BPA against a bag of unrelated
    # chemistry, but "can the model tell a binder from a non-binder INSIDE the
    # BPA chemotype". Actives are restricted to look-alikes too, so both sides
    # of the AUROC come from the same structural family.
    print(f"\n{'='*74}\nLook-alike compounds only -- BPA chemotype, both classes\n{'='*74}")
    print(f"{'receptor':<11}{'BPA':>8}{'rank':>10}{'%ile':>7}{'act':>6}{'neg':>6}{'AUROC':>8}   held-out")
    for rec in RECEPTORS:
        if rec not in bis or "BPA" not in bis[rec]:
            continue
        bpa_score = bis[rec]["BPA"]
        pos, neg, pos_ho, neg_ho = [], [], [], []
        for cid in like:
            outcome = T[rec].get(cid)
            if outcome not in ("Active", "Inactive") or cid not in edc.get(rec, {}):
                continue
            s_ = edc[rec][cid]
            (pos if outcome == "Active" else neg).append(s_)
            if cid not in trained:
                (pos_ho if outcome == "Active" else neg_ho).append(s_)
        if not neg:
            continue
        above = sum(1 for x in neg if x >= bpa_score)
        pct = 100.0 * (1 - above / len(neg))
        ho = (f"{auroc(pos_ho, neg_ho):.3f} ({len(pos_ho)}+/{len(neg_ho)}-)"
              if pos_ho and neg_ho else f"n/a ({len(pos_ho)}+/{len(neg_ho)}-)")
        print(f"{rec:<11}{bpa_score:>8.3f}{f'{above+1}/{len(neg)+1}':>10}{pct:>6.1f}%"
              f"{len(pos):>6}{len(neg):>6}{auroc(pos, neg) if pos else float('nan'):>8.3f}   {ho}")

    # the flagship receptor, compound by compound -- small enough to just read
    rec = "ERalpha"
    rows = []
    for cid in like:
        outcome = T[rec].get(cid)
        if outcome in ("Active", "Inactive") and cid in edc.get(rec, {}):
            rows.append((edc[rec][cid], cid, outcome, cid in trained))
    rows.append((bis[rec]["BPA"], "6623", T[rec].get("6623", "?"), False))
    rows.sort(reverse=True)
    print(f"\n{rec}, every look-alike with a measured outcome "
          f"(* = used in fine-tuning, <-- = BPA):")
    for score, cid, outcome, tr in rows:
        mark = " <-- BPA" if cid == "6623" else ""
        print(f"   {score:>7.3f}  CID {cid:<9} {outcome:<9}{'*' if tr else ' '}{mark}")


if __name__ == "__main__":
    main()
