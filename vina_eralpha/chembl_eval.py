"""Same pocket, same methods, different answer key.

Against Tox21 ERalpha antagonist labels every method sat at chance: DrugCLIP
0.481, Vina 0.515, a 2D fingerprint 0.536. Measured binding itself only reaches
0.606 against those labels, so the target may simply not have been measurable.

This scores the same two methods against ChEMBL assay_type=B labels -- Ki/Kd/
IC50 measured directly against the protein, not a cell reporter. Active is
pChEMBL >= 6 (<=1uM), inactive < 5 (>10uM), sampled with at most three
compounds per Murcko scaffold so one optimised series cannot carry the result.

A jump here means the methods were fine and the answer key was wrong.
"""
import csv
import glob
import pickle
import random
import statistics as st
import sys

sys.path.insert(0, "../edc_finetune_data")
import lmdb
from eval_auroc import auroc

SAMPLE = "chembl_sample.csv"
DC_MATRIX = "../edc_finetune_data/chembl_eval_v2/score_matrix.csv"
DC_LMDB = "../edc_finetune_data/mols_chembl_panel.lmdb"
RECEPTOR = "ERalpha"


def vina():
    out = {}
    for f in glob.glob("chembldock_*.csv"):
        for r in csv.DictReader(open(f)):
            if r.get("vina_affinity"):
                out[r["cid"]] = -float(r["vina_affinity"])   # lower binds better
    return out


def drugclip():
    env = lmdb.open(DC_LMDB, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        smi2id = {pickle.loads(v)["smi"]: k.decode() for k, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open(DC_MATRIX)))
    ids = [smi2id.get(s) for s in rows[0][1:]]
    for r in rows[1:]:
        if r[0] == RECEPTOR:
            return {i: float(v) for i, v in zip(ids, r[1:]) if i}
    return {}


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    x, y = [ra[v] for v in a], [rb[v] for v in b]
    mx, my = st.mean(x), st.mean(y)
    num = sum((i - mx) * (j - my) for i, j in zip(x, y))
    den = (sum((i - mx) ** 2 for i in x) * sum((j - my) ** 2 for j in y)) ** 0.5
    return num / den if den else 0.0


def boot(pos, neg, n=4000):
    random.seed(0)
    v = sorted(auroc([random.choice(pos) for _ in pos],
                     [random.choice(neg) for _ in neg]) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n)]


def main():
    meta = {r["chembl_id"]: (r["label"], float(r["pchembl"]))
            for r in csv.DictReader(open(SAMPLE))}
    V, D = vina(), drugclip()
    ids = [i for i in meta if i in V and i in D]
    pos = [i for i in ids if meta[i][0] == "Active"]
    neg = [i for i in ids if meta[i][0] == "Inactive"]
    print(f"ChEMBL 결합 라벨, {RECEPTOR} 3UU7 포켓")
    print(f"  {len(ids)} compounds, {len(pos)}+/{len(neg)}-\n")

    print(f"{'방법':<14}{'ChEMBL AUROC':>14}   {'95% CI':<18}{'Tox21 (참고)':>14}")
    tox = {"Vina": 0.515, "DrugCLIP": 0.481}
    for name, S in [("Vina", V), ("DrugCLIP", D)]:
        p, n = [S[i] for i in pos], [S[i] for i in neg]
        lo, hi = boot(p, n)
        print(f"{name:<14}{auroc(p, n):>14.3f}   [{lo:.3f}, {hi:.3f}]   {tox[name]:>11.3f}")

    # the binarisation throws away most of the label, so check the continuous form too
    pch = [meta[i][1] for i in ids]
    print(f"\n연속값 상관 (점수 vs pChEMBL, Spearman)")
    for name, S in [("Vina", V), ("DrugCLIP", D)]:
        print(f"  {name:<12}{spearman([S[i] for i in ids], pch):+.3f}")
    print(f"  {'Vina vs DrugCLIP':<12}{spearman([V[i] for i in ids], [D[i] for i in ids]):+.3f}")


if __name__ == "__main__":
    main()
