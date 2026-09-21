"""Does DrugCLIP + Vina beat either alone, inside the BPA chemotype?

Fusion can only amplify signal that already exists, so it is worth testing only
where at least one component has some. On the general hard set neither does --
Vina 0.515, DrugCLIP 0.481 scaffold-clean -- but T2 is the BPA chemotype, where
DrugCLIP reads ~0.6 and Glide read 0.911 on the bisphenols. The two are also
nearly uncorrelated (Spearman +0.148 over 148 compounds), which is the
precondition for combining them to add anything.

Scores are z-scored before summing because a cosine and a kcal/mol-ish affinity
are not on the same scale. Vina is negated first (lower binds better).
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

VALSET = "../edc_finetune_data/validation_set.csv"
DC_MATRIX = "../edc_finetune_data/t2_eval_v2/score_matrix.csv"
DC_LMDB = "../edc_finetune_data/mols_t2_panel.lmdb"
RECEPTOR = "ERalpha"


def vina_scores():
    out = {}
    for f in glob.glob("t2_*.csv"):
        for r in csv.DictReader(open(f)):
            if r["vina_affinity"]:
                out[r["cid"]] = -float(r["vina_affinity"])
    return out


def drugclip_scores():
    env = lmdb.open(DC_LMDB, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        smi2cid = {pickle.loads(v)["smi"]: k.decode() for k, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open(DC_MATRIX)))
    cids = [smi2cid.get(s) for s in rows[0][1:]]
    for r in rows[1:]:
        if r[0] == RECEPTOR:
            return {c: float(v) for c, v in zip(cids, r[1:]) if c}
    return {}


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    x, y = [ra[v] for v in a], [rb[v] for v in b]
    mx, my = st.mean(x), st.mean(y)
    num = sum((i - mx) * (j - my) for i, j in zip(x, y))
    den = (sum((i - mx) ** 2 for i in x) * sum((j - my) ** 2 for j in y)) ** 0.5
    return num / den if den else 0.0


def boot_ci(pos, neg, n=2000):
    random.seed(0)
    vals = sorted(auroc([random.choice(pos) for _ in pos],
                        [random.choice(neg) for _ in neg]) for _ in range(n))
    return vals[50], vals[1949]


def main():
    labels = {r["cid"]: r["tox21_ERalpha"] for r in csv.DictReader(open(VALSET))
              if r["tier"] == "T2"}
    V, D = vina_scores(), drugclip_scores()
    cids = [c for c in V if c in D and labels.get(c) in ("Active", "Inactive")]
    pos = [c for c in cids if labels[c] == "Active"]
    neg = [c for c in cids if labels[c] == "Inactive"]
    print(f"T2, {RECEPTOR}: {len(cids)} compounds, {len(pos)}+/{len(neg)}-")
    print(f"DrugCLIP vs Vina Spearman: {spearman([V[c] for c in cids], [D[c] for c in cids]):+.3f}\n")

    def z(d):
        vals = [d[c] for c in cids]
        m, s = st.mean(vals), st.pstdev(vals) or 1.0
        return {c: (d[c] - m) / s for c in cids}

    zv, zd = z(V), z(D)
    scorers = [
        ("DrugCLIP 단독", lambda c: D[c]),
        ("Vina 단독", lambda c: V[c]),
        ("융합 z합 1:1", lambda c: zv[c] + zd[c]),
        ("융합 Vina 2 : DC 1", lambda c: 2 * zv[c] + zd[c]),
        ("융합 Vina 1 : DC 2", lambda c: zv[c] + 2 * zd[c]),
    ]
    print(f"{'점수':<22}{'AUROC':>8}   95% CI")
    for name, f in scorers:
        p, n = [f(c) for c in pos], [f(c) for c in neg]
        lo, hi = boot_ci(p, n)
        print(f"{name:<22}{auroc(p, n):>8.3f}   [{lo:.3f}, {hi:.3f}]")

    # paired bootstrap on the thing that actually matters: fusion minus the best single
    print("\n융합 − 단독 (paired bootstrap):")
    for name, f in scorers[2:]:
        random.seed(0)
        obs = auroc([f(c) for c in pos], [f(c) for c in neg])
        base = max(auroc([D[c] for c in pos], [D[c] for c in neg]),
                   auroc([V[c] for c in pos], [V[c] for c in neg]))
        ds = []
        for _ in range(2000):
            bp = [random.choice(pos) for _ in pos]
            bn = [random.choice(neg) for _ in neg]
            a = auroc([f(c) for c in bp], [f(c) for c in bn])
            b = max(auroc([D[c] for c in bp], [D[c] for c in bn]),
                    auroc([V[c] for c in bp], [V[c] for c in bn]))
            ds.append(a - b)
        ds.sort()
        lo, hi = ds[50], ds[1949]
        flag = "" if lo < 0 < hi else "   <- CI가 0 제외"
        print(f"  {name:<20}{obs - base:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")


if __name__ == "__main__":
    main()
