"""AUROC for a score_matrix.csv against pairs.csv labels.

Reports per-receptor AUROC, not just pooled. Pooled mixes scores from different
pockets, so it picks up receptor-level score offsets rather than the ranking
ability that virtual screening actually uses -- on this panel pooled reads
~0.06 higher than every individual receptor except THRbeta, which carries 1143
negatives against 4 positives.

Two evaluation sets:
  held-out-positives  positives from the valid CID split vs ALL hard negatives
                      (what auroc_before_after.csv reported)
  leak-free           both classes restricted to valid CIDs, so no compound the
                      model trained on appears on either side

Each argument is a matrix path, optionally suffixed ":conf0", ":max" or
":mean" to pool a conformer-expanded matrix (see expand_conformers.py) back
per compound. Plain matrices ignore the suffix.

Usage: python eval_auroc.py eval_new/score_matrix.csv eval_conf/score_matrix.csv:max
"""
import csv
import json
import math
import pickle
import sys
from collections import defaultdict

import lmdb

RECEPTORS = ["ERalpha", "AR", "PPARgamma", "PR", "THRbeta"]

# Conformer pooling, for matrices built from mols_edc_conformers.lmdb where a
# compound owns one column per conformer. "conf0" is the no-ensembling control:
# it isolates "averaging over shapes helps" from "this shape beats the single
# arbitrary one AffinityMolDataset would have drawn".
POOLS = {
    "conf0": lambda v: v[0],
    "max": max,
    "mean": lambda v: sum(v) / len(v),
}


def auroc(pos, neg):
    """Mann-Whitney U with midranks for ties."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    vals = [v for v, _ in merged]
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1
        i = j + 1
    s_pos = sum(ranks[k] for k in range(len(merged)) if merged[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (s_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def bedroc(pos, neg, alpha=80.5):
    """Truchon-Bayly BEDROC: AUROC averages every pair, this weights the top.

    9/13 moved pair-ranking up (48.00 -> 49.92%) while top-1 selection fell
    (12.20 -> 4.88%), so a pair-averaged metric alone cannot tell whether an
    ensembling change actually helps screening.
    """
    n, total = len(pos), len(pos) + len(neg)
    if n == 0 or n == total:
        return float("nan")
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], reverse=True)
    ra = n / total
    s = sum(math.exp(-alpha * (i + 1) / total)
            for i, (_, lab) in enumerate(ranked) if lab == 1)
    rie = (s / n) / ((1 - math.exp(-alpha)) / (total * (math.exp(alpha / total) - 1)))
    scale = ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
    return rie * scale + 1 / (1 - math.exp(alpha * (1 - ra)))


METRICS = {"AUROC": auroc, "BEDROC": bedroc}


def load_smiles_to_cid(path="mols_edc_panel.lmdb"):
    """score_matrix columns are RDKit-canonical SMILES, pairs.csv keys are CIDs."""
    env = lmdb.open(path, readonly=True, lock=False, subdir=False)
    out = {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            out[pickle.loads(v)["smi"]] = k.decode()
    env.close()
    return out


def load_matrix(path, smi2cid, pool="conf0"):
    """Columns are SMILES; a conformer-expanded matrix repeats one per shape.

    A plain matrix has one column per compound, so every pool agrees on it.
    """
    rows = list(csv.reader(open(path)))
    cids = [smi2cid.get(s) for s in rows[0][1:]]
    fn = POOLS[pool]
    out = {}
    for r in rows[1:]:
        by_cid = defaultdict(list)
        for c, v in zip(cids, r[1:]):
            if c:
                by_cid[c].append(float(v))
        out[r[0]] = {c: fn(v) for c, v in by_cid.items()}
    return out


def evaluate(matrix, pairs, valid_cids, leak_free):
    pooled = ([], [])
    per = defaultdict(lambda: ([], []))
    for cid, receptor, label in pairs:
        if receptor not in matrix or cid not in matrix[receptor]:
            continue
        if label == "positive" and cid not in valid_cids:
            continue  # positive must be held out in both modes
        if leak_free and cid not in valid_cids:
            continue  # negatives too
        score = matrix[receptor][cid]
        side = 0 if label == "positive" else 1
        pooled[side].append(score)
        per[receptor][side].append(score)
    return pooled, per


def main(specs):
    smi2cid = load_smiles_to_cid()
    valid_cids = set(json.load(open("valid_cids.json")))
    pairs = [(r["cid"], r["receptor"], r["label"])
             for r in csv.DictReader(open("pairs.csv"))]

    loaded = []
    for spec in specs:
        path, _, pool = spec.partition(":")
        loaded.append((spec, load_matrix(path, smi2cid, pool or "conf0")))

    for mode, leak_free in [("held-out-positives", False), ("leak-free", True)]:
        for name, metric in METRICS.items():
            print(f"\n### {mode} - {name}")
            print(f"{'matrix':<44}{'pooled':>9}" + "".join(f"{r:>12}" for r in RECEPTORS))
            for spec, matrix in loaded:
                pooled, per = evaluate(matrix, pairs, valid_cids, leak_free)
                cells = "".join(f"{metric(*per[r]):>12.3f}" for r in RECEPTORS)
                print(f"{spec:<44}{metric(*pooled):>9.4f}{cells}")
        n = [f"{r} {len(per[r][0])}+/{len(per[r][1])}-" for r in RECEPTORS]
        print(f"{'  n:':<44}{sum(len(x) for x in pooled):>9}  " + "  ".join(n))


if __name__ == "__main__":
    args = sys.argv[1:] or ["eval_old/score_matrix.csv", "eval_new/score_matrix.csv"]
    main(args)
