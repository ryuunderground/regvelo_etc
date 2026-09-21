"""Does the encoder see conformers at all, and does the pocket change which one wins?

Two questions, both answered straight off a conformer-expanded score matrix
(expand_conformers.py + score_matrix.py), no training required.

1. SENSITIVITY -- how far apart do one compound's conformers score, next to how
   far apart different compounds score? A within/between ratio near zero means
   the distance-matrix input barely registers the shape change, which is the
   9/10 finding (conformer weights stayed uniform, weighted score moved 9.5e-5)
   reproduced for free.

2. POCKET-DEPENDENCE -- do different receptors pick different best conformers?
   The planned architecture pools conformers under a pocket-conditioned
   attention query, which only buys something if the winning shape actually
   depends on the pocket. If one conformer wins everywhere, conditioning the
   pool on e_pkt has nothing to learn and the pool may as well be a mean.

Usage: python conformer_spread.py eval_conf/score_matrix.csv
"""
import csv
import statistics
import sys
from collections import defaultdict


def load(path):
    """-> pocket names, and per pocket a {compound: [score per conformer]}."""
    rows = list(csv.reader(open(path)))
    smis = rows[0][1:]
    groups = defaultdict(list)
    for col, smi in enumerate(smis):
        groups[smi].append(col)
    pockets, per_pocket = [], []
    for r in rows[1:]:
        vals = [float(v) for v in r[1:]]
        pockets.append(r[0])
        per_pocket.append({smi: [vals[c] for c in cols] for smi, cols in groups.items()})
    return pockets, per_pocket


def main(path):
    pockets, per_pocket = load(path)
    multi = [s for s, v in per_pocket[0].items() if len(v) > 1]
    print(f"{len(per_pocket[0])} compounds, {len(multi)} with >1 conformer, "
          f"{len(pockets)} pockets\n")

    print(f"{'pocket':<12}{'within(max-min)':>17}{'between(sd)':>13}{'ratio':>8}")
    for name, scores in zip(pockets, per_pocket):
        within = statistics.mean(max(scores[s]) - min(scores[s]) for s in multi)
        between = statistics.stdev([statistics.mean(v) for v in scores.values()])
        print(f"{name:<12}{within:>17.4f}{between:>13.4f}{within / between:>8.2f}")

    # Do two pockets crown the same conformer more often than chance (1/n)?
    print(f"\n{'pocket pair agreement on best conformer':<44}{'obs':>8}{'chance':>8}")
    agree = chance = 0
    for s in multi:
        n = len(per_pocket[0][s])
        best = [max(range(n), key=lambda i: sc[s][i]) for sc in per_pocket]
        pairs = [(a, b) for a in range(len(pockets)) for b in range(a + 1, len(pockets))]
        agree += sum(best[a] == best[b] for a, b in pairs) / len(pairs)
        chance += 1 / n
    print(f"{'  mean over compounds with >1 conformer':<44}"
          f"{agree / len(multi):>8.3f}{chance / len(multi):>8.3f}")
    print("\nagreement ~= chance -> pocket picks its own shape (pool worth conditioning)")
    print("agreement ~= 1.0    -> one shape wins everywhere (pocket-conditioning is dead weight)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_conf/score_matrix.csv")
