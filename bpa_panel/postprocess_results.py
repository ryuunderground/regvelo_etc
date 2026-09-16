"""Turn score_matrix.csv (raw SMILES headers) into a readable ranked result:
abbreviation columns, plus each pocket's within-row percentile rank for every
compound (matches the "top 43%" framing from the earlier lab discussion)."""
import csv

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def main():
    compounds = list(csv.DictReader(open("compounds.csv")))
    smi_to_abbr = {canon(c["smiles"]): c["abbreviation"] for c in compounds}

    with open("score_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    mol_smis = header[1:]
    abbrs = [smi_to_abbr.get(s, s) for s in mol_smis]

    # readable score matrix
    with open("results_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receptor"] + abbrs)
        for row in rows:
            w.writerow(row)

    # long format with within-pocket percentile rank (1 = best)
    with open("results_ranked.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receptor", "compound", "score", "rank_in_receptor", "percentile_in_receptor"])
        for row in rows:
            receptor = row[0]
            scores = [(abbrs[i], float(v)) for i, v in enumerate(row[1:])]
            scores.sort(key=lambda x: -x[1])
            n = len(scores)
            for rank, (compound, score) in enumerate(scores, start=1):
                pct = 100 * (n - rank + 1) / n
                w.writerow([receptor, compound, f"{score:.4f}", rank, f"{pct:.1f}"])

    print("wrote results_scores.csv and results_ranked.csv")


if __name__ == "__main__":
    main()
