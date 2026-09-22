"""Compound-level binding dataset from a cached ChEMBL pull, censored rows kept.

Filtering to standard_relation='=' throws away the ">10uM" style records, and
those are the most confident non-binders in the database -- on ERalpha that is
2,378 compounds with no exact record at all, against an inactive pool of 456.
Recovering them is most of the class balance.

A '>' record bounds the concentration from below, so it bounds pChEMBL from
above: ">30000 nM" means pChEMBL < 4.52. That is enough to call a compound
inactive and enough to rank it at the bottom, which is all a Spearman metric
needs.

Between-assay noise is 0.52 log (3.3x) by the repeat-measurement analysis, and
restricting to a single standard_type does not reduce it, so measurements are
pooled by median and no type filter is applied.

  python build_chembl_dataset.py chembl_eralpha_raw.json ERalpha chembl_eralpha_dataset.csv
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict

ACTIVE_P = 6.0      # <=1uM
INACTIVE_P = 5.0    # >10uM


def pchembl_from_nm(nm):
    return 9.0 - math.log10(nm) if nm and nm > 0 else None


def main(raw_path, receptor, out_path):
    acts = json.load(open(raw_path))
    print(f"{len(acts)}건 읽음")

    exact, censored, smiles = defaultdict(list), defaultdict(list), {}
    for a in acts:
        mol = a.get("molecule_chembl_id")
        if not mol:
            continue
        if a.get("canonical_smiles"):
            smiles.setdefault(mol, a["canonical_smiles"])
        rel = a.get("standard_relation")
        if rel == "=" and a.get("pchembl_value"):
            exact[mol].append(float(a["pchembl_value"]))
        elif rel == ">" and a.get("standard_units") == "nM":
            try:
                p = pchembl_from_nm(float(a["standard_value"]))
            except (TypeError, ValueError):
                continue
            if p is not None:
                censored[mol].append(p)      # pChEMBL is below this

    rows = []
    for mol in set(exact) | set(censored):
        if mol not in smiles:
            continue
        ex, cen = exact.get(mol, []), censored.get(mol, [])
        if ex:
            p = st.median(ex)
            # an exact measurement beats a bound
            label = ("Active" if p >= ACTIVE_P
                     else "Inactive" if p < INACTIVE_P else "")
            rows.append([mol, receptor, f"{p:.3f}", "", len(ex), len(cen),
                         label, "exact", smiles[mol]])
        else:
            # censored only: the tightest bound is the smallest upper limit
            ub = min(cen)
            label = "Inactive" if ub < INACTIVE_P else ""
            rows.append([mol, receptor, "", f"{ub:.3f}", 0, len(cen),
                         label, "censored", smiles[mol]])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "receptor", "pchembl_median", "pchembl_upper_bound",
                    "n_exact", "n_censored", "label", "kind", "smiles"])
        w.writerows(rows)

    n_act = sum(1 for r in rows if r[6] == "Active")
    n_ina = sum(1 for r in rows if r[6] == "Inactive")
    n_cen_ina = sum(1 for r in rows if r[6] == "Inactive" and r[7] == "censored")
    n_mid = sum(1 for r in rows if not r[6])
    print(f"\n{len(rows)}개 화합물 -> {out_path}")
    print(f"  Active   {n_act}")
    print(f"  Inactive {n_ina}   (그중 검열 기록으로 복구된 것 {n_cen_ina})")
    print(f"  중간대(5~6) 제외 {n_mid}")
    print(f"\n복구 전 Inactive는 {n_ina - n_cen_ina}개였습니다.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
