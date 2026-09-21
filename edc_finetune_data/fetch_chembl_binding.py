"""Pull ChEMBL direct-binding activities for a target, as an alternative label set.

Every method tried so far sits at chance against the Tox21 ERalpha labels --
DrugCLIP 0.481, Vina 0.515, a 2D fingerprint 0.536. Three unrelated approaches
failing identically points at the labels rather than the methods, and Tox21
antagonist activity is a cell reporter readout: a compound can score Active
through cytotoxicity or reporter interference without ever touching the LBD.

ChEMBL assay_type=B is direct binding, measured as Ki/Kd/IC50 against the
protein. If a structure-based method does markedly better on these labels than
on Tox21, the label semantics were the problem.

  python fetch_chembl_binding.py CHEMBL206 chembl_eralpha.csv

pchembl_value is ChEMBL's own -log10(molar) for the standard types, so it is
used directly rather than recomputed. Compounds with several measurements keep
the median. SMILES come off the activity records themselves -- the separate
molecule lookup this used to do needed comma-separated ids, and with the wrong
separator it silently returned one molecule per request instead of fifty.
"""
import csv
import json
import statistics as st
import sys
import time
import urllib.request
from collections import defaultdict

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000


def get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:                          # noqa: BLE001
            if attempt == retries - 1:
                print(f"  failed: {e}")
                return None
            time.sleep(3 * (attempt + 1))
    return None


def activities(target):
    """Binding assays only, with an exact relation and a pChEMBL value."""
    out, offset = [], 0
    while True:
        url = (f"{BASE}/activity.json?target_chembl_id={target}&assay_type=B"
               f"&standard_relation==&limit={PAGE}&offset={offset}")
        d = get(url)
        if not d:
            break
        rows = d["activities"]
        out += [a for a in rows if a.get("pchembl_value")]
        total = d["page_meta"]["total_count"]
        offset += PAGE
        print(f"  {min(offset, total)}/{total} scanned, {len(out)} usable", flush=True)
        if offset >= total:
            break
        time.sleep(0.3)
    return out



def main(target, out_path):
    print(f"activities for {target}:")
    acts = activities(target)
    by_mol = defaultdict(list)
    smi = {}
    for a in acts:
        mol = a["molecule_chembl_id"]
        by_mol[mol].append(float(a["pchembl_value"]))
        if a.get("canonical_smiles"):
            smi.setdefault(mol, a["canonical_smiles"])
    print(f"\n{len(acts)} measurements over {len(by_mol)} compounds, "
          f"{len(smi)} with SMILES")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "pchembl_median", "n_measurements", "smiles"])
        for mol, vals in sorted(by_mol.items()):
            if mol in smi:
                w.writerow([mol, f"{st.median(vals):.3f}", len(vals), smi[mol]])
    n = sum(1 for _ in open(out_path)) - 1
    print(f"\nwrote {n} compounds to {out_path}")

    vals = [st.median(v) for m, v in by_mol.items() if m in smi]
    vals.sort()
    print(f"pChEMBL: min {vals[0]:.2f}  25% {vals[len(vals)//4]:.2f}  "
          f"median {vals[len(vals)//2]:.2f}  75% {vals[3*len(vals)//4]:.2f}  max {vals[-1]:.2f}")
    print(f"  >=6 (<=1uM, 'active'): {sum(1 for v in vals if v >= 6)}")
    print(f"  <5  (>10uM, 'inactive'): {sum(1 for v in vals if v < 5)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
