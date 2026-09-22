"""Binding data for the whole bisphenol panel, not just BPA.

BPA turned out to have a measured profile across receptors -- ERRgamma at 5.5nM,
ERalpha and ERbeta near 700-800nM, AR in the tens of micromolar. The panel's
other ten compounds are the flagship case study, so the same question applies to
them: is there a measured answer key, or only Tox21 cell calls?

Looks each compound up in ChEMBL by exact structure (its PubChem SMILES,
canonicalised), then pulls every activity and keeps the nuclear receptor ones.
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PANEL_CSV = "../bpa_panel/compounds.csv"

# ChEMBL target ids seen carrying BPA data, plus the panel's receptors
NUCLEAR = {
    "CHEMBL206": "ERα", "CHEMBL242": "ERβ", "CHEMBL4245": "ERRγ",
    "CHEMBL1871": "AR", "CHEMBL3072": "AR(2)", "CHEMBL2034": "GR",
    "CHEMBL235": "PPARγ", "CHEMBL208": "PR", "CHEMBL1909044": "PR(2)",
    "CHEMBL1947": "THRβ", "CHEMBL2061": "RXRα", "CHEMBL1741186": "RORγ",
    "CHEMBL3401": "ERRα", "CHEMBL1994": "VDR", "CHEMBL3401 ": "-",
}


def get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception:                                   # noqa: BLE001
            time.sleep(4 * (i + 1))
    return None


def chembl_id_for(smiles):
    """Exact-structure lookup. ChEMBL canonicalises, so the raw SMILES is fine."""
    d = get(f"{BASE}/molecule.json?molecule_structures__canonical_smiles__flexmatch="
            f"{urllib.parse.quote(smiles)}&limit=5")
    if not d or not d.get("molecules"):
        return None
    return d["molecules"][0]["molecule_chembl_id"]


def activities(mol_id):
    out, offset = [], 0
    while True:
        d = get(f"{BASE}/activity.json?molecule_chembl_id={mol_id}"
                f"&limit=1000&offset={offset}")
        if not d:
            break
        out += d["activities"]
        total = d["page_meta"]["total_count"]
        offset += 1000
        if offset >= total:
            break
        time.sleep(0.3)
    return out


def main():
    panel = list(csv.DictReader(open(PANEL_CSV)))
    rows = []
    for r in panel:
        name, smi = r["abbreviation"], r["smiles"]
        cid = chembl_id_for(smi)
        if not cid:
            print(f"{name:<7} ChEMBL에 없음", flush=True)
            rows.append((name, None, {}))
            time.sleep(0.5)
            continue
        acts = activities(cid)
        per = defaultdict(list)
        for a in acts:
            t = a.get("target_chembl_id")
            if t in NUCLEAR and a.get("pchembl_value"):
                per[NUCLEAR[t]].append(float(a["pchembl_value"]))
        summary = " ".join(f"{k} {max(v):.2f}" for k, v in sorted(per.items()))
        print(f"{name:<7} {cid:<16} 활성 {len(acts):>4}건   {summary or '핵수용체 pChEMBL 없음'}",
              flush=True)
        rows.append((name, cid, dict(per)))
        time.sleep(0.5)

    with open("bisphenol_chembl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["abbreviation", "chembl_id", "receptor", "pchembl_values"])
        for name, cid, per in rows:
            for rec, vals in sorted(per.items()):
                w.writerow([name, cid, rec, ";".join(f"{v:.2f}" for v in vals)])
    print("\nbisphenol_chembl.csv 저장")


if __name__ == "__main__":
    sys.exit(main())
