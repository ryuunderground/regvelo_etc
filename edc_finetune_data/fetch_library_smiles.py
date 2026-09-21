"""Fetch SMILES for the WHOLE Tox21 library, not just the compounds in pairs.csv.

compounds_smiles.csv only covers compounds active at >=1 receptor, which is the
fine-tuning pool. VALIDATION.md needs the other side too: the T2/T3 tiers are
mostly measured non-binders, and those are exactly the rows pairs.csv drops.

Writes library_smiles.csv and leaves compounds_smiles.csv alone -- build_mols.py
still reads that one, so nothing downstream shifts under us.

Resumable: an existing library_smiles.csv is read back and only missing CIDs
are requested.
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

RAW_DIR = "raw"
OUT = "library_smiles.csv"
BATCH = 100
PAUSE = 0.25          # PubChem asks for <=5 requests/sec
RETRIES = 3


def library_cids():
    """Every CID with a measured outcome in any assay we pulled."""
    cids = set()
    for name in sorted(os.listdir(RAW_DIR)):
        if not name.startswith("aid_") or not name.endswith(".json"):
            continue
        d = json.load(open(os.path.join(RAW_DIR, name)))
        n0 = len(cids)
        for row in d["Table"]["Row"]:
            cid = row["Cell"][2]
            if cid:
                cids.add(cid)
        print(f"  {name}: {len(d['Table']['Row'])} rows, running total {len(cids)} (+{len(cids)-n0})")
    return sorted(cids, key=int)


def fetch(batch):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           + ",".join(batch) + "/property/IsomericSMILES,MolecularWeight/JSON")
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["PropertyTable"]["Properties"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as e:
            if attempt == RETRIES - 1:
                print(f"    give up on batch starting {batch[0]}: {e}")
                return []
            time.sleep(2 ** attempt)
    return []


def main():
    print("collecting CIDs from raw assay pulls:")
    cids = library_cids()
    print(f"\n{len(cids)} unique CIDs in the library")

    have = {}
    if os.path.exists(OUT):
        have = {r["cid"]: r for r in csv.DictReader(open(OUT))}
        print(f"{len(have)} already in {OUT}, resuming")

    todo = [c for c in cids if c not in have]
    print(f"{len(todo)} to fetch, {(len(todo) + BATCH - 1) // BATCH} requests\n")

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        for p in fetch(batch):
            smi = (p.get("IsomericSMILES") or p.get("SMILES")
                   or p.get("ConnectivitySMILES") or p.get("CanonicalSMILES"))
            if smi:
                have[str(p["CID"])] = {"cid": str(p["CID"]), "smiles": smi,
                                       "molecular_weight": p.get("MolecularWeight", "")}
        done = min(i + BATCH, len(todo))
        if done % 1000 < BATCH or done == len(todo):
            print(f"  {done}/{len(todo)} requested, {len(have)} SMILES held")
        time.sleep(PAUSE)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cid", "smiles", "molecular_weight"])
        w.writeheader()
        for cid in cids:
            if cid in have:
                w.writerow(have[cid])

    missing = [c for c in cids if c not in have]
    print(f"\nwrote {len(have)}/{len(cids)} to {OUT}")
    if missing:
        open("library_missing_cids.txt", "w").write("\n".join(missing))
        print(f"{len(missing)} CIDs had no SMILES -> library_missing_cids.txt")


if __name__ == "__main__":
    sys.exit(main())
