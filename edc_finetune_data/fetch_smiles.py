"""Fetch canonical SMILES for all CIDs in pairs.csv, in batches."""
import csv
import json
import time
import urllib.request

BATCH = 100


def main():
    cids = sorted(set(row["cid"] for row in csv.DictReader(open("pairs.csv"))), key=int)
    print(len(cids), "unique CIDs")

    smiles = {}
    failed_batches = []
    for i in range(0, len(cids), BATCH):
        batch = cids[i : i + BATCH]
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + ",".join(batch)
            + "/property/IsomericSMILES,MolecularWeight/JSON"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            for prop in data["PropertyTable"]["Properties"]:
                smiles[str(prop["CID"])] = (prop["SMILES"], prop.get("MolecularWeight"))
        except Exception as e:
            print(f"batch {i}: FAILED ({e})")
            failed_batches.append(batch)
        print(f"batch {i}-{i+len(batch)}: {len(smiles)} total so far")
        time.sleep(0.3)

    with open("compounds_smiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", "smiles", "molecular_weight"])
        for cid in cids:
            if cid in smiles:
                w.writerow([cid, smiles[cid][0], smiles[cid][1]])

    print(f"\nwrote {len(smiles)}/{len(cids)} SMILES to compounds_smiles.csv")
    if failed_batches:
        print(f"{len(failed_batches)} batches failed, missing CIDs:", sum(failed_batches, []))


if __name__ == "__main__":
    main()
