"""Pull PubChem BioAssay concise tables into raw/, same shape as the existing files.

The five receptors already here came from hand-run pulls with no script behind
them. This one takes AIDs on the command line so the next receptor is one
command, not a repeat of that.

  python fetch_assays.py 1259394 1259396 743241 743242 1346982

Existing files are a mix of Summary and Confirmatory assays (PPARgamma 743191
and THRbeta 743066 are Confirmatory), so there is no convention to keep beyond
preferring a Summary when the receptor has one -- it is the curated call across
the primary screen and its counterscreens.
"""
import json
import os
import sys
import time
import urllib.request
from collections import Counter

RAW_DIR = "raw"
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{}/concise/JSON"


def fetch(aid, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(URL.format(aid), timeout=180) as r:
                return json.load(r)
        except Exception as e:                       # noqa: BLE001 -- report and retry
            if attempt == retries - 1:
                print(f"  {aid}: giving up ({e})")
                return None
            time.sleep(3 * (attempt + 1))
    return None


def main(aids):
    os.makedirs(RAW_DIR, exist_ok=True)
    for aid in aids:
        path = os.path.join(RAW_DIR, f"aid_{aid}.json")
        if os.path.exists(path):
            print(f"{aid}: already in {RAW_DIR}, skipping")
            continue
        print(f"{aid}: fetching...", flush=True)
        d = fetch(aid)
        if d is None:
            continue
        rows = d["Table"]["Row"]
        with open(path, "w") as f:
            json.dump(d, f)
        cells = rows[0]["Cell"]
        outcomes = Counter(r["Cell"][3] for r in rows)
        cids = {r["Cell"][2] for r in rows if r["Cell"][2]}
        print(f"  saved {len(rows)} rows, {len(cids)} CIDs  [{cells[9]}]")
        print(f"  {cells[8][:88]}")
        print(f"  {dict(outcomes)}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
