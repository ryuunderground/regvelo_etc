"""Pull every ChEMBL binding activity for a target, censored records included.

The earlier pull stopped at 6,000 of ~10,380 records because a single HTTP 500
broke the paging loop. Here a failed page is retried with backoff and then
skipped rather than ending the run, and raw records are cached to disk so the
analysis can be re-run without re-pulling.

Censored rows matter: '>' means the compound did not bind up to the highest
concentration tested, so they are the most confident non-binders in the
database, and filtering to standard_relation='=' removes them entirely.

  python fetch_chembl_full.py CHEMBL206 chembl_eralpha_raw.json
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000


def get(url, retries=6):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 500s from this endpoint are transient; back off further each time
            wait = 5 * (i + 1) ** 2
            print(f"    HTTP {e.code}, {wait}s 후 재시도 ({i+1}/{retries})", flush=True)
            time.sleep(wait)
        except Exception as e:                               # noqa: BLE001
            wait = 5 * (i + 1)
            print(f"    {type(e).__name__}, {wait}s 후 재시도 ({i+1}/{retries})", flush=True)
            time.sleep(wait)
    return None


def main(target, out_path):
    first = get(f"{BASE}/activity.json?target_chembl_id={target}&assay_type=B&limit=1")
    if not first:
        sys.exit("첫 요청 실패")
    total = first["page_meta"]["total_count"]
    print(f"{target}: 결합 assay 활성 {total}건\n")

    acts, missed = [], []
    for offset in range(0, total, PAGE):
        url = (f"{BASE}/activity.json?target_chembl_id={target}&assay_type=B"
               f"&limit={PAGE}&offset={offset}")
        d = get(url)
        if d is None:
            missed.append(offset)
            print(f"  {offset}: 건너뜀", flush=True)
            continue
        acts += d["activities"]
        print(f"  {min(offset + PAGE, total)}/{total}  누적 {len(acts)}", flush=True)
        time.sleep(0.4)

    # one more attempt at any page that failed, before giving up on it
    for offset in list(missed):
        d = get(f"{BASE}/activity.json?target_chembl_id={target}&assay_type=B"
                f"&limit={PAGE}&offset={offset}")
        if d:
            acts += d["activities"]
            missed.remove(offset)
            print(f"  재시도 성공 {offset}", flush=True)

    keep = [{k: a.get(k) for k in
             ("molecule_chembl_id", "canonical_smiles", "standard_type",
              "standard_relation", "standard_value", "standard_units",
              "pchembl_value", "assay_chembl_id", "document_chembl_id")}
            for a in acts]
    json.dump(keep, open(out_path, "w"))
    print(f"\n{len(keep)}/{total}건 저장 -> {out_path}")
    if missed:
        print(f"⚠ 실패한 페이지 offset: {missed}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
