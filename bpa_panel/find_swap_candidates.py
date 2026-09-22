"""Find a usable (PDB entry, reference ligand) pair for each new receptor.

build_pockets.py needs both, and guessing PDB ids from memory is how you end up
carving a pocket around a cryoprotectant. This asks RCSB for ligand-bound
structures of each UniProt accession, takes the best-resolution ones, and reads
off their non-polymer components, filtering the usual buffer and cryo junk so
what is left is the actual bound ligand.

Prints candidates for review; it does not build anything.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA = "https://data.rcsb.org/rest/v1/core"

# receptor, UniProt accession -- the 8 we want to add
TARGETS = [
    ("ERalpha", "P03372"), ("ERRgamma", "P62508"),
    ("ERRbeta", "O95718"), ("PPARgamma", "P37231"),
]

# crystallisation additives, cryoprotectants, ions -- never the ligand of interest
JUNK = {
    "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G",
    "ACT", "ACY", "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "NI", "CD",
    "TRS", "MES", "EPE", "BTB", "CIT", "FLC", "TAR", "MPD", "DMS", "IOD", "BR",
    "IMD", "FMT", "NO3", "AZI", "BME", "DTT", "EOH", "IPA", "URE", "SCN", "NH4",
    "2OH",   # BPA itself -- excluded, the whole point is a pocket not carved by it
}


def get(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:                                  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return None


def entries(acc, limit=25):
    """Ligand-bound entries for this accession, best resolution first."""
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers."
                         "reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": acc}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.nonpolymer_entity_count",
            "operator": "greater", "value": 0}},
    ]}, "return_type": "entry", "request_options": {
        "paginate": {"start": 0, "rows": limit},
        "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                  "direction": "asc"}]}}
    d = get(f"{SEARCH}?json={urllib.parse.quote(json.dumps(q))}")
    return [x["identifier"] for x in d["result_set"]] if d else []


def ligands(pdb_id):
    """Non-polymer components with their formula weights, junk removed."""
    d = get(f"{DATA}/entry/{pdb_id}")
    if not d:
        return None, []
    res = (d.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0]
    ids = d.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
    out = []
    for eid in ids:
        e = get(f"{DATA}/nonpolymer_entity/{pdb_id}/{eid}")
        if not e:
            continue
        comp = e.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if not comp or comp in JUNK:
            continue
        c = get(f"{DATA}/chemcomp/{comp}")
        mw = (c or {}).get("chem_comp", {}).get("formula_weight")
        name = (c or {}).get("chem_comp", {}).get("name", "")
        if mw and float(mw) >= 200:
            out.append((comp, float(mw), name[:40]))
        time.sleep(0.15)
    return res, out


def main():
    for name, acc in TARGETS:
        print(f"\n=== {name} ({acc}) ===", flush=True)
        found = 0
        for pdb_id in entries(acc):
            res, ligs = ligands(pdb_id)
            if not ligs:
                continue
            ligs.sort(key=lambda x: -x[1])
            r = f"{res:.2f}A" if res else "?"
            print(f"  {pdb_id}  {r:>7}  " +
                  " | ".join(f"{c} {mw:.0f} {n}" for c, mw, n in ligs[:2]), flush=True)
            found += 1
            if found >= 3:
                break
        if not found:
            print("  후보 없음")


if __name__ == "__main__":
    sys.exit(main())
