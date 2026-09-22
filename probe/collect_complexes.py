"""Collect nuclear-receptor co-crystals as (pocket, ligand, bound pose) triples.

This is the dataset for the atom-pocket distance probe. BindCLIP uses that probe
(its Table 4) to ask whether the embeddings carry interaction geometry at all,
reporting RMSE 1.05 for DrugCLIP against 0.87 for itself. Running it on our own
checkpoint needs no training and no PDBbind -- only structures, which RCSB
serves directly, so it sidesteps the pdbbind.org.cn outage.

Per structure: the pocket is every residue within 6A of the ligand (the same
definition build_pockets.py uses), the ligand is its crystallographic pose, and
the regression target is each ligand atom's minimum distance to any pocket atom.

Junk components are skipped -- carving a site around a cryoprotectant would make
the probe measure nothing.
"""
import json
import os
import pickle
import sys
import time
import urllib.parse
import urllib.request

import lmdb
import numpy as np
from biopandas.pdb import PandasPdb

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_URL = "https://files.rcsb.org/download/{}.pdb"
CACHE = "pdb_cache"
OUT = "complexes.lmdb"
RADIUS = 6.0

# the receptors the panel covers, by UniProt
ACCESSIONS = {
    "ESR1": "P03372", "ESR2": "Q92731", "ESRRG": "P62508", "AR": "P10275",
    "PGR": "P06401", "PPARG": "P37231", "THRB": "P10828", "NR3C1": "P04150",
    "NR3C2": "P08235", "VDR": "P11473", "NR1I2": "O75469", "PPARA": "Q07869",
    "PPARD": "Q03181", "THRA": "P10827", "RXRA": "P19793",
}

JUNK = {
    "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G",
    "ACT", "ACY", "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "NI", "CD",
    "TRS", "MES", "EPE", "BTB", "CIT", "FLC", "TAR", "MPD", "DMS", "IOD", "BR",
    "IMD", "FMT", "NO3", "AZI", "BME", "DTT", "EOH", "IPA", "URE", "SCN", "NH4",
    "CPS", "B7G", "JZR", "LDA", "C8E", "BOG", "SDS", "OLC", "PLM", "MYR",
}


def get(url, retries=3, raw=False):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode() if raw else json.load(r)
        except Exception:                                  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return None


def entry_ids(acc, limit):
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


def fetch_pdb(pdb_id):
    path = os.path.join(CACHE, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    txt = get(PDB_URL.format(pdb_id), raw=True)
    if not txt or txt.count("\nATOM") < 100:
        return None
    open(path, "w").write(txt)
    return path


def build(pdb_path, receptor):
    """Largest non-junk het group, its pocket, and per-atom distances."""
    ppdb = PandasPdb().read_pdb(pdb_path)
    het = ppdb.df["HETATM"]
    het = het[(~het["residue_name"].isin(JUNK)) & (het["element_symbol"] != "H")]
    if het.empty:
        return None
    # one instance, the one with the most heavy atoms
    groups = list(het.groupby(["chain_id", "residue_number", "residue_name"]))
    (chain, resnum, resname), lig = max(groups, key=lambda kv: len(kv[1]))
    if not 8 <= len(lig) <= 80:                 # skip ions and huge cofactors
        return None

    prot = ppdb.df["ATOM"]
    prot = prot[(prot["chain_id"] == chain) & (prot["element_symbol"] != "H")]
    if prot.empty:
        return None
    pc = prot[["x_coord", "y_coord", "z_coord"]].to_numpy()
    lc = lig[["x_coord", "y_coord", "z_coord"]].to_numpy()
    dist = np.linalg.norm(pc[:, None, :] - lc[None, :, :], axis=-1)

    key = prot["chain_id"].astype(str) + prot["residue_number"].astype(str)
    close = set(key[(dist < RADIUS).any(axis=1)])
    mask = key.isin(close).to_numpy()
    if mask.sum() < 30:
        return None
    return {
        "receptor": receptor,
        "ligand_code": resname,
        "pocket_atoms": prot.loc[mask, "atom_name"].tolist(),
        "pocket_coordinates": pc[mask],
        "ligand_atoms": lig["element_symbol"].tolist(),
        "ligand_coordinates": lc,
        # the probe target: how close each ligand atom sits to the pocket
        "atom_min_dist": dist[mask].min(axis=0),
    }


def main(per_receptor=40):
    os.makedirs(CACHE, exist_ok=True)
    env = lmdb.open(OUT, subdir=False, map_size=4 * 1024 ** 3)
    n = 0
    with env.begin(write=True) as txn:
        for receptor, acc in ACCESSIONS.items():
            ids = entry_ids(acc, per_receptor)
            kept = 0
            for pdb_id in ids:
                p = fetch_pdb(pdb_id)
                if not p:
                    continue
                try:
                    rec = build(p, receptor)
                except Exception:                          # noqa: BLE001
                    rec = None
                if rec is None:
                    continue
                rec["pdb_id"] = pdb_id
                txn.put(f"{receptor}_{pdb_id}".encode(), pickle.dumps(rec))
                kept += 1
                n += 1
                time.sleep(0.1)
            print(f"  {receptor:<7} {kept:>3}/{len(ids)} 사용", flush=True)
    env.close()
    print(f"\n{n}개 복합체 -> {OUT}")


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
