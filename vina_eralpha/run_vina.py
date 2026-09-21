"""Dock an ERalpha sample with AutoDock Vina and score it against Tox21.

The question is narrow: is the binding signal present in the pocket DrugCLIP
was handed? Vina is zero-shot, so unlike DrugCLIP it has no training set to be
contaminated by and any sample is fair game for its own number. The sample
still includes every DrugCLIP-clean active it can, so the head-to-head subset
is as large as it can be.

  python run_vina.py --n-active 150 --n-inactive 150
  python run_vina.py --limit 6        # timing probe

Results append to results.csv, so an interrupted run resumes.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

EDC = "../edc_finetune_data"
AID = "743078"
RESULTS = "results.csv"   # per-shard: results_<shard>.csv
LIG_DIR = "ligands"
RANK = {"Active": 2, "Inactive": 1, "Inconclusive": 0}


def tox21():
    out = {}
    for row in json.load(open(f"{EDC}/raw/aid_{AID}.json"))["Table"]["Row"]:
        cid, o = row["Cell"][2], row["Cell"][3]
        if cid and (cid not in out or RANK[o] > RANK[out[cid]]):
            out[cid] = o
    return out


def box():
    kv = {}
    for line in open("box.txt"):
        k, v = line.split("=")
        kv[k.strip()] = v.strip()
    return kv


def make_pdbqt(cid, smi):
    path = os.path.join(LIG_DIR, f"{cid}.pdbqt")
    if os.path.exists(path):
        return path
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Tox21 ships salts and hydrates -- "...NC=N4.Cl", "...N.O.Cl.Cl". Vina
    # docks one connected molecule, so a multi-fragment ligand fails in under a
    # second. That took out 66 of 292 on the first pass, and salts are a
    # chemical class rather than a random slice, so dropping them would bias
    # the sample. Keep the largest fragment and neutralise it.
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:                                  # noqa: BLE001
        pass
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=1, useRandomCoords=True) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:                                  # noqa: BLE001
        pass
    sdf = os.path.join(LIG_DIR, f"{cid}.sdf")
    Chem.SDWriter(sdf).write(mol)
    r = subprocess.run(["obabel", sdf, "-O", path], capture_output=True, text=True)
    os.remove(sdf)
    return path if os.path.exists(path) and os.path.getsize(path) else None


def dock(path, b, exhaustiveness):
    cmd = ["vina", "--receptor", "receptor.pdbqt", "--ligand", path,
           "--center_x", b["center_x"], "--center_y", b["center_y"],
           "--center_z", b["center_z"], "--size_x", b["size_x"],
           "--size_y", b["size_y"], "--size_z", b["size_z"],
           "--exhaustiveness", str(exhaustiveness), "--cpu", "1",
           "--seed", "1", "--out", path.replace(".pdbqt", "_out.pdbqt")]
    # A ligand with many rotatable bonds can run for minutes; three of them hit
    # the timeout and the uncaught TimeoutExpired killed the whole shard, taking
    # ~10 untried compounds with it. Record the miss and keep going.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError):
        return None
    best = None
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                best = float(parts[1])
            except ValueError:
                pass
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-active", type=int, default=150)
    ap.add_argument("--n-inactive", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="timing probe: dock only N")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--cid-file", default="", help="dock exactly these CIDs (one per line)")
    ap.add_argument("--smiles-csv", default="",
                    help="dock rows of a csv with id/label/smiles columns (ChEMBL etc)")
    ap.add_argument("--out-prefix", default="results", help="results file prefix")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(LIG_DIR, exist_ok=True)
    T = tox21()
    smiles = {r["cid"]: r["smiles"] for r in csv.DictReader(open(f"{EDC}/library_smiles.csv"))}
    clean = set(json.load(open("clean_actives.json"))) if os.path.exists("clean_actives.json") else set()

    if args.smiles_csv:
        # ChEMBL ids are not in library_smiles.csv, so carry the SMILES through
        rows = list(csv.DictReader(open(args.smiles_csv)))
        key = "chembl_id" if "chembl_id" in rows[0] else "id"
        smiles.update({r[key]: r["smiles"] for r in rows})
        todo = [(r[key], r["label"]) for r in rows]
        random.seed(1)
        random.shuffle(todo)
        if args.nshards > 1:
            todo = todo[args.shard::args.nshards]
        return dock_list(todo, args, smiles)

    if args.cid_file:
        wanted = [c.strip() for c in open(args.cid_file) if c.strip()]
        todo = [(c, T.get(c, "")) for c in wanted if c in smiles]
        random.seed(1)
        random.shuffle(todo)
        if args.nshards > 1:
            todo = todo[args.shard::args.nshards]
        return dock_list(todo, args)

    acts = [c for c, o in T.items() if o == "Active" and c in smiles]
    inacts = [c for c, o in T.items() if o == "Inactive" and c in smiles]
    random.seed(1)
    # every DrugCLIP-clean active first, then fill at random
    acts = sorted(acts, key=lambda c: (c not in clean, random.random()))[:args.n_active]
    random.shuffle(inacts)
    inacts = inacts[:args.n_inactive]
    todo = [(c, "Active") for c in acts] + [(c, "Inactive") for c in inacts]
    random.shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]
    # shard AFTER the shuffle so every worker gets a mix of both classes
    if args.nshards > 1:
        todo = todo[args.shard::args.nshards]

    return dock_list(todo, args)


def dock_list(todo, args, smiles=None):
    """Dock a (id, outcome) list, resuming from whatever is already scored."""
    if smiles is None:
        smiles = {r["cid"]: r["smiles"]
                  for r in csv.DictReader(open(f"{EDC}/library_smiles.csv"))}
    out_path = (f"{args.out_prefix}.csv" if args.nshards == 1
                else f"{args.out_prefix}_{args.shard}.csv")
    # a CID scored under any prefix or shard is done -- shard assignment shifts
    # when the compound list changes, so checking only this file re-docks them
    done = set()
    for f_ in glob.glob(f"{args.out_prefix}_*.csv"):
        for r in csv.DictReader(open(f_)):
            # tolerate any same-prefixed file that is not a results file: the
            # input chembl_sample.csv got caught by this glob and the KeyError
            # killed every shard before it docked anything
            if r.get("vina_affinity"):
                done.add(r["cid"])
    todo = [t for t in todo if t[0] not in done]
    print(f"shard {args.shard}: {len(todo)} to dock ({len(done)} already scored)", flush=True)

    b = box()
    new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["cid", "tox21_outcome", "vina_affinity", "seconds"])
    t0 = time.time()
    for i, (cid, outcome) in enumerate(todo, 1):
        start = time.time()
        p = make_pdbqt(cid, smiles[cid])
        score = dock(p, b, args.exhaustiveness) if p else None
        w.writerow([cid, outcome, "" if score is None else f"{score:.3f}",
                    f"{time.time() - start:.1f}"])
        f.flush()
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/i:.1f}s/ligand  "
                  f"eta {(len(todo)-i)*el/i/60:.0f} min", flush=True)
    f.close()


if __name__ == "__main__":
    sys.exit(main())
