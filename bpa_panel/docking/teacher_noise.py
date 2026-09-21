"""How reliable is the docking teacher, per conformer?

The plan on the conformer-contrastive branch feeds a per-conformer docking
score s_i into the pooling layer. That only works if s_i actually tracks the
conformer. 9/10 warned it might not: two BPA shapes 0.0196A apart drew docking
scores 1.419 apart, which is noise, not preference.

Four inputs, all the same molecule:
  confA      a conformer
  confB      a genuinely different conformer (>=1A heavy-atom RMSD)
  confA_rot  confA rigidly rotated and translated -- chemically identical, so
             any score change is pure noise
  confA_jit  confA jittered to ~0.0196A RMSD, matching the 9/10 case

Run in both modes. Default mode wipes the input conformer (processor.py:68,
clearConfs=True) and re-embeds from SMILES with a hardcoded seed 42, so every
variant should score identically -- which would mean s_i carries no conformer
information at all. --use_current_ligand_conf keeps the given shape, which is
the mode the plan needs.

Usage: python teacher_noise.py [ligand.sdf] [receptor]
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IFACE = f"{ROOT}/UniMolDockingV2/unimol_docking_v2/interface"
VENV = f"{ROOT}/.venv/bin"
JITTER_RMSD = 0.0196  # the 9/10 case


def from_lmdb(cid, path=None):
    """Rebuild a multi-conformer mol from the panel LMDB entry for one CID."""
    import pickle

    import lmdb

    path = path or f"{ROOT}/edc_finetune_data/mols_edc_panel.lmdb"
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        entry = pickle.loads(txn.get(cid.encode()))
    env.close()

    # entry["smi"] is canonicalised AFTER embedding, and canonicalising
    # reorders atoms -- the coordinates follow the ORIGINAL PubChem SMILES that
    # build_mols.py parsed, so rebuild from that one.
    import csv

    src = {r["cid"]: r["smiles"]
           for r in csv.DictReader(open(f"{ROOT}/edc_finetune_data/compounds_smiles.csv"))}
    mol = Chem.MolFromSmiles(src[cid])
    stored = [a.GetSymbol() for a in mol.GetAtoms()]
    assert stored == list(entry["atoms"]), "atom order drifted from build_mols.py"
    for coords in entry["coordinates"]:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(np.asarray(coords, dtype=float)):
            conf.SetAtomPosition(i, p.tolist())
        mol.AddConformer(conf, assignId=True)
    return mol


def write(mol, conf_id, path, transform=None):
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    c = Chem.Conformer(mol.GetConformer(conf_id))
    if transform is not None:
        pos = np.array([list(c.GetAtomPosition(i)) for i in range(c.GetNumAtoms())])
        for i, p in enumerate(transform(pos)):
            c.SetAtomPosition(i, p.tolist())
    m.AddConformer(c, assignId=True)
    Chem.SDWriter(path).write(m)
    return m


def variants(sdf, out_dir):
    if sdf.startswith("lmdb:"):
        # The stored conformers already survived build_mols.py's MMFF pass and
        # 1A prune, so they are genuinely distinct -- and they are the exact
        # shapes s_i would be computed over. Re-embedding here instead just
        # collapses everything back to one MMFF minimum.
        mol = from_lmdb(sdf.split(":", 1)[1])
    else:
        mol = next(m for m in Chem.SDMolSupplier(sdf) if m)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=30, randomSeed=0)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)

    far = max(range(1, mol.GetNumConformers()),
              key=lambda i: rdMolAlign.GetBestRMS(Chem.Mol(mol), Chem.Mol(mol), i, 0))

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1  # keep it a rotation, not a reflection

    n = mol.GetConformer(0).GetNumAtoms()
    # per-coordinate sigma that lands the whole-molecule RMSD on JITTER_RMSD
    jit = rng.normal(scale=JITTER_RMSD / np.sqrt(3), size=(n, 3))

    paths = {
        "confA": write(mol, 0, f"{out_dir}/confA.sdf"),
        "confB": write(mol, far, f"{out_dir}/confB.sdf"),
        "confA_rot": write(mol, 0, f"{out_dir}/confA_rot.sdf",
                           lambda p: p @ q.T + np.array([7.0, -3.0, 11.0])),
        "confA_jit": write(mol, 0, f"{out_dir}/confA_jit.sdf", lambda p: p + jit),
    }
    ref = paths["confA"]
    print(f"{mol.GetNumConformers()} conformers available; heavy-atom RMSD vs confA:")
    for name, m in paths.items():
        if name == "confA":
            continue
        # GetBestRMS realigns, so it reports shape difference only
        print(f"  {name:<10} {rdMolAlign.GetBestRMS(Chem.Mol(m), Chem.Mol(ref)):.4f} A")
    return list(paths)


def dock(name, tag, sdf, receptor, out_dir, use_current):
    env = dict(os.environ, PATH=f"{VENV}:{os.environ['PATH']}")
    cmd = [f"{VENV}/python", "demo.py", "--mode", "single",
           "--conf-size", "10", "--cluster",
           "--input-protein", f"{HERE}/inputs/{receptor}.pdb",
           "--input-ligand", sdf,
           "--input-docking-grid", f"{HERE}/inputs/{receptor}_grid.json",
           "--output-ligand-name", f"{tag}_{name}",
           "--output-ligand-dir", out_dir,
           "--steric-clash-fix", "--model-dir", "checkpoint_best.pt"]
    if use_current:
        cmd.append("--use_current_ligand_conf")
    r = subprocess.run(cmd, cwd=IFACE, env=env, capture_output=True, text=True)
    m = re.search(r"predicted RMSD.*?:\s*([0-9.]+)", r.stdout)
    if not m:
        print(r.stdout[-600:], r.stderr[-600:])
        return None
    return float(m.group(1))


def main(sdf, receptor):
    out_dir = os.environ.get("NOISE_DIR", "/tmp/teacher_noise")
    os.makedirs(out_dir, exist_ok=True)
    names = variants(sdf, out_dir)

    print(f"\n{'input':<12}{'default mode':>15}{'use_current_conf':>20}")
    rows = {}
    for name in names:
        a = dock(name, "def", f"{out_dir}/{name}.sdf", receptor, out_dir, False)
        b = dock(name, "cur", f"{out_dir}/{name}.sdf", receptor, out_dir, True)
        rows[name] = (a, b)
        print(f"{name:<12}{a:>15.4f}{b:>20.4f}")

    for col, mode in ((0, "default mode"), (1, "use_current_ligand_conf")):
        v = [rows[n][col] for n in names]
        base = rows["confA"][col]
        print(f"\n{mode}:")
        print(f"  spread over all four inputs   {max(v) - min(v):.4f}")
        print(f"  confA -> confA_rot (must be 0) {rows['confA_rot'][col] - base:+.4f}")
        print(f"  confA -> confA_jit ({JITTER_RMSD}A)  {rows['confA_jit'][col] - base:+.4f}")
        print(f"  confA -> confB (real change)   {rows['confB'][col] - base:+.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/inputs/BPZ.sdf",
         sys.argv[2] if len(sys.argv) > 2 else "ERalpha")
