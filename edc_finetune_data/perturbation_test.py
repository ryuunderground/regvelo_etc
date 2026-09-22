"""Does the DrugCLIP score notice when BPA's binding chemistry is removed?

BindCLIP's Fig 1a perturbs actives with interaction-disrupting edits and checks
whether the score drops; DrugCLIP's mean drop was 0.04 against BindCLIP's 0.18.
BPA is the ideal case for this project: its two para phenol hydroxyls are what
mimic estradiol's, and they make the ER hydrogen bonds. Masking them should cost
a structure-aware model a lot.

The control matters as much as the perturbation. Adding a methyl to the rings
changes size and lipophilicity by about as much as methylating a hydroxyl, but
leaves both hydrogen bonds intact. If the score tracks the control and not the
masking, it is reading physicochemistry rather than interactions -- which is the
shortcut BindCLIP names, and which logP beating the model already suggested.
"""
import csv
import pickle
import subprocess
import sys

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

# name, smiles, what it does to the ER hydrogen bonds
PANEL = [
    ("BPA",            "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",       "원본"),
    ("BPA_1OMe",       "CC(C)(c1ccc(OC)cc1)c1ccc(O)cc1",      "수소결합 1개 차단"),
    ("BPA_2OMe",       "CC(C)(c1ccc(OC)cc1)c1ccc(OC)cc1",     "수소결합 2개 차단"),
    ("BPA_deOH",       "CC(C)(c1ccccc1)c1ccccc1",             "수산기 제거"),
    ("BPA_3Me",        "CC(C)(c1ccc(O)c(C)c1)c1ccc(O)c(C)c1", "대조: 고리에 메틸, 수소결합 유지"),
    ("BPA_3Me2",       "CC(C)(c1ccc(O)c(C)c1)c1ccc(O)cc1",    "대조: 메틸 1개, 수소결합 유지"),
]
LMDB = "mols_perturb.lmdb"
OUT = "perturb_eval_v2"


def build():
    env = lmdb.open(LMDB, subdir=False, map_size=50 * 1024 * 1024)
    rows = []
    with env.begin(write=True) as txn:
        for name, smi, note in PANEL:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                sys.exit(f"bad smiles for {name}")
            mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
            m = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(m, numConfs=5, pruneRmsThresh=1,
                                       maxAttempts=50, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMoleculeConfs(m)
            except Exception:                              # noqa: BLE001
                pass
            m = Chem.RemoveHs(m)
            txn.put(name.encode(), pickle.dumps({
                "atoms": [a.GetSymbol() for a in m.GetAtoms()],
                "coordinates": [c.GetPositions() for c in m.GetConformers()],
                "smi": Chem.MolToSmiles(m), "cid": name}))
            rows.append((name, note, mw, logp))
    env.close()
    return rows


def score():
    subprocess.run([
        "../.venv/bin/python", "../DrugCLIP/unimol/score_matrix.py", "train_data",
        "--user-dir", "../DrugCLIP/unimol", "--valid-subset", "valid",
        "--task", "drugclip", "--loss", "in_batch_softmax", "--arch", "drugclip",
        "--path", "save_dir_v2/checkpoint_best.pt",
        "--results-path", OUT, "--emb-dir", f"{OUT}/emb",
        "--mol-path", LMDB, "--pocket-path", "../bpa_panel/pocket_bpa_panel.lmdb",
        "--max-pocket-atoms", "450", "--batch-size", "8", "--num-workers", "0",
        "--cpu", "--seed", "1", "--log-interval", "100", "--log-format", "simple",
    ], capture_output=True, text=True, check=True)


def main():
    meta = {n: (note, mw, lp) for n, note, mw, lp in build()}
    score()
    env = lmdb.open(LMDB, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        smi2name = {pickle.loads(v)["smi"]: k.decode() for k, v in txn.cursor()}
    env.close()
    rows = list(csv.reader(open(f"{OUT}/score_matrix.csv")))
    names = [smi2name.get(s) for s in rows[0][1:]]
    S = {r[0]: {n: float(v) for n, v in zip(names, r[1:]) if n} for r in rows[1:]}

    for rec in ("ERalpha", "ERRgamma"):
        base = S[rec]["BPA"]
        print(f"\n=== {rec} ===")
        print(f"{'화합물':<12}{'점수':>8}{'Δ vs BPA':>11}{'MW':>8}{'logP':>7}   설명")
        for name, _, _ in PANEL:
            s = S[rec][name]
            _, mw, lp = meta[name]
            d = "" if name == "BPA" else f"{s - base:+.4f}"
            print(f"{name:<12}{s:>8.4f}{d:>11}{mw:>8.1f}{lp:>7.2f}   {meta[name][0]}")


if __name__ == "__main__":
    main()
