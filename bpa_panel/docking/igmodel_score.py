"""Re-score Uni-Mol Docking V2 poses with IGModel (pred_rmsd + pred_pkd).

Uni-Mol writes SDFs that stop at `M  END` with no `$$$$` terminator, and
IGModel's sdf_split() slices on exactly that string -- so an unpatched pose
file parses as zero poses and dies in a bare `except ValueError`. Rewriting
the pose through RDKit's SDWriter fixes it without touching IGModel.

Two jobs:
  verify  the 3 BPA poses whose true symmetry-aware RMSD against the crystal
          structure is known, so pred_rmsd can be checked against an answer
  panel   the 36 receptor x ligand pairs, for spread

Setup (IGModel/ is gitignored -- 144MB of bundled weights):

    git clone https://github.com/zchwang/IGModel.git
    uv venv --python 3.11 IGModel/.venv
    uv pip install --python IGModel/.venv/bin/python \
        dgl "torch==2.2.0" "torchdata==0.7.1" "numpy<2" packaging \
        openbabel-wheel spyrmsd torch-geometric pyyaml pydantic rdkit \
        pandas scipy

torch is pinned to 2.2.0 because dgl 2.2.0 ships prebuilt graphbolt dylibs
only for torch 2.1.0-2.3.0, and torchdata to 0.7.1 because dgl still imports
torchdata.datapipes, dropped in 0.10. The bundled run_scoring.sh passes a
-prefix flag scoring.py no longer accepts.

Usage: python igmodel_score.py verify|panel
"""
import csv
import os
import subprocess
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IG = f"{ROOT}/IGModel"
TMP = os.environ.get("IG_TMP", "/tmp/igmodel")

# receptor -> its crystal reference ligand, for pocket definition
VERIFY = ["ERalpha", "ERRgamma", "PPARgamma"]


def clean_receptor(src, dst, ref_xyz):
    """One chain, no alternate locations.

    IGModel keys pocket atoms by residue and reads pdb_to_xyz_dict["CA"], so a
    structure carrying altlocs or several copies of the chain collides residues
    across copies and loses the alpha carbon (KeyError: 'CA'). Our panel PDBs
    are full asymmetric units -- ERalpha alone has chains A/B/F/G and 38 altloc
    records -- which is the same duplicate-copy trap that hit build_pockets.py.

    Keeps the chain sitting closest to the reference coordinates.
    """
    atoms = [l for l in open(src) if l.startswith("ATOM")
             and l[16] in (" ", "A")]

    def dist2(line):
        x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        return min((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 for a, b, c in ref_xyz)

    by_chain = {}
    for l in atoms:
        by_chain.setdefault(l[21], []).append(l)
    chain = min(by_chain, key=lambda c: min(dist2(l) for l in by_chain[c]))

    with open(dst, "w") as f:
        # blank the altloc column so nothing downstream re-splits on it
        f.writelines(l[:16] + " " + l[17:] for l in by_chain[chain])
        f.write("END\n")
    return dst, chain, len(by_chain[chain])


def pose_xyz(sdf):
    mol = next((m for m in Chem.SDMolSupplier(sdf, sanitize=False) if m), None)
    return mol.GetConformer().GetPositions().tolist()


def normalise(src, dst, name):
    """Rewrite a pose SDF so it carries a name and a $$$$ terminator."""
    mol = next((m for m in Chem.SDMolSupplier(src, sanitize=False) if m), None)
    if mol is None:
        return None
    mol.SetProp("_Name", name)
    with Chem.SDWriter(dst) as w:
        w.write(mol)
    return dst


def score(rec_pdb, ref_lig, pose_sdf, out_csv, model="saved_model.pth"):
    cmd = [f"{IG}/.venv/bin/python", f"{IG}/scripts/scoring.py",
           "-rec_fpath", rec_pdb, "-pose_fpath", pose_sdf,
           "-model", f"{IG}/models/{model}", "-out_fpath", out_csv]
    if ref_lig:
        cmd += ["-ref_lig_fpath", ref_lig]
    r = subprocess.run(cmd, cwd=f"{IG}/scripts", capture_output=True, text=True)
    if not os.path.exists(out_csv):
        return None, r.stdout[-400:] + r.stderr[-400:]
    rows = list(csv.DictReader(open(out_csv)))
    return rows, None


def verify():
    truth = {r["receptor"]: r for r in
             csv.DictReader(open(f"{HERE}/verify_bpa/rmsd_vs_crystal.csv"))}
    print(f"{'receptor':<11}{'true RMSD':>11}{'IG pred_rmsd':>14}"
          f"{'IG pred_pKd':>13}{'UniMol prmsd':>14}")
    for rec in VERIFY:
        ref_lig = f"{HERE}/verify_bpa/{rec}_true_BPA.sdf"
        pose = normalise(f"{HERE}/verify_bpa/predict_sdf/{rec}__BPA_verify.sdf",
                         f"{TMP}/{rec}_pose.sdf", f"{rec}_BPA")
        pdb, chain, n = clean_receptor(f"{HERE}/inputs/{rec}.pdb",
                                       f"{TMP}/{rec}_clean.pdb", pose_xyz(ref_lig))
        rows, err = score(pdb, ref_lig, pose, f"{TMP}/ig_{rec}.csv")
        t = truth[rec]
        if not rows:
            print(f"{rec:<11}{'FAILED':>11}  {err}")
            continue
        print(f"{rec:<11}{float(t['symmetry_aware_best_rmsd_angstrom']):>11.3f}"
              f"{float(rows[0]['pred_rmsd']):>14.3f}"
              f"{float(rows[0]['pred_pkd']):>13.3f}"
              f"{float(t['model_self_reported_confidence']):>14.2f}")


def panel():
    """36 pairs. No per-receptor reference ligand exists, so the self-ref
    weights define the pocket from the docking pose itself."""
    out = []
    for r in csv.DictReader(open(f"{HERE}/predict_sdf/predicted_rmsd.csv")):
        name = r["output_ligand_name"]
        rec, lig = name.split("__")
        pose = normalise(r["output_ligand_sdf"], f"{TMP}/{name}.sdf", name)
        if pose is None:
            continue
        # self-ref defines the pocket from the pose, so clean around the pose too
        pdb, _, _ = clean_receptor(f"{HERE}/inputs/{rec}.pdb",
                                   f"{TMP}/{rec}_clean_{lig}.pdb", pose_xyz(pose))
        rows, err = score(pdb, None, pose,
                          f"{TMP}/ig_{name}.csv", model="self-ref_model.pth")
        if not rows:
            print(f"  {name}: FAILED {err[:120]}")
            continue
        out.append({"receptor": rec, "ligand": lig,
                    "ig_pred_rmsd": rows[0]["pred_rmsd"],
                    "ig_pred_pkd": rows[0]["pred_pkd"],
                    "unimol_prmsd": r["predicted_rmsd"]})
        print(f"  {name:<24} pred_rmsd {float(rows[0]['pred_rmsd']):6.3f}  "
              f"pKd {float(rows[0]['pred_pkd']):6.3f}")
    dst = f"{HERE}/igmodel_panel.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out)} rows to {dst}")


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    {"verify": verify, "panel": panel}[sys.argv[1] if len(sys.argv) > 1 else "verify"]()
