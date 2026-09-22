"""How much does BPA's own crystal pose move between structures of one receptor?

ERRgamma has three BPA-bound entries (2E2R, 2P7G, 6I63). Superimposing the
proteins and comparing the ligands gives the experimental reproducibility of
the pose itself -- and no docking method can be expected to beat it. Step 7
reported 0.55A for a Uni-Mol prediction against 2E2R; that number only means
something next to this floor.

Symmetry-corrected throughout: BPA's two para-hydroxyphenyl rings are
equivalent, so atom-order RMSD calls an identical pose wrong (step 7 measured
3.71A naive against 0.55A corrected).
"""
import itertools

import numpy as np
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

BPA_SMILES = "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"
LIG = "2OH"
PDB_DIR = "../../pdb"
# same receptor, three independent structures
ERRGAMMA = ["2E2R", "2P7G", "6I63"]


def load(pdb_id):
    ppdb = PandasPdb().read_pdb(f"{PDB_DIR}/{pdb_id}.pdb")
    lig = ppdb.df["HETATM"]
    lig = lig[(lig["residue_name"] == LIG) & (lig["element_symbol"] != "H")]
    if lig.empty:
        return None
    # one copy only -- pooling copies is the bug from README step 8
    first = lig[["chain_id", "residue_number"]].iloc[0]
    lig = lig[(lig["chain_id"] == first["chain_id"])
              & (lig["residue_number"] == first["residue_number"])]
    prot = ppdb.df["ATOM"]
    prot = prot[prot["chain_id"] == first["chain_id"]]
    ca = prot[prot["atom_name"] == "CA"]
    sub = PandasPdb()
    sub._df = {"ATOM": lig.iloc[0:0], "HETATM": lig, "ANISOU": lig.iloc[0:0],
               "OTHERS": lig.iloc[0:0]}
    return {
        "block": sub.to_pdb_stream(records=["HETATM"]).getvalue(),
        "lig": lig[["x_coord", "y_coord", "z_coord"]].to_numpy(),
        "ca": ca[["x_coord", "y_coord", "z_coord"]].to_numpy(),
        "resnum": ca["residue_number"].to_numpy(),
    }


def kabsch(P, Q):
    """Rotation+translation putting P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def mol_from_pdb_block(block):
    """Bond orders from the SMILES template, atom order from the PDB.

    Overwriting a SMILES-built conformer by index instead silently pairs each
    PDB atom with a different RDKit atom, so the bond graph stops matching the
    coordinates and the symmetry correction compares nonsense -- that is what
    produced a spurious 4.2A here.
    """
    raw = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
    if raw is None:
        return None
    template = Chem.MolFromSmiles(BPA_SMILES)
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, raw)
    except Exception:                                     # noqa: BLE001
        return None


def main():
    data = {p: load(p) for p in ERRGAMMA}
    for p, d in data.items():
        print(f"{p}: 리간드 {len(d['lig'])}원자, Cα {len(d['ca'])}개")

    print(f"\nERRγ 구조 간 BPA 포즈 차이 (단백질 정렬 후, 대칭 보정)")
    print(f"{'비교':<16}{'Cα 정렬 RMSD':>14}{'BPA 포즈 RMSD':>15}")
    for a, b in itertools.combinations(ERRGAMMA, 2):
        A, B = data[a], data[b]
        # match residues by number so the Ca sets correspond
        common = np.intersect1d(A["resnum"], B["resnum"])
        ia = np.array([np.where(A["resnum"] == r)[0][0] for r in common])
        ib = np.array([np.where(B["resnum"] == r)[0][0] for r in common])
        R, t = kabsch(A["ca"][ia], B["ca"][ib])
        ca_rmsd = np.sqrt((((A["ca"][ia] @ R.T + t) - B["ca"][ib]) ** 2).sum(1).mean())
        ma, mb = mol_from_pdb_block(A["block"]), mol_from_pdb_block(B["block"])
        if ma is None or mb is None:
            print(f"{a}-{b:<11}{ca_rmsd:>14.2f}   분자 해석 실패")
            continue
        conf = ma.GetConformer()
        for i in range(ma.GetNumAtoms()):
            x, y, z = A["lig"][i]
            nx, ny, nz = R @ np.array([x, y, z]) + t
            conf.SetAtomPosition(i, (float(nx), float(ny), float(nz)))
        rms = rdMolAlign.CalcRMS(ma, mb)          # symmetry-aware, no realignment
        print(f"{a}-{b:<11}{ca_rmsd:>14.2f}{rms:>15.2f}")

    print("\n이 값이 실험 재현성의 바닥입니다. 모델 포즈 정확도는 이보다 좋을 수 없습니다.")


if __name__ == "__main__":
    main()
