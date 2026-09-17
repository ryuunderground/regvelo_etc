"""Extract the real crystallographic BPA pose (with proper bonds, via
template matching) for ERalpha/ERRgamma/PPARgamma, so we can compute a
true RMSD against the model's predicted pose -- not just its self-reported
confidence."""
import pickle

from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

BPA_SMILES = "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"

TARGETS = {
    "ERalpha": ("../pdb/3UU7.pdb", "2OH"),
    "ERRgamma": ("../pdb/2E2R.pdb", "2OH"),
    "PPARgamma": ("../pdb/9F7W.pdb", "2OH"),
}


def extract_true_bpa(pdb_path, lig_code):
    ppdb = PandasPdb().read_pdb(pdb_path)
    lig = ppdb.df["HETATM"]
    lig = lig[lig["residue_name"] == lig_code]
    lig = lig[lig["element_symbol"] != "H"]  # some entries model explicit ligand H
    first_instance = lig[["chain_id", "residue_number"]].iloc[0]
    lig = lig[
        (lig["chain_id"] == first_instance["chain_id"])
        & (lig["residue_number"] == first_instance["residue_number"])
    ]

    # build a minimal PDB block for just these HETATM lines so RDKit can read positions
    lig_renamed = lig.copy()
    lig_renamed["record_name"] = "HETATM"
    block_lines = []
    for _, row in lig_renamed.iterrows():
        block_lines.append(
            "HETATM{:>5} {:<4} {:<3} {:1}{:>4}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2}".format(
                int(row["atom_number"]), row["atom_name"][:4], "LIG", "A", 1,
                row["x_coord"], row["y_coord"], row["z_coord"], 1.0, 0.0,
                row["element_symbol"],
            )
        )
    block = "\n".join(block_lines) + "\nEND\n"

    raw_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    template = Chem.MolFromSmiles(BPA_SMILES)
    true_mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    return true_mol


def main():
    for name, (pdb_path, lig_code) in TARGETS.items():
        mol = extract_true_bpa(pdb_path, lig_code)
        Chem.MolToMolFile(mol, f"verify_bpa/{name}_true_BPA.sdf")
        print(name, "-> true BPA pose extracted,", mol.GetNumAtoms(), "heavy atoms")


if __name__ == "__main__":
    main()
