"""Prepare the ERalpha receptor and docking box for Vina, from the same 3UU7
entry and the same reference ligand that build_pockets.py uses.

Keeping the pocket definition identical matters: the whole point is to ask
whether a different method finds signal in the pocket DrugCLIP was given, so
the box has to sit where DrugCLIP's pocket sits.

3UU7 carries more than one copy of the reference ligand, and pooling them is
the bug that produced the 700-atom PPARgamma pocket (README step 8). Same fix
here: keep the first (chain, resnum) instance and build the box around it.
"""
import os
import subprocess
import sys

import numpy as np
from biopandas.pdb import PandasPdb

PDB = "../bpa_panel/pdb/3UU7.pdb"
LIGAND_CODE = "2OH"          # BPA, per bpa_panel/receptors.csv
PAD = 8.0                    # Angstrom added around the ligand extent
OUT = "."


def main():
    ppdb = PandasPdb().read_pdb(PDB)
    het = ppdb.df["HETATM"]
    lig = het[het["residue_name"] == LIGAND_CODE]
    lig = lig[lig["element_symbol"] != "H"]
    if lig.empty:
        sys.exit(f"{LIGAND_CODE} not found in {PDB}")

    first = lig[["chain_id", "residue_number"]].iloc[0]
    lig = lig[(lig["chain_id"] == first["chain_id"])
              & (lig["residue_number"] == first["residue_number"])]
    coords = lig[["x_coord", "y_coord", "z_coord"]].to_numpy()
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = extent + 2 * PAD

    print(f"reference ligand {LIGAND_CODE}: chain {first['chain_id']} "
          f"res {first['residue_number']}, {len(coords)} heavy atoms")
    print(f"box center  {center[0]:8.3f} {center[1]:8.3f} {center[2]:8.3f}")
    print(f"box size    {size[0]:8.3f} {size[1]:8.3f} {size[2]:8.3f}")

    # protein only, and only the chain the reference ligand sits in -- the other
    # copies are separate binding sites and would just add noise to the grid
    prot = ppdb.df["ATOM"]
    prot = prot[prot["chain_id"] == first["chain_id"]]
    ppdb.df["ATOM"] = prot          # records=["ATOM"] means the rest is ignored
    rec_pdb = os.path.join(OUT, "receptor.pdb")
    ppdb.to_pdb(rec_pdb, records=["ATOM"])
    print(f"\nwrote {rec_pdb}: chain {first['chain_id']}, {len(prot)} atoms")

    rec_pdbqt = os.path.join(OUT, "receptor.pdbqt")
    cmd = ["obabel", rec_pdb, "-O", rec_pdbqt, "-xr", "-p", "7.4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(" ".join(cmd))
    print((r.stderr or r.stdout).strip()[:300])

    with open(os.path.join(OUT, "box.txt"), "w") as f:
        f.write(f"center_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\n"
                f"center_z = {center[2]:.3f}\n"
                f"size_x = {size[0]:.3f}\nsize_y = {size[1]:.3f}\nsize_z = {size[2]:.3f}\n")
    print("wrote box.txt")


if __name__ == "__main__":
    main()
