"""Rebuild the BPA-carved pockets from structures solved with a different ligand.

Four of the sixteen pockets are carved as a 6A shell around BPA itself, and
those four take ranks 1, 2, 3 and 9 when BPA is scored -- mean rank 3.8 against
10.1 for the rest. Two readings fit: the pocket is a BPA-shaped hole, or BPA
genuinely binds those receptors and that is why co-crystals exist.

Swapping in a structure of the same receptor solved with an unrelated ligand
separates them. If BPA keeps its rank, the signal is real; if it drops, the
carving was doing the work. ERRbeta cannot be swapped -- 6LIT is its only
ligand-bound entry and BPA is that ligand.
"""
import pickle

import lmdb

from build_pockets import extract_pocket
from build_pockets_nr16 import NEW

ORIGINAL_7 = [
    ("ERalpha", "7BAA", "T5Z"),        # was 3UU7 / BPA
    ("ERbeta", "5TOA", "EST"),
    ("ERRgamma", "6XY5", "O4E"),       # was 2E2R / BPA
    ("PPARgamma", "9V8H", "BRL"),      # was 9F7W / BPA; BRL is rosiglitazone
    ("AR", "2AMA", "DHT"),
    ("THRbeta", "3GWS", "T3"),
    ("PR", "1A28", "STR"),
]
SWAPPED = {"ERalpha", "ERRgamma", "PPARgamma"}


def main():
    env = lmdb.open("pocket_nr16_swap.lmdb", subdir=False, map_size=200 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, (receptor, pdb_id, lig) in enumerate(ORIGINAL_7 + NEW):
            atoms, coords = extract_pocket(f"pdb/{pdb_id}.pdb", lig)
            tag = "  ← 교체" if receptor in SWAPPED else ""
            print(f"  {receptor:<11} {pdb_id}  ref {lig:<4} {len(atoms):>4}원자{tag}")
            txn.put(str(i).encode(), pickle.dumps({
                "pocket": receptor, "pocket_atoms": atoms,
                "pocket_coordinates": coords}))
    env.close()
    print("\npocket_nr16_swap.lmdb 저장")


if __name__ == "__main__":
    main()
