"""Extend the panel from 7 pockets to 16, keeping the original file untouched.

The 7-pocket set has one clear negative (THRbeta) and every other receptor is a
reported BPA target, so specificity cannot really be tested. What is missing are
the *sibling* receptors, whose binding pockets are the most similar and
therefore the hardest negatives:

  steroid   AR, PR  ->  add GR, MR
  PPAR      PPARgamma -> add PPARalpha, PPARdelta
  THR       THRbeta -> add THRalpha
  ERR       ERRgamma -> add ERRbeta (6LIT is a BPA co-crystal)
  plus VDR, PXR, RXRalpha, which the lab's Glide runs already cover

PDB entries and reference ligands were picked by find_pocket_candidates.py from
resolution-sorted RCSB hits, reading off each entry's actual components rather
than from memory -- carving a pocket around a cryoprotectant is an easy mistake.
4UDD was skipped for GR because its largest component is CHAPS detergent, and
5U3Q's second component is a glucopyranoside detergent, not the ligand.

Writes pocket_nr16.lmdb. build_pockets.py and pocket_bpa_panel.lmdb are left
alone so every earlier result stays reproducible.
"""
import pickle

import lmdb

from build_pockets import TARGETS as ORIGINAL, extract_pocket

NEW = [
    # receptor, pdb, reference ligand -- siblings first
    ("GR", "4P6W", "MOF"),          # mometasone furoate; steroid sibling of AR/PR
    ("MR", "4PF3", "HFN"),          # steroid sibling
    ("PPARalpha", "6KAX", "PLM"),   # palmitic acid, a native PPARalpha ligand
    ("PPARdelta", "5U3Q", "7UJ"),
    ("THRalpha", "2H79", "T3"),     # same ligand as our THRbeta pocket
    ("ERRbeta", "6LIT", "2OH"),     # BPA itself -- a co-crystal we can verify against
    ("VDR", "1IE9", "VDX"),         # calcitriol, the natural hormone
    ("PXR", "9FZJ", "SRL"),
    ("RXRalpha", "6LB4", "E8L"),
]

OUT = "pocket_nr16.lmdb"


def main():
    env = lmdb.open(OUT, subdir=False, map_size=200 * 1024 * 1024)
    ok, failed = 0, []
    with env.begin(write=True) as txn:
        for i, (receptor, pdb_id, lig) in enumerate(list(ORIGINAL) + NEW):
            try:
                atoms, coords = extract_pocket(f"pdb/{pdb_id}.pdb", lig)
            except Exception as e:                        # noqa: BLE001
                failed.append((receptor, pdb_id, lig, str(e)[:60]))
                print(f"  {receptor:<11} {pdb_id}  실패: {str(e)[:60]}")
                continue
            tag = "" if receptor in [t[0] for t in ORIGINAL] else "  (신규)"
            print(f"  {receptor:<11} {pdb_id}  ref {lig:<4} {len(atoms):>4}원자{tag}")
            txn.put(str(i).encode(), pickle.dumps({
                "pocket": receptor,
                "pocket_atoms": atoms,
                "pocket_coordinates": coords,
            }))
            ok += 1
    env.close()
    print(f"\n{ok}개 포켓 -> {OUT}")
    if failed:
        print("실패:", failed)


if __name__ == "__main__":
    main()
