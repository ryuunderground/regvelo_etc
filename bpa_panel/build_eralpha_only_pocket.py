"""Single-pocket lmdb containing only ERalpha, so scoring isn't max-pooled
across the whole 7-receptor panel."""
import pickle

import lmdb

from build_pockets import extract_pocket

atoms, coords = extract_pocket("pdb/3UU7.pdb", "2OH")
env = lmdb.open("pocket_eralpha_only.lmdb", subdir=False, map_size=10 * 1024 * 1024)
with env.begin(write=True) as txn:
    txn.put(b"0", pickle.dumps({"pocket": "ERalpha", "pocket_atoms": atoms, "pocket_coordinates": coords}))
env.close()
print("wrote ERalpha-only pocket:", len(atoms), "atoms")
