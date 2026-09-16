"""Random sample from the real 2.94M-compound library (DrugCLIP/mols.lmdb),
plus BPA appended, to get BPA's percentile against a broad, non-bisphenol
chemical space -- not just against its own analogs."""
import pickle
import random

import lmdb

N_SAMPLE = 20000
SEED = 1

random.seed(SEED)

src = lmdb.open("../DrugCLIP/mols.lmdb", readonly=True, lock=False, subdir=False)
with src.begin() as txn:
    n_total = txn.stat()["entries"]
    sample_idx = random.sample(range(n_total), N_SAMPLE)

dst = lmdb.open("mols_library_sample.lmdb", subdir=False, map_size=2 * 1024 * 1024 * 1024)
with src.begin() as stxn, dst.begin(write=True) as dtxn:
    n_written = 0
    for idx in sample_idx:
        v = stxn.get(str(idx).encode())
        if v is None:
            continue
        dtxn.put(str(n_written).encode(), v)
        n_written += 1

    # append BPA (same entry used in the bisphenol panel)
    bpa_env = lmdb.open("mols_bpa_panel.lmdb", readonly=True, lock=False, subdir=False)
    with bpa_env.begin() as btxn:
        bpa_entry = pickle.loads(btxn.get(b"0"))
        assert bpa_entry["abbreviation"] == "BPA"
    bpa_env.close()
    dtxn.put(str(n_written).encode(), pickle.dumps(bpa_entry))
    bpa_index = n_written
    n_written += 1

src.close()
dst.close()
print(f"wrote {n_written} compounds ({n_written - 1} sampled + BPA at index {bpa_index})")
