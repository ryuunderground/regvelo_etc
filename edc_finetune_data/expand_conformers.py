"""One LMDB row per (compound, conformer).

AffinityMolDataset picks a single arbitrary conformer per compound
(affinity_dataset.py:363-380, np.random.randint(size)), so score_matrix.py
has only ever scored one shape per molecule. Expanding the rows makes it
score all of them; eval_auroc.py pools the resulting duplicate columns back
per compound.

Columns come back keyed by SMILES and every conformer of a compound shares
one, but base SMILES are unique across the panel (1551/1551), so grouping by
column header recovers the compound exactly. Row order is irrelevant.
"""
import pickle

import lmdb

SRC = "mols_edc_panel.lmdb"
DST = "mols_edc_conformers.lmdb"


def main():
    src = lmdb.open(SRC, subdir=False, readonly=True, lock=False)
    dst = lmdb.open(DST, subdir=False, map_size=2 * 1024**3)

    n_mol = n_conf = 0
    with src.begin() as rt, dst.begin(write=True) as wt:
        for _, v in rt.cursor():
            entry = pickle.loads(v)
            n_mol += 1
            for k, conf in enumerate(entry["coordinates"]):
                row = dict(entry, coordinates=[conf], conf_idx=k)
                row.setdefault("label", 1)
                wt.put(str(n_conf).encode(), pickle.dumps(row))
                n_conf += 1
    src.close()
    dst.close()
    print(f"{n_mol} compounds -> {n_conf} conformer rows in {DST}")


if __name__ == "__main__":
    main()
