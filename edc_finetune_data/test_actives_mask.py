"""Self-check for the actives-aware in-batch mask (unimol/models/drugclip.py).

Reproduces the masking arithmetic on a hand-made batch and asserts that a
compound active at several receptors is masked at ALL of them, not just at the
one its row is paired with. Run: python test_actives_mask.py
"""
import csv
import pickle
import sys

import lmdb
import torch


def build_mask(pocket_list, smi_list, actives_list):
    """Same arithmetic as DrugCLIP.forward; returns the -1e6 indicator matrix."""
    n = len(pocket_list)
    actives = [set(str(a).split("|")) for a in actives_list]
    pocket_dup = torch.tensor(
        [[1 if pocket_list[i] in actives[j] else 0 for j in range(n)] for i in range(n)],
        dtype=torch.float32)
    mol_dup = torch.tensor(
        [[1 if smi_list[i] == smi_list[j] else 0 for j in range(n)] for i in range(n)],
        dtype=torch.float32)
    return pocket_dup + mol_dup - 2 * torch.eye(n)


def masked(m):
    """forward() applies `indicator * -1e6`, so any positive entry is excluded."""
    return m > 0


def test_masking():
    #          row:  pocket      mol   actives of that mol
    batch = [("ERalpha", "mA", "ERalpha|PR"),   # promiscuous: also a PR binder
             ("PR",      "mB", "PR"),           # PR-only
             ("PR",      "mA", "ERalpha|PR"),   # same mol as row 0, other pocket
             ("AR",      "mC", "AR")]
    pockets, smis, actives = zip(*batch)
    m = build_mask(list(pockets), list(smis), list(actives))

    x = masked(m)
    assert torch.all(m.diagonal() == 0), "diagonal (the true pair) must stay unmasked"
    # the fix: PR row vs mA, which is a confirmed PR active but paired to ERalpha
    assert x[1][0], "promiscuous active not masked at its other receptor"
    assert x[1][2], "row 2 is a PR positive, must be masked from row 1"
    # old same-pocket behaviour still holds
    assert not x[1][1] and not x[2][2]
    # unrelated receptor stays a usable negative
    assert not x[3][0] and not x[0][3], "AR/ERalpha pair wrongly masked"
    # mA appears twice; both copies mask each other on the mol axis too
    assert x[0][2] and x[2][0]


def test_fallback_matches_old_behaviour():
    """actives_list absent -> model passes pocket_list, i.e. same-pocket masking."""
    pockets = ["ERalpha", "PR", "ERalpha"]
    smis = ["mA", "mB", "mC"]
    x = masked(build_mask(pockets, smis, pockets))  # fallback path
    assert x[0][2] and x[2][0], "same pocket must still be masked"
    assert not x[0][1]


def test_lmdb_has_actives():
    """The built LMDB carries the field, and it is consistent with pairs.csv."""
    try:
        env = lmdb.open("train_data/train.lmdb", readonly=True, lock=False, subdir=False)
    except lmdb.Error:
        print("skip: train_data/train.lmdb not built")
        return
    rows = list(csv.DictReader(open("pairs.csv")))
    smi2cid = {}
    m = lmdb.open("mols_edc_panel.lmdb", readonly=True, lock=False, subdir=False)
    with m.begin() as t:
        for k, v in t.cursor():
            smi2cid[pickle.loads(v)["smi"]] = k.decode()
    m.close()
    truth = {}
    for r in rows:
        if r["label"] == "positive":
            truth.setdefault(r["cid"], set()).add(r["receptor"])
    with env.begin() as t:
        for _, v in t.cursor():
            d = pickle.loads(v)
            cid = smi2cid[d["smi"]]
            got = set(d["actives"].split("|"))
            assert d["pocket"] in got, f"{cid}: own pocket missing from actives"
            assert got <= truth[cid], f"{cid}: actives {got} not backed by pairs.csv"
    env.close()


if __name__ == "__main__":
    test_masking()
    test_fallback_matches_old_behaviour()
    test_lmdb_has_actives()
    print("ok")
    sys.exit(0)
