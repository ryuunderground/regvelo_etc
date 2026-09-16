"""Add literature-verification columns to results_ranked.csv for the 4
compounds that scored above BPA for ERalpha, based on parallel literature
research (see eralpha_literature_check.csv for full detail/citations)."""
import csv

VERDICTS = {
    ("ERalpha", "BPZ"): (
        "CONFIRMED_STRONGER",
        "Mesnage 2017 Toxicol Sci (E-Screen AC50 BPZ 0.11uM vs BPA 0.36uM, ~3x) and "
        "Pelch 2019 Toxicol Sci (EC50 BPZ 4.0e-7M vs BPA 1.2e-6M, ~3x) both agree.",
    ),
    ("ERalpha", "BPB"): (
        "CONFIRMED_STRONGER",
        "Pelch 2019 (EC50 BPB 3.2e-7M vs BPA 1.2e-6M, ~3.8x) and Serra 2019 EHP "
        "systematic review of 15 studies ('BPB potency similar to or greater than BPA').",
    ),
    ("ERalpha", "BPC"): (
        "WRONG_COMPOUND_UNVERIFIED",
        "Panel's BPC = PubChem CID 6620 (dimethyl regioisomer), which has NO published "
        "ERalpha data. The toxicologically notorious 'BPC' in Delfosse 2012 PNAS / Liu "
        "2021 PLOS ONE (ERalpha IC50 ~2.7nM, ~410x BPA) is a DIFFERENT compound "
        "(CAS 14868-03-2, dichloro-vinylidene bridge). Score not validated either way.",
    ),
    ("ERalpha", "TCBPA"): (
        "MIXED",
        "Riu 2011 Toxicol Sci: EC50 TCBPA 0.53uM vs BPA 0.29uM (comparable, BPA "
        "slightly more potent). Pelch 2019: TCBPA only 21.3% max efficacy vs E2 "
        "(near-inactive). Two studies disagree; does not confirm TCBPA > BPA.",
    ),
}

with open("results_ranked.csv") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys()) + ["eralpha_literature_verdict", "eralpha_literature_note"]
for row in rows:
    key = (row["receptor"], row["compound"])
    verdict, note = VERDICTS.get(key, ("", ""))
    row["eralpha_literature_verdict"] = verdict
    row["eralpha_literature_note"] = note

with open("results_ranked.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("annotated", sum(1 for r in rows if r["eralpha_literature_verdict"]), "rows")
