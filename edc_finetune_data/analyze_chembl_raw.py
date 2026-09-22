"""Read the cached ChEMBL pull and settle two design questions.

1. Censored records. '>' means "did not bind up to this concentration", which
   makes them the most confident non-binders in the database -- and filtering to
   standard_relation='=' drops them, which is why the inactive pool is small.

2. Between-assay spread. pChEMBL pools IC50/Ki/Kd across labs and formats. For
   compounds measured several times, the spread across those measurements is the
   noise floor for using pChEMBL as a continuous regression target.

  python analyze_chembl_raw.py chembl_eralpha_raw.json
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict


def main(path):
    acts = json.load(open(path))
    print(f"{len(acts)}건\n")

    print("standard_relation 분포:")
    for k, v in Counter(a.get("standard_relation") for a in acts).most_common():
        print(f"  {str(k):<6}{v:>7}  ({v/len(acts):5.1%})")

    # what are the None-relation rows? if they carry a value and a pChEMBL they
    # are usable; if not they are just unparsed records
    none_rows = [a for a in acts if a.get("standard_relation") is None]
    print(f"\nrelation=None {len(none_rows)}건의 정체:")
    print(f"  standard_value 있음 : {sum(1 for a in none_rows if a.get('standard_value'))}")
    print(f"  pchembl_value 있음  : {sum(1 for a in none_rows if a.get('pchembl_value'))}")

    print("\nstandard_type 분포 (상위 8):")
    for k, v in Counter(a.get("standard_type") for a in acts).most_common(8):
        print(f"  {str(k):<12}{v:>7}")

    # ---- censored ------------------------------------------------------
    exact_mol = {a["molecule_chembl_id"] for a in acts
                 if a.get("standard_relation") == "=" and a.get("pchembl_value")}
    cens = [a for a in acts if a.get("standard_relation") == ">"]
    cens_mol = {a["molecule_chembl_id"] for a in cens}
    only_cens = cens_mol - exact_mol
    print(f"\n--- 검열 기록 ---")
    print(f"'>' 기록 {len(cens)}건, 화합물 {len(cens_mol)}개")
    print(f"  '=' 기록이 전혀 없는 화합물: {len(only_cens)}개  <- 현재 완전 누락")
    vals = sorted(float(a["standard_value"]) for a in cens
                  if a.get("standard_value") and a.get("standard_units") == "nM")
    if vals:
        print(f"  '>' 값 (nM): 25% {vals[len(vals)//4]:,.0f}  중앙 {st.median(vals):,.0f}  "
              f"75% {vals[3*len(vals)//4]:,.0f}")
        strong = sum(1 for v in vals if v >= 10000)
        print(f"  10µM 이상에서 미결합 확인: {strong}/{len(vals)} ({strong/len(vals):.0%})")
    print(f"현재 Inactive(pChEMBL<5) 456개 -> 복구 시 약 {456 + len(only_cens)}개")

    # ---- between-assay spread ------------------------------------------
    by_mol, types = defaultdict(list), defaultdict(list)
    for a in acts:
        if a.get("pchembl_value") and a.get("standard_relation") == "=":
            by_mol[a["molecule_chembl_id"]].append(float(a["pchembl_value"]))
            types[a["molecule_chembl_id"]].append(a.get("standard_type"))
    rep = {m: v for m, v in by_mol.items() if len(v) >= 3}
    print(f"\n--- 배치 효과 ---")
    print(f"3회 이상 측정된 화합물: {len(rep)}개 (이전 43개)")
    spreads = sorted(max(v) - min(v) for v in rep.values())
    sds = sorted(st.pstdev(v) for v in rep.values())
    print(f"  최대-최소 (log): 25% {spreads[len(spreads)//4]:.2f}  "
          f"중앙 {st.median(spreads):.2f}  75% {spreads[3*len(spreads)//4]:.2f}")
    m_sd = st.median(sds)
    print(f"  표준편차 중앙값: {m_sd:.2f} log = 약 {10**m_sd:.1f}배")

    same = [max(v) - min(v) for m, v in rep.items() if len(set(types[m])) == 1]
    mixed = [max(v) - min(v) for m, v in rep.items() if len(set(types[m])) > 1]
    if same and mixed:
        print(f"  측정종류 단일 {len(same):>4}개: 퍼짐 중앙 {st.median(same):.2f}")
        print(f"  측정종류 혼합 {len(mixed):>4}개: 퍼짐 중앙 {st.median(mixed):.2f}")

    # if we restrict to one type, how much data survives and how tight is it?
    print("\n  종류별로 따로 보면 (3회 이상, 그 종류만):")
    for t in ("IC50", "Ki", "Kd", "EC50"):
        per = defaultdict(list)
        for a in acts:
            if (a.get("standard_type") == t and a.get("pchembl_value")
                    and a.get("standard_relation") == "="):
                per[a["molecule_chembl_id"]].append(float(a["pchembl_value"]))
        r = [v for v in per.values() if len(v) >= 3]
        if len(r) >= 5:
            sp = sorted(max(v) - min(v) for v in r)
            print(f"    {t:<6} 화합물 {len(per):>5}개 / 반복측정 {len(r):>4}개 / "
                  f"퍼짐 중앙 {st.median(sp):.2f} log")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chembl_eralpha_raw.json")
