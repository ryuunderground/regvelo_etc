#!/bin/bash
# ChEMBL은 오늘 여러 번 429/500을 냈다. 리셉터 사이에 90초를 두고 순차로 받는다.
cd "$(dirname "$0")"
set -- CHEMBL242:ERbeta CHEMBL4245:ERRgamma CHEMBL1871:AR CHEMBL208:PR \
       CHEMBL1947:THRbeta CHEMBL2034:GR CHEMBL235:PPARgamma CHEMBL2061:RXRalpha
for pair in "$@"; do
  tid=${pair%%:*}; name=${pair##*:}
  f="chembl_${name}_raw.json"
  [ -f "$f" ] && { echo "$name 이미 있음"; continue; }
  echo "=== $name ($tid) ==="
  ../.venv/bin/python fetch_chembl_full.py "$tid" "$f"
  sleep 90
done
echo "ALL RECEPTORS DONE"
