#!/bin/bash
cd "$(dirname "$0")"
N=12
for i in $(seq 0 $((N-1))); do
  ../.venv/bin/python run_vina.py --smiles-csv chembl_sample.csv --out-prefix chembldock \
    --shard $i --nshards $N > chembldock_shard_$i.log 2>&1 &
done
wait
echo "CHEMBL DONE"
