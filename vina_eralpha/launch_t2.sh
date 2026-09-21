#!/bin/bash
cd "$(dirname "$0")"
N=12
for i in $(seq 0 $((N-1))); do
  ../.venv/bin/python run_vina.py --cid-file t2_cids.txt --out-prefix t2 \
    --shard $i --nshards $N > t2_shard_$i.log 2>&1 &
done
wait
echo "T2 DONE"
