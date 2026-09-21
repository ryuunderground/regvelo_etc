#!/bin/bash
# 12 workers; Vina gets --cpu 1 each so the parallelism is across ligands,
# which scales better than Vina's internal threading.
cd "$(dirname "$0")"
N=12
for i in $(seq 0 $((N-1))); do
  ../.venv/bin/python run_vina.py --n-active 150 --n-inactive 150 \
    --shard $i --nshards $N > shard_$i.log 2>&1 &
done
wait
echo "ALL SHARDS DONE"
