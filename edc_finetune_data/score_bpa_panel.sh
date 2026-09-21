#!/bin/bash
# Same as score_panels.sh but for the 11-compound bisphenol panel.
#   usage: ./score_bpa_panel.sh <checkpoint> <out_dir>
set -e
cd "$(dirname "$0")"
../.venv/bin/python ../DrugCLIP/unimol/score_matrix.py train_data \
  --user-dir ../DrugCLIP/unimol --valid-subset valid \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path "$1" --results-path "$2" --emb-dir "$2/emb" \
  --mol-path ../bpa_panel/mols_bpa_panel.lmdb \
  --pocket-path ../bpa_panel/pocket_bpa_panel.lmdb \
  --max-pocket-atoms 450 --batch-size 8 --num-workers 0 --cpu --seed 1 \
  --log-interval 100 --log-format simple
