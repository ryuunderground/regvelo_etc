#!/bin/bash
# Re-score the EDC panel with a given checkpoint.  All three comparison points
# (pretrained / v1 / v2) must be scored with identical settings -- notably
# --max-pocket-atoms 450, since PPARgamma's pocket is 422 atoms and the default
# 256 would silently truncate it.
#   usage: ./score_panels.sh <checkpoint> <out_dir>
set -e
cd "$(dirname "$0")"
CKPT=$1; OUT=$2
../.venv/bin/python ../DrugCLIP/unimol/score_matrix.py train_data \
  --user-dir ../DrugCLIP/unimol --valid-subset valid \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path "$CKPT" --results-path "$OUT" --emb-dir "$OUT/emb" \
  --mol-path mols_edc_panel.lmdb \
  --pocket-path ../bpa_panel/pocket_bpa_panel.lmdb \
  --max-pocket-atoms 450 --batch-size 8 --num-workers 0 --cpu --seed 1 \
  --log-interval 100 --log-format simple
