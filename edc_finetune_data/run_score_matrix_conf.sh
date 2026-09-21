#!/bin/bash
# Score every conformer instead of the single arbitrary one AffinityMolDataset
# draws, so eval_auroc.py can compare conf0 / max / mean pooling.
#
# Conformer counterpart to score_panels.sh, same settings -- notably
# --max-pocket-atoms 450, since PPARgamma's pocket is 422 atoms and the
# default 256 would silently truncate it.
#   usage: ./run_score_matrix_conf.sh [checkpoint] [out_dir]
#
# emb-dir lives under out_dir because encode_*_once caches on the LMDB
# basename alone, not the checkpoint; a shared dir would serve one
# checkpoint's vectors to another.
set -e
cd "$(dirname "$0")"
CKPT=${1:-save_dir/checkpoint_best.pt}
OUT=${2:-eval_conf}
../.venv/bin/python ../DrugCLIP/unimol/score_matrix.py train_data \
  --user-dir ../DrugCLIP/unimol \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path "$CKPT" \
  --mol-path mols_edc_conformers.lmdb \
  --pocket-path ../bpa_panel/pocket_bpa_panel.lmdb \
  --emb-dir "$OUT/emb" \
  --results-path "$OUT" \
  --max-pocket-atoms 450 --batch-size 32 --cpu --num-workers 0 \
  --log-format simple

echo
echo "now compare pooling rules:"
echo "  uv run --with lmdb python eval_auroc.py \\"
echo "    eval_new/score_matrix.csv \\"
echo "    eval_conf/score_matrix.csv:conf0 \\"
echo "    eval_conf/score_matrix.csv:max \\"
echo "    eval_conf/score_matrix.csv:mean"
