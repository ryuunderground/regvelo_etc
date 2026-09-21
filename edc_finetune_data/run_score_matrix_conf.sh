#!/bin/bash
# Score every conformer instead of the single arbitrary one AffinityMolDataset
# draws, so eval_auroc.py can compare conf0 / max / mean pooling.
#
# Uses the v1 checkpoint on purpose: eval_new/score_matrix.csv is v1, so the
# pooled numbers stay a within-checkpoint comparison and isolate the conformer
# effect from the actives-mask change that save_dir_v2 introduces.
#
# emb-dir must be fresh -- encode_*_once caches on the LMDB basename alone,
# not the checkpoint, so reusing emb_new would serve stale v1 pocket vectors.
set -e
cd "$(dirname "$0")"
../.venv/bin/python ../DrugCLIP/unimol/score_matrix.py train_data \
  --user-dir ../DrugCLIP/unimol \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --path save_dir/checkpoint_best.pt \
  --mol-path mols_edc_conformers.lmdb \
  --pocket-path ../bpa_panel/pocket_bpa_panel.lmdb \
  --emb-dir ../DrugCLIP/data/emb_conf \
  --results-path eval_conf \
  --max-pocket-atoms 450 --batch-size 32 --cpu --num-workers 0 \
  --log-format simple

echo
echo "now compare pooling rules:"
echo "  uv run --with lmdb python eval_auroc.py \\"
echo "    eval_new/score_matrix.csv \\"
echo "    eval_conf/score_matrix.csv:conf0 \\"
echo "    eval_conf/score_matrix.csv:max \\"
echo "    eval_conf/score_matrix.csv:mean"
