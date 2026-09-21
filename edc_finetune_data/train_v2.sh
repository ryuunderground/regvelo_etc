#!/bin/bash
# Re-run of the EDC hard-negative fine-tune with the actives-aware in-batch mask.
# Same hyperparameters as the first run (train_run.log), only save dirs differ.
set -e
cd "$(dirname "$0")"
../.venv/bin/unicore-train train_data \
  --user-dir ../DrugCLIP/unimol \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 \
  --clip-norm 1.0 --weight-decay 0.0 \
  --lr-scheduler polynomial_decay --lr 1e-5 --warmup-ratio 0.06 \
  --total-num-update 1000000 --max-epoch 3 \
  --batch-size 8 --batch-size-valid 8 --max-pocket-atoms 450 \
  --seed 1 --cpu --num-workers 0 --find-unused-parameters \
  --log-interval 20 --log-format simple \
  --save-dir save_dir_v2 --tmp-save-dir tmp_save_dir_v2 \
  --restore-file "$PWD/restore_checkpoint.pt" \
  --reset-optimizer --reset-dataloader --reset-lr-scheduler --reset-meters \
  --best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric \
  --keep-last-epochs 3 --patience 2000
