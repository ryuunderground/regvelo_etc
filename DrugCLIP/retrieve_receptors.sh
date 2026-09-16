results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"
MOL_PATH="query_ligand.lmdb"     # path to the fixed query ligand(s)
POCKET_PATH="receptor_lib.lmdb"  # path to the candidate receptor pocket library
EMB_DIR="./data/emb"             # path to the cached pocket embedding file
data_path="data"

python ./unimol/retrieve_receptors.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip \
       --max-pocket-atoms 256 \
       --cpu --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --emb-dir $EMB_DIR
