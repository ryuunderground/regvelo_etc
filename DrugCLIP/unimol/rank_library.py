#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Rank every molecule in a library against a single pocket, sorted, no
# top-k truncation -- for finding one specific compound's true percentile
# against a broad/diverse library instead of retrieval.py's top-10000 list.

import csv
import logging
import os
import sys
import torch
from unicore import checkpoint_utils, distributed_utils, options
from unicore import tasks
import unicore

if not torch.cuda.is_available():
    unicore.utils.move_to_cuda = lambda sample, device=None: sample

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.rank_library")


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    logger.info(args)

    model.eval()

    pocket_names, mol_names, res = task.score_matrix(model, args.mol_path, args.pocket_path, args.emb_dir)
    assert len(pocket_names) == 1, "rank_library.py expects exactly one pocket"
    scores = res[0]

    order = sorted(range(len(mol_names)), key=lambda i: -scores[i])

    os.makedirs(args.results_path, exist_ok=True)
    out_path = os.path.join(args.results_path, "library_ranking.csv")
    n = len(mol_names)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "percentile", "smi", "score"])
        for rank, i in enumerate(order, start=1):
            percentile = 100 * (n - rank + 1) / n
            w.writerow([rank, f"{percentile:.3f}", mol_names[i], f"{scores[i]:.4f}"])
    logger.info(f"wrote {n} ranked compounds against pocket {pocket_names[0]} to {out_path}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path to the compound library")
    parser.add_argument("--pocket-path", type=str, default="", help="path to a single-pocket lmdb")
    parser.add_argument("--emb-dir", type=str, default="", help="path for cached embeddings")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
