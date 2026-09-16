#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Full pocket x molecule similarity matrix for a small custom panel, saved as
# CSV, instead of retrieval.py/retrieve_receptors.py's top-k truncation.

import csv
import logging
import os
import sys
import torch
from unicore import checkpoint_utils, distributed_utils, options
from unicore import tasks
import unicore

if not torch.cuda.is_available():
    # ponytail: unicore.utils.move_to_cuda calls torch.cuda.current_device()
    # unconditionally; no-op it on CPU-only machines instead of forking unicore.
    unicore.utils.move_to_cuda = lambda sample, device=None: sample

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.score_matrix")


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

    os.makedirs(args.results_path, exist_ok=True)
    out_path = os.path.join(args.results_path, "score_matrix.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pocket"] + list(mol_names))
        for i, pocket in enumerate(pocket_names):
            writer.writerow([pocket] + [f"{v:.4f}" for v in res[i]])
    logger.info(f"wrote {len(pocket_names)}x{len(mol_names)} matrix to {out_path}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path to the molecule panel")
    parser.add_argument("--pocket-path", type=str, default="", help="path to the pocket panel")
    parser.add_argument("--emb-dir", type=str, default="", help="path for cached embeddings")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
