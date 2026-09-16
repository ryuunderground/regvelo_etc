#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
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
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)

    # ponytail: upstream never surfaced the ranked results; write them out.
    os.makedirs(args.results_path, exist_ok=True)
    out_path = os.path.join(args.results_path, "retrieval_results.tsv")
    with open(out_path, "w") as f:
        for name, score in zip(names, scores):
            f.write(f"{name}\t{score:.4f}\n")
    logger.info(f"wrote {len(names)} ranked results to {out_path}")
    for name, score in list(zip(names, scores))[:10]:
        logger.info(f"{score:.4f}\t{name}")


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--emb-dir", type=str, default="", help="path for saved embedding data")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
