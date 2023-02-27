# Made by Cyto
#     　　　　 ＿ ＿
# 　　　　　／＞　　 フ
# 　　　　　|   _　 _l
# 　 　　　／` ミ＿xノ
# 　　 　 /　　　 　 |
# 　　　 /　 ヽ　　 ﾉ
# 　 　 │　　|　|　|
# 　／￣|　　 |　|　|
# 　| (￣ヽ＿_ヽ_)__)
# 　＼二つ ；
import argparse
import logging
import os
import os.path as osp
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter


def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/Users/cytosine/Documents/Algonauts2023/data",
                        help="Path to Algonauts2023 dataset")
    parser.add_argument("--subject", type=str, default="subj01",
                        choices=["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"],
                        help="Used to select which subject to learn")
    parser.add_argument("--hemisphere", type=str, choices=["L", "R"], default="L",
                        help="Select which half of the brain to model")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50"], help="Select different models")
    # parser.add_argument("--pretrained_weight", type=str, help="Path to pretrained weight")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--kfold", type=int, default=5, help="KFold for cross validation")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch number to train the model")

    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")

    parser.add_argument("--seed", type=int, default=1001, help="Random seed")

    parser.add_argument("--report_step", type=int, default=100, help="Num of report steps for logging")
    parser.add_argument("--save_path", type=str, default="./logs", help="Path to save training logs and models")
    parser.add_argument("--note", type=str, help="Note for training.")

    parser.add_argument("--inference", action="store_true", help="Run inference script.")

    # parser.add_argument("--config", type=str, help="Path of a json defined configuration. Can be used to save time.")

    return parser.parse_args()


def initialize(args):
    # set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup logs dir
    if not osp.isdir(args.save_path):
        os.mkdir(args.save_path)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    args.save_path = osp.join(args.save_path, "{timestamp}_{subj}_{hemisphere}_{model}_rs{rs}".format(
        timestamp=timestamp, subj=args.subject, hemisphere=args.hemisphere, model=args.model, rs=args.seed))
    if args.note:
        args.save_path += "_{}".format(args.note)

    # init tensorboard & logging
    args.summarywriter = SummaryWriter(args.save_path)
    log_path = osp.join(args.save_path, "training_logs.txt")

    time_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=time_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(time_format))
    logging.getLogger().addHandler(fh)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
