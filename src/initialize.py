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
import logging
import os
import os.path as osp
import sys
import time

import torch
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


def __common_initialize(args):

    # set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup logs dir
    if not osp.isdir(args.save_path):
        os.makedirs(args.save_path)

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def train_initialize(args):

    __common_initialize(args)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    args.save_path = osp.join(args.save_path, "{timestamp}_{subj}_{hemisphere}_rs{rs}".format(
        timestamp=timestamp, subj=args.subject, hemisphere=args.hemisphere, rs=args.seed))
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

    return args


def inference_initialize(args):

    __common_initialize(args)

    args.save_path = osp.join(args.save_path, "{subj}_{model}".format(
        subj=args.subject, hemisphere=args.hemisphere,
        model=args.pretrained_weight.split("/")[-1].split(".")[0]))
    if args.note:
        args.save_path += "_{}".format(args.note)

    if not osp.isdir(args.save_path):
        os.makedirs(args.save_path)

    return args


def extract_initialize(args):

    __common_initialize(args)

    args.save_path = osp.join(
        args.save_path, args.pretrained_weight.split("/")[-1].split(".")[0])
    if args.note:
        args.save_path += "_{}".format(args.note)

    if not osp.isdir(args.save_path):
        os.makedirs(args.save_path)

    return args
