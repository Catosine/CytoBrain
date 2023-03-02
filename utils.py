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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from constant import TRAIN_MEAN, TRAIN_STD, TEST_MEAN, TEST_STD

def __base_argParse(parser):

    parser.add_argument("--data", type=str, default="/Users/cytosine/Documents/Algonauts2023/data",
                        help="Path to Algonauts2023 dataset")
    parser.add_argument("--subject", type=str, default="subj01",
                        choices=["subj01", "subj02", "subj03", "subj04",
                                 "subj05", "subj06", "subj07", "subj08"],
                        help="Used to select which subject to learn")
    parser.add_argument("--hemisphere", type=str, choices=["L", "R"], default="L",
                        help="Select which half of the brain to model")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet18", "resnet50"], help="Select different models")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed")

    parser.add_argument("--note", type=str, help="Note.")

    return parser


def __train_argParse(parser):

    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/Dev set ratio")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Epoch number to train the model")

    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--lr_regressor", type=float,
                        help="Learing rate for regressor")

    parser.add_argument("--report_step", type=int, default=10,
                        help="Num of report steps for logging")

    parser.add_argument("--save_path", type=str, default="./logs",
                        help="Path to save training logs and models")

    return parser


def __infer_argParse(parser):

    parser.add_argument("--pretrained_weight", type=str,
                        help="Path to pretrained weight")
    parser.add_argument("--save_path", type=str, default="./prediction",
                        help="Path to save training logs and models")
    parser.add_argument("--output_size", type=int, default=2048, help="Output size of pretrained model")

    return parser


def train_argParse():

    parser = argparse.ArgumentParser()

    parser = __base_argParse(parser)
    parser = __train_argParse(parser)

    return parser.parse_args()


def infer_argParse():

    parser = argparse.ArgumentParser()

    parser = __base_argParse(parser)
    parser = __infer_argParse(parser)

    return parser.parse_args()


def train_dev_split(dset, train_ratio=0.8):

    total_size = len(dset)
    train_idx = np.random.choice(total_size, int(
        total_size * train_ratio), replace=False)
    dev_idx = np.ones(total_size)
    dev_idx[train_idx] = 0
    dev_idx = np.argwhere(dev_idx).reshape(-1)

    return Subset(dset, train_idx), Subset(dset, dev_idx)


def __common_initialize(args):

    # set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup logs dir
    if not osp.isdir(args.save_path):
        os.mkdir(args.save_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def train_initialize(args):

    __common_initialize(args)

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

    return args


def inference_initialize(args):

    __common_initialize(args)

    args.save_path = osp.join(args.save_path, "{subj}_{hemisphere}_{model}".format(
        subj=args.subject, hemisphere=args.hemisphere, 
        model=args.pretrained_weight.split("/")[-1].split(".")[0]))
    if args.note:
        args.save_path += "_{}".format(args.note)

    if not osp.isdir(args.save_path):
        os.mkdir(args.save_path)

    return args


def build_optimizer(model, lr, lr_regressor):

    lr_regressor = lr_regressor if lr_regressor else lr

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    regressor_param = list()
    regressor_param_no_decay = list()
    backbone_param = list()
    backbone_param_no_decay = list()
    for n, p in param_optimizer:
        if n.startswith(("fc")):
            if not any(nd in n for nd in no_decay):
                regressor_param.append(p)
            else:
                regressor_param_no_decay.append(p)
        else:
            if not any(nd in n for nd in no_decay):
                backbone_param.append(p)
            else:
                backbone_param_no_decay.append(p)

    optimizer_grouped_parameters = [
        {'params': backbone_param, 'weight_decay_rate': 0.01, "lr": lr},
        {'params': backbone_param_no_decay, 'weight_decay_rate': 0.0, "lr": lr},
        {'params': regressor_param, 'weight_decay_rate': 0.01, "lr": lr_regressor},
        {'params': regressor_param_no_decay,
            'weight_decay_rate': 0.0, "lr": lr_regressor}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    return optimizer


def build_model(model, output_size, pretrained=None):

    if model == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_size, bias=True)
        )
    elif model == "resnet50":
        model = torchvision.models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=output_size, bias=True)
        )
    else:
        raise NotImplemented

    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)

    return model


def build_transform(subj, train=True):

    """
        Build transform as preprocessing
    """

    mean = TRAIN_MEAN[subj] if train else TEST_MEAN[subj]
    std = TRAIN_STD[subj] if train else TEST_STD[subj]

    tf = [transforms.ToTensor()]

    if train:
        tf += [
            #transforms.Pad(64),
            transforms.RandomCrop(size=(256)),
            transforms.RandomHorizontalFlip(),
        ]

    tf.append(transforms.Normalize(mean, std))

    tf = transforms.Compose(tf)

    return tf


def compute_pearson(pred, target):
    
    """
        Compute pearson correlation coefficient

        Args:
            pred,           torch.Tensor, prediction
            target,         torch.Tensor, target
    """

    pearson = torch.corrcoef(torch.concat([pred, target]))
    size = pred.size(0)
    mask = [[False for _ in range(size*2)] for _ in range(size*2)]

    for i in range(size):
        mask[i][i+size] = True

    pearson = pearson[mask]

    return pearson


if __name__ == "__main__":

    a = torch.rand(16, 128)
    b = torch.rand(16, 128)

    print(torch.concat([a, b]).shape)

    res = compute_pearson(a, b)

    for i, (x, y) in enumerate(zip(a, b)):

        print("Computed: {} Torch.corrcoef: {}".format(
            res[i], torch.corrcoef(torch.stack([x, y]))[0, 1]))

