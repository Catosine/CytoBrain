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
import os.path as osp

import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics import SpearmanCorrCoef

from dataset import Algonauts2023Raw
from utils import train_argParse, initialize, train_dev_split, build_optimizer
from trainer import NNTrainer


def main(args):
    # initialize
    args = initialize(args)
    logging.info("Initialization ready.")
    logging.info(args)

    # load dataset
    dataset = Algonauts2023Raw(osp.join(args.data, args.subject))
    logging.info("Dataset <{}> loaded.".format(args.subject))

    # split train & dev set
    train_set, dev_set = train_dev_split(dataset, args.train_ratio)

    logging.info("#Total: {}".format(len(dataset)))
    logging.info("#Train: {}".format(len(train_set)))
    logging.info("#Dev: {}".format(len(dev_set)))
    fmri_size = train_set[0][1].shape[0]
    logging.info("Hemisphere FMRI size: {}".format(fmri_size))

    # setup model
    if args.model == "resnet50":
        model = torchvision.models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=8192, bias=True),
            nn.Linear(in_features=8192, out_features=fmri_size, bias=True)
        )
    else:
        raise NotImplemented

    model.to(args.device)
    logging.info(
        "Model initialized. Loaded to <{}> device.".format(args.device))

    # setup optimizer & scheduler
    optimizer = build_optimizer(model, args.lr, args.lr_regressor)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=100)
    
    # setup scoring function
    scoring_fn = SpearmanCorrCoef()

    # setup criterion
    criterion = nn.MSELoss()

    # initializing trainer
    trainer = NNTrainer(model, criterion, scoring_fn, optimizer, scheduler, args.summarywriter, logging, args.save_path)

    # start training
    trainer.run(train_set, dev_set, args.epoch, args.batch_size, args.report_step)

if __name__ == "__main__":
    args = train_argParse()
    main(args)
