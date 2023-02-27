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
import torch.utils.data as data
import torchvision

from dataset import Algonauts2023Raw
from utils import train_argParse, initialize


def main(args):
    # initialize
    args = initialize(args)
    logging.info("Initialization ready.")
    logging.info(args)

    # load dataset
    train_set = Algonauts2023Raw(osp.join(args.data, args.subject))
    logging.info("Training data <{}> loaded.".format(args.subject))
    logging.info("Total samples: {}".format(len(train_set)))
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=100)

    print(model)

    for i, f in data.DataLoader(train_set, batch_size=8):
        print(i)
        print(f)
        print(i.shape)
        print(f.shape)
        print(type(i))
        print(type(f))
        break

    pass


if __name__ == "__main__":
    args = train_argParse()
    main(args)
