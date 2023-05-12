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
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer
if transformers.__version__ >= "5.0.0":
    from transformers import ViTImageProcessor as image_processor
else:
    from transformers import ViTFeatureExtractor as image_processor

from src.dataset import Algonauts2023Raw
from src.model import VisionEncoderDecoderRegressor as VEDR
from src.utils import train_dev_split, build_optimizer, compute_pearson_torch, my_training_collate_fn
from src.arg_parse import train_argParse
from src.initialize import train_initialize
from src.trainer import NNTrainer


def main(args):
    # initialize
    args = train_initialize(args)
    logging.info("Initialization ready.")
    logging.info(args)

    # load dataset
    dataset = Algonauts2023Raw(osp.join(
        args.data, args.subject), hemisphere=args.hemisphere, caption_file=osp.join(args.data, args.caption), train=True, return_pil=args.use_pil)
    logging.info("Dataset <{}> loaded.".format(args.subject))

    # split train & dev set
    train_set, dev_set = train_dev_split(dataset, args.train_ratio)
    logging.info("#Total: {}".format(len(dataset)))
    logging.info("#Train: {}".format(len(train_set)))
    logging.info("#Dev: {}".format(len(dev_set)))
    fmri_size = train_set[0][1].shape[0]
    logging.info("Hemisphere FMRI size: {}".format(fmri_size))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_training_collate_fn, num_workers=4)
    val_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_training_collate_fn, num_workers=4)

    # setup model
    #model = build_model(args.model, fmri_size, args.pretrained_weight)
    model = VEDR.from_pretrained(args.pretrained_weights, regressor_out_features=fmri_size,
                                 regressor_dropout_prob=0.1, regressor_feature_method="last4", use_both_encoder_decoder_features=True)
    model.to(args.device)

    logging.info("Model initialized. Loaded to <{}> device.".format(args.device))

    # setup feature extractor and tokenizer
    feat_extractor = image_processor.from_pretrained(args.pretrained_weights)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)

    # setup optimizer & scheduler
    optimizer = build_optimizer(model, args.lr, args.lr_regressor)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=args.epoch*len(train_loader))

    # setup scoring function
    scoring_fn = compute_pearson_torch

    # setup criterion
    criterion = nn.MSELoss()

    # initializing trainer
    trainer = NNTrainer(model, feat_extractor, tokenizer, criterion, scoring_fn, optimizer,
                        scheduler, args.summarywriter, logging, args.save_path)

    # start training
    trainer.run(train_loader, val_loader, args.epoch, args.report_step, args.eval_step, args.early_stopping)


if __name__ == "__main__":
    args = train_argParse()
    main(args)
