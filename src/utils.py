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
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from .constant import TRAIN_MEAN, TRAIN_STD, TEST_MEAN, TEST_STD


def train_dev_split(dset, train_ratio=0.8):

    total_size = len(dset)
    train_idx = np.random.choice(total_size, int(
        total_size * train_ratio), replace=False)
    dev_idx = np.ones(total_size)
    dev_idx[train_idx] = 0
    dev_idx = np.argwhere(dev_idx).reshape(-1)

    return Subset(dset, train_idx), Subset(dset, dev_idx)


def build_optimizer(model, lr, lr_regressor, regressor_kword="regressor"):

    lr_regressor = lr_regressor if lr_regressor else lr

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    regressor_param = list()
    regressor_param_no_decay = list()
    backbone_param = list()
    backbone_param_no_decay = list()
    for n, p in param_optimizer:
        if n.startswith((regressor_kword)):
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
    elif model == "resnet152":
        model = torchvision.models.resnet152()
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=output_size, bias=True)
        )
    elif model == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
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


def build_regression_feats(decoder_embeds, encoder_embeds=None, method="cls"):
    """
        Build the features used for regression
        Args:
            decoder_embeds,         tuple of torch.Tensor, hidden states of decoder model
            encoder_embeds,         tuple of torch.Tensor, hidden states of encoder model
            method,                 str, method of how to build regression features

        Returns:
            embeds,                 torch.Tensor, the embeddings used for regressor
    """

    if method == "cls":

        # use [CLS] feature of last layer of decoder as regression features
        embeds = decoder_embeds[-1][:, 0]

        # if encoder_embeds exists, then concat embeds with [CLS] feature of last layer of encoder
        if encoder_embeds:
            embeds = torch.concat([embeds, encoder_embeds[-1][:, 0]], dim=1)

    elif method == "mean":

        # use mean of last layer of decoder as regression features
        embeds = decoder_embeds[-1].mean(dim=1)

        # if encoder_embeds exists, then concat embeds with mean of last layer of encoder
        if encoder_embeds:
            embeds = torch.concat([embeds, encoder_embeds[-1].mean(dim=1)], dim=1)

    elif method == "max":

        # use mean of last layer of decoder as regression features
        embeds = decoder_embeds[-1].max(1).values

        # if encoder_embeds exists, then concat embeds with mean of last layer of encoder
        if encoder_embeds:
            embeds = torch.concat([embeds, encoder_embeds[-1].max(1).values], dim=1)

    elif method == "last4":

        # use mean of last 4 layer of decoder as regression features
        embeds = torch.concat([
            decoder_embeds[-4], decoder_embeds[-3],
            decoder_embeds[-2], decoder_embeds[-1]
        ], dim=1).mean(dim=1)

        if encoder_embeds:
            e_embeds = torch.concat([
                encoder_embeds[-4], encoder_embeds[-3],
                encoder_embeds[-2], encoder_embeds[-1]
            ], dim=1).mean(dim=1)
            embeds = torch.concat([embeds, e_embeds], dim=1)

    else:
        raise NotImplemented

    return embeds


def compute_pearson_torch(pred, target):
    
    """
        Compute pearson correlation coefficient

        Args:
            pred,           torch.Tensor, prediction
            target,         torch.Tensor, target
    """

    pearson = list()
    for p, t in zip(pred.T, target.T):
        pearson.append(torch.corrcoef(torch.stack((p, t)))[0][1])

    return torch.stack(pearson)


def compute_perason_numpy(pred, target):

    corrcoef = list()
    for pred, target in zip(pred.T, target.T):

        s, _ = pearsonr(x=pred, y=target)
        corrcoef.append(s)

    return np.array(corrcoef)


def build_train_script(args):

    command = "python3 task_train.py"

    for k, v in vars(args).items():

        if v:
            command += " \ \n\t--{key} {value}".format(key=k, value=v)

    return command


def my_collate_fn(x):

    imgs = list()
    ids = list()
    for img, _, id in x:
        imgs.append(img)
        ids.append(id)

    return imgs, ids


def my_training_collate_fn(x):

    imgs = list()
    fmris = list()
    captions = list()
    for i, f, c in x:
        imgs.append(i)
        fmris.append(f)
        captions.append(c)
    
    return imgs, fmris, captions


if __name__ == "__main__":

    a = torch.rand(16, 128)
    b = torch.rand(16, 128)

    p = compute_pearson_torch(a, b)
    print(p)
    print(p.shape)

