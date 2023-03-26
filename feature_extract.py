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
import os
import os.path as osp

import cv2
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from tqdm import tqdm

from src.utils import build_model
from src.initialize import extract_initialize
from src.arg_parse import extract_argParse
from src.constant import TRAIN_MEAN, TRAIN_STD, TEST_MEAN, TEST_STD


def __build_transform(subj, train=False):

    mean = TRAIN_MEAN[subj] if train else TEST_MEAN[subj]
    std = TRAIN_STD[subj] if train else TEST_STD[subj]

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def main(args):
    # initialize
    args = extract_initialize(args)
    print("Initialization ready.")
    print(args)

    # init transform
    tf = __build_transform(args.subject, args.train)

    # setup model
    model = build_model(args.model, 100, args.pretrained_weight)
    model = create_feature_extractor(model, return_nodes=args.layers)

    print("Pretrained model loaded from {}".format(args.pretrained_weight))
    model.to(args.device)
    print("Model initialized. Loaded to <{}> device.".format(args.device))

    # get inferred results
    print("Start feature extraction")
    args.data = osp.join(args.data, args.subject,
                         "training_split/training_images" if args.train else "test_split/test_images")
    for img in tqdm(os.listdir(args.data)):

        # load img
        img_data = cv2.imread(osp.join(args.data, img)
                              ).astype(np.float32)
        
        # convert BGR to RGB
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        img_data = tf(img_data)
        img_data.to(args.device)

        # extract
        model.eval()
        with torch.no_grad():
            pred = model(img_data.unsqueeze(0))

            for k, v in pred.items():

                if not osp.isdir(osp.join(args.save_path, k)):
                    os.makedirs(osp.join(args.save_path, k))

                np.save(osp.join(args.save_path, k, "{}.npy".format(
                    img.split(".")[0])), v.detach().cpu().numpy().astype(np.float32))

    print("Done.")


if __name__ == "__main__":

    args = extract_argParse()
    main(args)
