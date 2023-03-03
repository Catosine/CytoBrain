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
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.dataset import Algonauts2023Raw
from src.utils import extract_argParse, extract_initialize, build_model, build_transform

def main(args):
    # initialize
    args = extract_initialize(args)
    print("Initialization ready.")
    print(args)

    # load dataset
    dataset = Algonauts2023Raw(osp.join(args.data, args.subject), train=True, hemisphere=args.hemisphere, transform=build_transform(args.subject, False))
    print("#Total: {}".format(len(dataset)))

    # setup model
    model = build_model(args.model, 100, args.pretrained_weight)
    model = create_feature_extractor(model, return_nodes=args.layers)


    print("Pretrained model loaded from {}".format(args.pretrained_weight))
    model.to(args.device)
    print("Model initialized. Loaded to <{}> device.".format(args.device))

    # get inferred results
    print("Start feature extraction")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    output = dict()
    for l in args.layers:
        output[l] = list()

    for img, _ in tqdm(dataloader):
        model.eval()

        img = img.to(args.device)

        with torch.no_grad():
            pred = model(img)

            for k in pred.keys():
                output[k].append(pred[k].detach().cpu())

    filename_template = "{subj}_{hemisphere}_{pretrained}".format(
        subj=args.subject, hemisphere=args.hemisphere,  pretrained=args.pretrained_weight.split("/")[-1].split(".")[0])
    args.save_path = osp.join(args.save_path, filename_template)

    print("Saving features")
    for k, v in tqdm(output.items()):

        v = torch.concat(v).numpy()
        np.save(args.save_path+"{}.npy".format(k), v.astype(np.float32))

        print("Feature from {} is saved. Shape: {}".format(k, v.shape))

if __name__ == "__main__":
    
    args = extract_argParse()
    main(args)
