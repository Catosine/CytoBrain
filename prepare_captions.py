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
import json
import argparse
import os.path as osp

import pandas as pd


def parseArg():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd_info", type=str, default="nsd_stim_info_merged.csv",
                        help="the location of nsd_stim_info_merged.csv file. This file may be accessed from NSD website: https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data")
    parser.add_argument("--coco_train_captions", type=str, default="captions_train2017.json",
                        help="the location of captions_train2017.json file. This file may be accessed from COCO website: https://cocodataset.org")
    parser.add_argument("--coco_val_captions", type=str, default="captions_val2017.json",
                        help="the location of captions_val2017.json file. This file may be accessed from COCO website: https://cocodataset.org")
    parser.add_argument("--save", type=str, default=".", help="Path to save the output captions.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArg()

    nsd_info = pd.read_csv(args.nsd_info)
    nsd_info = pd.DataFrame(nsd_info, columns=["nsdId","cocoSplit","cocoId"])

    with open(args.coco_train_captions, "r", encoding="UTF-8") as f:
        train_cap = json.load(f)["annotations"]

    with open(args.coco_val_captions, "r", encoding="UTF-8") as f:
        val_cap = json.load(f)["annotations"]

    coco_captions = dict()
    for x in train_cap:
        coco_id = "train-{}".format(x["image_id"])
        cap = x["caption"]
        if coco_id in coco_captions:
            coco_captions[coco_id].append(cap)
        else:
            coco_captions[coco_id] = [cap]

    for x in val_cap:
        coco_id = "val-{}".format(x["image_id"])
        cap = x["caption"]
        if coco_id in coco_captions:
            coco_captions[coco_id].append(cap)
        else:
            coco_captions[coco_id] = [cap]

    nsd_captions = dict()
    for _, (nsdId, cocoSplit, cocoId) in nsd_info.iterrows():
        nsd_captions[nsdId] = coco_captions["{}-{}".format(cocoSplit[:-4], cocoId)]

    with open(osp.join(args.save, "nsd_captions.json"), "w", encoding="UTF-8") as f:
        json.dump(nsd_captions, f)
