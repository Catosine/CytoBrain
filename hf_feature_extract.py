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
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm
import numpy as np
from src.utils import my_collate_fn
from src.dataset import Algonauts2023Raw


def parseArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, default="../../backbone.nosync/vit-gpt2-image-captioning",
                        help="Pretrained weights for Huggingface models")
    parser.add_argument("--data", type=str,
                        default="../../data.nosync", help="Path to images")
    parser.add_argument("--subject", type=str, choices=[
                        "subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"], default="subj01")
    parser.add_argument("--feature_type", type=str,
                        choices=["encoder", "decoder"], default="encoder")
    parser.add_argument("--train", action="store_true",
                        help="Extract train features")
    parser.add_argument("--save_path", type=str, help="Save path")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    # initialize model
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_weights)
    feature_extractor = ViTImageProcessor.from_pretrained(
        args.pretrained_weights)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # construct paths
    path = os.path.join(args.data, args.subject)
    args.save_path = os.path.join(args.save_path, args.pretrained_weights.split(
        "/")[-1], "{}-raw".format(args.feature_type))

    # load data
    dset = Algonauts2023Raw(path, train=args.train,
                            return_img_ids=True, return_pil=True)

    # generate parameters
    gen_kwargs = {"max_length": 16, "num_beams": 4,
                  "return_dict_in_generate": True, "output_hidden_states": True}

    ids = list()
    features = list()
    model.eval()
    print("Start to extract features...")
    for img, id in tqdm(DataLoader(dset, batch_size=16, num_workers=4, collate_fn=my_collate_fn)):

        pixel_values = feature_extractor(
            images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        if args.feature_type == "encoder":
            with torch.no_grad():
                feats = model.encoder(
                    pixel_values, output_hidden_states=True).hidden_states
        else:
            feats = model.generate(
                pixel_values, **gen_kwargs).encoder_hidden_states

        feats = [x for x in feats]
        feats = torch.stack(feats[-4:]).cpu()

        for i in range(len(id)):
            hs = feats[:, i]
            hs = hs.numpy().astype(np.float32)

            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)

            np.save(os.path.join(args.save_path,
                    id[i].split(".")[0]+".npy"), hs)

    print("Done")
