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


def __base_argParse(parser):

    parser.add_argument("--data", type=str, default="/Users/cytosine/Documents/Algonauts2023/data.nosync",
                        help="Path to Algonauts2023 dataset")
    parser.add_argument("--subject", type=str, default="subj01",
                        choices=["subj01", "subj02", "subj03", "subj04",
                                 "subj05", "subj06", "subj07", "subj08"],
                        help="Used to select which subject to learn")
    parser.add_argument("--caption", type=str, default="./nsd_captions.json", help="Path to captions")
    parser.add_argument("--use_pil", action="store_true", help="Use PIL when loading data")
    parser.add_argument("--hemisphere", type=str, choices=["L", "R"], default="L",
                        help="Select which half of the brain to model")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet152"], help="Select different models")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to pretrained weight")
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

    parser.add_argument("--early_stopping", type=int, default=0, help="Apply early stopping that stops training after N-nonimproving evaluations")
    parser.add_argument("--eval_per_step", type=int, default=100, help="Run the devset evaluation every K step.")
    parser.add_argument("--report_step", type=int, default=10,
                        help="Num of report steps for logging")

    parser.add_argument("--save_path", type=str, default="./logs",
                        help="Path to save training logs and models")

    return parser


def __infer_argParse(parser):

    parser.add_argument("--save_path", type=str, default="./prediction",
                        help="Path to save training logs and models")
    parser.add_argument("--output_size", type=int, default=2048,
                        help="Output size of pretrained model")

    return parser


def __feat_extract_argPargs(parser):

    parser.add_argument("--save_path", type=str, default="./features",
                        help="Path to save training logs and models")
    parser.add_argument("--layers", type=str, nargs="+",
                        help="The layer which features are extracted from")
    parser.add_argument("--train", action="store_true", help="if extract data from train set")

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


def extract_argParse():

    parser = argparse.ArgumentParser()

    parser = __base_argParse(parser)
    parser = __feat_extract_argPargs(parser)

    return parser.parse_args()
