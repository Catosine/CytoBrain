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

from src.dataset import Algonauts2023Raw
from src.utils import infer_argParse, inference_initialize, build_model, build_transform
from src.trainer import NNTrainer

def main(args):
    # initialize
    args = inference_initialize(args)
    print("Initialization ready.")
    print(args)

    # load dataset
    dataset = Algonauts2023Raw(osp.join(args.data, args.subject), train=False, transform=build_transform(args.subject, False))
    print("#Total: {}".format(len(dataset)))
    print("Predicted FMRI size: {}".format(args.output_size))

    # setup model
    model = build_model(args.model, args.output_size, args.pretrained_weight)
    print("Pretrained model loaded from {}".format(args.pretrained_weight))
    model.to(args.device)
    print("Model initialized. Loaded to <{}> device.".format(args.device))

    # get inferred results
    print("Start inferencing")
    output = NNTrainer.infer(model, dataset, args.batch_size)

    np.save(osp.join(args.save_path, args.hemishpere.lower() + "h_pred_test.npy"), output.astype(np.float32))
    print("prediction saved.")

if __name__ == "__main__":
    
    args = infer_argParse()
    main(args)
