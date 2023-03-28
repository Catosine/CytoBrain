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
import re

import cv2
import numpy as np
from torch.utils.data import Dataset


class Algonauts2023Raw(Dataset):
    """
        Load original data for Algonauts2023 dataset
    """

    def __init__(self, data_path: str, hemisphere: str = "L", transform=None, train: bool = True):
        """
            Initialize a torch.utils.data.Dataset object for algonauts2023 dataset

            Args:
                data_path,              str, path to the algonauts2023 dataset which contains only ONE subject
                hemisphere,             str, select which hemisphere of the brain to be modeled
                                            can ONLY select "L" or "R"
                                            and ONLY applicable when train is TRUE
                transform,              torchvision.transform methods, apply normalization to the dataset
                train,                  bool, training data will be loaded if True. Test data otherwise.
        """

        # collect data paths
        path_struct = osp.join(data_path, "{}_split")
        self.dataset = list()
        self.transform = transform
        self.train = train

        if train:
            shared_path = osp.join(
                path_struct.format("training"), "training_{}")
            if hemisphere == "L":
                self.fmri = np.load(osp.join(shared_path.format(
                    "fmri"), "lh_training_fmri.npy"))
            elif hemisphere == "R":
                self.fmri = np.load(osp.join(shared_path.format(
                    "fmri"), "rh_training_fmri.npy"))

            self.feature_path = shared_path.format("images")

        else:
            self.feature_path = osp.join(
                path_struct.format("test"), "test_images")

        self.dataset = list(os.listdir(self.feature_path))

        # sorted in ascending order if not train set
        if not train:
            self.dataset = sorted(self.dataset, key=lambda x: int(
                re.findall("\d{4}", x)[0]) - 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
            Load designated sample

            Arg:
                index,          int, sample id

            Returns:
                image,          np.ndarray, the 3d numpy array of the image used to retrive fmri data
                fmri,           np.ndarray, the hemisphere FMRI data generated by the image
        """

        feat_file = self.dataset[index]
        feat_idx = int(re.findall("\d{4}", feat_file)[0]) - 1

        img = cv2.imread(osp.join(self.feature_path, feat_file)
                         ).astype(np.float32)

        # convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, self.fmri[feat_idx] if self.train else 0


class Algonauts2023Feature(Dataset):

    """
        Load processed features of Algonauts2023 dataset
    """

    def __init__(self, data_path: str, extractor: str, layer: str,  hemisphere: str = "L", train: bool = True):
        """
            Initialize a torch.utils.data.Dataset object for algonauts2023 dataset

            Args:
                data_path,              str, path to the algonauts2023 dataset which contains only ONE subject
                extractor,              str, name of extractor of the feature
                layer,                  str, name of layer of extractor of the feauture
                hemisphere,             str, select which hemisphere of the brain to be modeled
                                            can ONLY select "L" or "R"
                                            and ONLY applicable when train is TRUE
                train,                  bool, training data will be loaded if True. Test data otherwise.
        """

        # collect data paths
        path_struct = osp.join(data_path, "{}_split")
        self.dataset = list()
        self.train = train

        if train:
            shared_path = osp.join(
                path_struct.format("training"), "training_{}")
            fmri_path = shared_path.format("fmri")
            if hemisphere == "L":
                self.fmri = np.load(
                    osp.join(fmri_path, "lh_training_fmri.npy"))
            elif hemisphere == "R":
                self.fmri = np.load(
                    osp.join(fmri_path, "rh_training_fmri.npy"))

            self.feature_path = shared_path.format("features")

        else:
            self.feature_path = osp.join(
                path_struct.format("test"), "test_features")

        self.feature_path = osp.join(self.feature_path, extractor, layer)

        self.dataset = list(os.listdir(self.feature_path))

        # sorted in ascending order if not train set
        if not train:
            self.dataset = sorted(self.dataset, key=lambda x: int(re.findall("\d{4}", x)[0]) - 1)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
            Load designated sample

            Arg:
                index,          int, sample id

            Returns:
                feature,        np.ndarray, the 1d numpy array of image features
                fmri,           np.ndarray, the hemisphere FMRI data generated by the image
        """

        feat_file = self.dataset[index]
        feat_idx = int(re.findall("\d{4}", feat_file)[0]) - 1

        # load
        feature = np.load(osp.join(self.feature_path, feat_file)
                          ).astype(np.float32)

        return feature, self.fmri[feat_idx] if self.train else 0


def get_features(data_path: str, extractor: str, layer: list, train: bool = True):

    path_struct = osp.join(data_path, "{}_split")

    if train:
        shared_path = osp.join(
            path_struct.format("training"), "training_{}")
        fmri_path = shared_path.format("fmri")
        print("Using fMRI from: {}".format(fmri_path))

        lfmri = np.load(osp.join(fmri_path, "lh_training_fmri.npy"))
        rfmri = np.load(osp.join(fmri_path, "rh_training_fmri.npy"))

        feature_path = shared_path.format("features")

    else:
        feature_path = osp.join(
            path_struct.format("test"), "test_features")

    print("Using data from: {}".format(feature_path))

    # get all file names
    img_files = os.listdir(osp.join(feature_path, extractor, layer[0]))

    # sorted in ascending order if not train
    if not train:
        img_files = sorted(img_files, key= lambda x:int(re.findall("\d{4}", x)[0]) - 1)

    features = list()
    lfmris = list()
    rfmris = list()
    for img in img_files:

        # load features from all layers and concat them together
        feat = np.hstack([np.load(osp.join(feature_path, extractor, l, img)
                                  ).astype(np.float32) for l in layer])
        features.append(feat)

        # get fmri id
        feat_idx = int(re.findall("\d{4}", img)[0]) - 1
        lfmris.append(lfmri[feat_idx] if train else np.ones(1))
        rfmris.append(rfmri[feat_idx] if train else np.ones(1))

    return np.vstack(features), lfmris, rfmris


if __name__ == "__main__":

    f, l, r = get_features("../../data.nosync/subj01", "resnet50-imagenet1k-v2", ["avgpool"], True)
    print(f.shape)
    print(l.shape)
    print(r.shape)
