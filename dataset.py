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
from tqdm import tqdm


class Algonauts2023Raw(Dataset):
    """
        Load original data for Algonauts2023 dataset
    """

    def __init__(self, data_path: str, hemisphere: str = "L", train: bool = True):

        """
            Initialize a torch.utils.data.Dataset object for algonauts2023 dataset

            Args:
                data_path,              str, path to the algonauts2023 dataset which contains only ONE subject
                area,                   str, select which hemisphere of the brain to be modeled
                                            can ONLY select "L" or "R"
                train,                  bool, training data will be loaded if True. Test data otherwise.
        """

        # collect data paths
        path_struct = osp.join(data_path, "{}_split")
        self.dataset = list()

        if train:
            shared_path = osp.join(path_struct.format("training"), "training_{}")
            if hemisphere == "L":
                fmri = np.load(osp.join(shared_path.format("fmri"), "lh_training_fmri.npy"))
            elif hemisphere == "R":
                fmri = np.load(osp.join(shared_path.format("fmri"), "rh_training_fmri.npy"))

            image_path = shared_path.format("images")
            for img in tqdm(os.listdir(image_path)):
                idx = int(re.findall("\d{4}", img)[0]) - 1
                self.dataset.append([cv2.imread(osp.join(image_path, img)), fmri[idx]])
        else:
            image_path = osp.join(path_struct.format("test"), "test_images")
            for img in tqdm(os.listdir(image_path)):
                self.dataset.append([cv2.imread(osp.join(image_path, img)), 0])

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

        sample = self.dataset[index]

        return sample[0], sample[1]


if __name__ == "__main__":
    dataset = Algonauts2023Raw("/Users/cytosine/Documents/Algonauts2023/data/subj01", train=True)
    print("Successfully loaded")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    print(len(dataloader))
