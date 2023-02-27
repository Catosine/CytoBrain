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
import torch.utils.data as data
from tqdm import tqdm


class Trainer:
    """
        Trainer
    """

    def __init__(self, model, criterion, scoring_fn, optimizer, scheduler, summarywriter, logging):

        """
            Initialize the trainer
        """

        self.model = model
        self.criterion = criterion
        self.scoring_fn = scoring_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summarywriter = summarywriter
        self.logging = logging

    def kfold_train_and_val(self, dataset, epoch=100, batch_size=64, k_fold=5, report_step=20, save_path=None):

        """
            K-Fold Training and validation

            Args:
                dataset,            torch.utils.dataset.data.Dataset, the dataset
                summarywriter,      torch.utils.tensorboard.SummaryWriter, tensorboard summary writer
                logging,            logger
                epoch,              int, maximized training epoch
                batch_size,         int, batch size
                k_fold,             int, fold number. For every epoch, the dataset will be split into K portion 
                                        and use K-1 of them to train and the rest for validation.
                save_path,          str, save path for best model & per epoch model
        """

        self.logging.info("Start {}-Fold training.".format(k_fold))

        dset_len = len(self.dataset)
        fraction = 1 / k_fold
        segment = int(dset_len * fraction)

        self.logging.info("Total #Samples: {}".format(dset_len))
        self.logging.info("Per Fold #Samples: {}".format(segment))

        for e in tqdm(range(epoch)):
            self.logging.info("==============<Epoch#{}>==============".format(e))

            i = e % k_fold

            trll = 0
            trlr = i * segment
            vall = trlr
            valr = (i + 1) * segment
            trrl = valr
            trrr = dset_len

            train_left_idx = list(range(trll, trlr))
            train_right_idx = list(range(trrl, trrr))

            train_idx = train_left_idx + train_right_idx
            val_idx = list(range(vall, valr))

            # train
            avg_train_loss, avg_train_score = self.__epoch_train(e, data.dataset.Subset(dataset, train_idx), batch_size,
                                                                 report_step=report_step)
            self.logging.info("Average Training Loss @Epoch#{}: {}".format(e, avg_train_loss))
            self.logging.info("Average Training Score @Epoch#{}: {}".format(e, avg_train_score))

            # validate
            avg_val_loss, avg_val_score = self.__epoch_val(e, data.dataset.Subset(dataset, val_idx), batch_size)
            self.logging.info("Average Validation Loss @Epoch#{}: {}".format(e, avg_val_loss))
            self.logging.info("Average Validation Score @Epoch#{}: {}".format(e, avg_val_score))

            torch.save(self.model.state_dict(), osp.join(save_path, "checkpoint_epoch_{}.pth".format(e)))

        self.logging.info("Done.")

    def __batch_train(self, img, fmri):

        self.model.train()

        pred = self.model(img)
        loss = self.criterion(pred, fmri)
        score = self.scoring_fn(pred, fmri)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        self.model.zero_grad()

        return pred.detach(), score.mean().detach(), loss.detach()

    def __batch_val(self, img, fmri):

        self.model.eval()

        with torch.no_grad():
            pred = self.model(img)
            loss = self.criterion(pred, fmri)
            score = self.scoring_fn(pred, fmri)

        return pred.detach(), score.mean().detach(), loss.detach()

    def __epoch_train(self, epoch, dataset, batch_size=64, num_workers=4, report_step=20):

        """
            Per epoch train method

            Args:
                epoch,              int, epoch number, only used for calculating report step
                dataset,            torch.utils.data.Dataset object
                batch_size,         int, batch size
                num_workers,        int, num of worker for dataloader initialization
                report_step,        int, logging report step
            
            Returns:
                avg_batch_loss,     np.float32, the averaged batch loss
                avg_batch_score,    np.float32, the averaged batch score
        """

        # init dataloader
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        per_batch_loss = list()
        per_batch_score = list()

        step = len(dataloader) * epoch

        for i, (img, fmri) in enumerate(dataloader):

            step += i

            _, score, loss = self.__batch_train(img, fmri)
            per_batch_loss.append(loss)
            per_batch_score.append(score)

            self.summarywriter.add_scalar("train/loss", loss, step)
            self.summarywriter.add_scalar("train/score", score.mean(), step)

            if (i + 1) % report_step == 0:
                self.logging.info(
                    "[Step #{}] Train Loss: {}\t Avg. Prediction Spearman Corrcoef.: {}".format(i, loss, score))

        return np.array(per_batch_loss).mean(), np.array(per_batch_score).mean()

    def __epoch_val(self, epoch, dataset, batch_size=64, num_workers=4):

        """
            Per epoch val method

            Args:
                epoch,              int, epoch number, only used for calculating report step
                dataset,            torch.utils.data.Dataset object
                batch_size,         int, batch size
                num_workers,        int, num of worker for dataloader initialization
            
            Returns:
                avg_batch_loss,     np.float32, the averaged batch loss
                avg_batch_score,    np.float32, the averaged batch score
        """

        # init dataloader
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        per_batch_loss = list()
        per_batch_score = list()

        step = len(dataloader) * epoch

        for i, (img, fmri) in enumerate(dataloader):
            step += i

            _, score, loss = self.__batch_val(img, fmri)
            per_batch_loss.append(loss)
            per_batch_score.per_batch_score.append(score)

            self.summarywriter.add_scalar("val/loss", loss, step)
            self.summarywriter.add_scalar("val/score", score, step)

        return np.array(per_batch_loss).mean(), np.array(per_batch_score).mean()

    def infer(self, dataset, batch_size=64, num_workers=4):

        """
            Inferring on given dataset

            Args:
                dataset,            torch.utils.data.Dataset object
                batch_size,         int, batch size
                num_workers,        int, num of worker for dataloader initialization

            Returns:
                results,            np.ndarray, the inferred fmri result

        """

        # init dataloader
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        results = list()

        for img, _ in tqdm(dataloader):
            self.model.eval()

            with torch.no_grad():
                pred = self.model(img)
                results.append(pred.detach())

        return torch.stack(results).numpy()
