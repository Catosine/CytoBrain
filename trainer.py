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


class NNTrainer:
    """
        NN task trainer. Including train/validate/infer
    """

    def __init__(self, model, criterion, scoring_fn, optimizer, scheduler, summarywriter, logging, save_path):
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
        self.save_path = save_path

    def run(self, train_set, val_set, epoch=100, batch_size=64, report_step=20, num_workers=4):
        """
            Standard Training and validation

            Args:
                train_set,          torch.utils.dataset.data.Dataset, the train set
                val_set,            torch.utils.dataset.data.Dataset, the validation set
                epoch,              int, maximized training epoch
                batch_size,         int, batch size
                report_step,        int, report step
                num_workers,        int, num workers for initializing dataloader
        """

        # initializing dataloaders
        train_loader = data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = data.DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        train_step = 0
        dev_step = 0
        best_score = -1
        self.logging.info("Start training")

        for e in tqdm(range(epoch)):

            self.logging.info(
                "==============<Epoch#{}>==============".format(e))

            # training
            for img, fmri in train_loader:

                _, score, loss = self.__batch_train(img, fmri)

                self.summarywriter.add_scalar("train/loss", loss, train_step)
                self.summarywriter.add_scalar("train/score", score, train_step)

                train_step += 1

                if train_step % 100 == 0:
                    self.logging.info(
                        "[Training @ Step#{}]\tAvg. Loss: {:.3f}\tAvg. Score: {:.3f}".format(train_step, loss, score))

            torch.save(self.model.state_dict(), osp.join(
                self.save_path, "checkpoint_epoch_{}.pt".format(e)))

            # validating
            dev_score = list()
            dev_loss = list()
            for img, fmri in val_loader:

                _, score, loss = self.__batch_val(img, fmri)
                dev_score.append(score)
                dev_loss.append(loss)

                self.summarywriter.add_scalar("dev/loss", loss, dev_step)
                self.summarywriter.add_scalar("dev/score", score, dev_step)

                dev_step += 1

            dev_score = np.array(dev_score).mean()
            dev_loss = np.array(dev_loss).mean()

            self.logging.info(
                "[Validating @ Epoch#{}]\tAvg. Loss: {:.3f}\tAvg. Score: {:.3f}".format(e, dev_loss, dev_score))

            if dev_score > best_score:
                best_score = dev_score
                self.logging.info("New best model found @ Epoch#{}.".format(e))
                torch.save(self.model.state_dict(), osp.join(
                    self.save_path, "checkpoint_best.pt"))

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

    @staticmethod
    def infer(model, dataset, batch_size=64, num_workers=4):
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
        dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        results = list()

        for img, _ in tqdm(dataloader):
            model.eval()

            with torch.no_grad():
                pred = model(img)
                results.append(pred.detach())

        return torch.stack(results).numpy()