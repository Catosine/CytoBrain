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

from .utils import my_training_collate_fn


class NNTrainer:
    """
        NN task trainer. Including train/validate/infer
    """

    def __init__(self, model, feature_extractor, tokenizer, criterion, scoring_fn, optimizer, scheduler, summarywriter, logging, save_path):
        """
            Initialize the trainer
        """

        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.scoring_fn = scoring_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summarywriter = summarywriter
        self.logging = logging
        self.save_path = save_path

    def eval(self, val_loader):

        # validating
        dev_pred = list()
        dev_fmri = list()
        
        for img, fmri, caption in tqdm(val_loader):

            self.model.eval()

            # load data to device
            device = next(self.model.parameters()).device
            fmri = torch.FloatTensor(np.stack(fmri)).to(device)

            pixel_values = self.feature_extractor(
                img, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device)

            labels = self.tokenizer(
                caption, return_tensors="pt", padding=True).input_ids.to(device)

            with torch.no_grad():
                pred = self.model(pixel_values=pixel_values,
                                labels=labels, output_hidden_states=True)

            dev_pred.append(pred)
            dev_fmri.append(fmri)

        dev_pred = torch.concat(dev_pred)
        dev_fmri = torch.concat(dev_fmri)

        # compute loss and score
        loss = self.criterion(dev_pred, dev_fmri)
        score = self.scoring_fn(dev_pred, dev_fmri)

        return score, loss

    def run(self, train_loader, val_loader, epoch=100, report_step=20, early_stopping=0):
        """
            Standard Training and validation

            Args:
                train_loader,       torch.utils.dataset.data.Dataset, the train set
                val_loader,         torch.utils.dataset.data.Dataset, the validation set
                epoch,              int, maximized training epoch
                report_step,        int, report step
                early_stopping,     int, early stopped after K non-improving evaluation steps
        """

        train_step = 0
        best_score = 0
        stopping_counter = 0
        self.logging.info("Start training")

        for e in range(epoch):

            self.logging.info(
                "==============<Epoch#{}>==============".format(e))

            # training
            for img, fmri, caption in tqdm(train_loader):

                _, score, loss = self.__batch_train(img, fmri, caption)
                
                self.summarywriter.add_scalar(
                    "train/batch/loss", loss, train_step)
                self.summarywriter.add_scalar(
                    "train/batch/avg. score", score.mean(), train_step)
                self.summarywriter.add_scalar(
                    "train/batch/median score", score.median(), train_step)

                train_step += 1

                if train_step % report_step == 0:
                    self.logging.info(
                        "[Training @ Step#{}]\tAvg. Loss: {:.3f}\tAvg. Score: {:.3f}\tMedian Score: {:.3f}".format(train_step, loss, score.mean(), score.median()))

            # evaluate
            dev_score, dev_loss = self.eval(val_loader)
            self.logging.info("[Validating @ Epoch#{}]\tLoss: {:.3f}\tAvg. Score: {:.3f}\tMedian Score: {:.3f}".format(
                train_step, dev_loss, dev_score.mean(), dev_score.median()))
            
            self.summarywriter.add_scalar("dev/loss", dev_loss, train_step)
            self.summarywriter.add_scalar("dev/avg. score", dev_score.mean(), train_step)
            self.summarywriter.add_scalar("dev/median score", dev_score.median(), train_step)

            if dev_score.median() > best_score:
                # find new best evaluation
                stopping_counter = 0
                best_score = dev_score.median()
                self.logging.info(
                    "New best model found @ Epoch#{}.".format(e))
                torch.save(self.model.state_dict(), osp.join(
                    self.save_path, "checkpoint_best.pt"))
            elif early_stopping and stopping_counter < early_stopping:
                # update early stopping counter
                stopping_counter += 1

                if stopping_counter >= early_stopping:
                    # trigger early stopping
                    self.logging.info(
                        "Early stopping triggered. The training process is stopped.")
                    return
            else:
                # no early stopping involved. The training will continue
                continue

            torch.save(self.model.state_dict(), osp.join(
                self.save_path, "checkpoint_epoch_{}.pt".format(e)))

        self.logging.info("Done.")

    def __batch_train(self, img, fmri, caption):
        
        self.model.train()

        # load data to device
        device = next(self.model.parameters()).device

        fmri = torch.FloatTensor(np.stack(fmri)).to(device)

        pixel_values = self.feature_extractor(
            img, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device)

        labels = self.tokenizer(
            caption, return_tensors="pt", padding=True).input_ids.to(device)

        pred = self.model(pixel_values=pixel_values,
                            labels=labels, output_hidden_states=True)

        # compute loss and score
        loss = self.criterion(pred, fmri)
        score = self.scoring_fn(pred, fmri)

        # backprop
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # clean-up
        self.model.zero_grad()

        return pred.detach().cpu(), score.detach().cpu(), loss.detach().cpu()


    @staticmethod
    def infer(model, feature_extractor, tokenizer, dataset, batch_size=64, num_workers=4):
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
            dataset, batch_size=batch_size, shuffle=False, collate_fn=my_training_collate_fn, num_workers=num_workers)

        results = list()

        device = next(model.parameters()).device

        for img, _, caption in tqdm(dataloader):

            model.eval()

            pixel_values = feature_extractor(img, return_tensors="pt")[
                "pixel_values"]
            pixel_values = pixel_values.to(device)

            labels = tokenizer(
                caption, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                pred = model(pixel_values=pixel_values,
                             labels=labels, output_hidden_states=True)
                results.append(pred.detach())

        return torch.stack(results).numpy()
