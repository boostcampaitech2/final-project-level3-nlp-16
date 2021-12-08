"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from src.utils.torch_utils import save_model


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        num_classes: int = 15,
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            num_classes: number of classes
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        self.num_classes = num_classes

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_top3_acc = -1.0
        best_test_f1 = -1.0
        best_test_top3_f1 = -1.0
        label_list = [i for i in range(self.num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, top3_correct, total = 0.0, 0, 0, 0
            preds, top3_preds, gt = [], [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                for key in data.keys():
                    data[key] = data[key].to(self.device)
                labels = labels.to(self.device)
                # data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                    
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                _, top3_pred = torch.topk(outputs, 3, largest=True, sorted=True)

                # True (#0) if `expected label` in top3_pred, False (0) if not
                a = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - top3_pred), dim=1).to(torch.bool)
                a = a.to(torch.int8)
                top3_pred = a * labels + (1-a) * top3_pred[:,0]

                total += labels.size(0)
                correct += (pred == labels).sum().item()
                top3_correct += (top3_pred == labels).sum().item()

                preds += pred.to("cpu").tolist()
                top3_preds += top3_pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                accuracy = correct / total
                top3_accuracy = top3_correct / total
                f1 = f1_score(y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0)
                top3_f1 = f1_score(y_true=gt, y_pred=top3_preds, labels=label_list, average="macro", zero_division=0)

                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {accuracy * 100:.2f}% "
                    f"F1(macro): {f1:.2f}"
                )

                if batch%100 == 0:
                    wandb.log({
                        "train/loss": running_loss/(batch+1),
                        "train/accuracy": accuracy,
                        "train/top3_accuracy": top3_accuracy,
                        "train/f1": f1,
                        "train/top3_f1": top3_f1,
                    })

            pbar.close()

            _, test_f1, test_acc, test_top3_f1, test_top3_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                "validation/accuracy": test_acc,
                "validation/top3_accuracy": test_top3_acc,
                "validation/f1": test_f1,
                "validation/top3_f1": test_top3_f1,
            })

            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_top3_acc = test_top3_acc
            best_test_f1 = test_f1
            best_test_top3_f1 = test_top3_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}, top3_f1: {best_test_top3_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        running_loss, correct, top3_correct, total = 0.0, 0, 0, 0
        preds, top3_preds, gt = [], [], []

        label_list = [i for i in range(self.num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            for item in data:
                data[item] = data[item].to(self.device)
            labels = labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            _, top3_pred = torch.topk(outputs, 3, largest=True, sorted=True)

            # True (#0) if `expected label` in top3_pred, False (0) if not
            a = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - top3_pred), dim=1).to(torch.bool)
            a = a.to(torch.int8)
            top3_pred = a * labels + (1-a) * top3_pred[:,0]
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            top3_correct += (top3_pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            top3_preds += top3_pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()

            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        top3_accuracy = top3_correct / total
        f1 = f1_score(y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0)
        top3_f1 = f1_score(y_true=gt, y_pred=top3_preds, labels=label_list, average="macro", zero_division=0)

        return loss, f1, accuracy, top3_f1, top3_accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
