from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb


_dict_num_to_category = {
    0:"노트북",
    1:"키보드/마우스",
    2:"데스크탑",
    3:"모니터",
    4:"DSLR",
    5:"필름카메라",
    6:"TV",
    7:"대형가전",
    8:"사무기기",
    9:"스피커",
    10:"이어폰/헤드폰",
    11:"스마트폰",
    12:"일반폰",
    13:"태블릿",
    14:"웨어러블",
}

class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        criterion: Tuple[nn.Module, nn.Module],
        optimizer: Tuple[optim.Optimizer, optim.Optimizer],
        scheduler,
        device: torch.device = "cpu",
        num_classes: int = 15,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            num_classes: number of classes
        """

        self.model = model
        self.model_path = model_path
        self.img_criterion, self.txt_criterion = criterion
        self.img_optimizer, self.txt_optimizer = optimizer
        self.img_scheduler, self.txt_scheduler = scheduler
        self.device = device
        self.num_classes = num_classes

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float, float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns data
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            accuracay and f1 score
        """
        best_val_acc = -1.0
        best_val_top3_acc = -1.0
        best_val_f1 = -1.0
        best_val_top3_f1 = -1.0
        label_list = [i for i in range(self.num_classes)]

        for epoch in range(n_epoch):
            total = 0
            gt = []
            img_running_loss, img_correct, img_top3_correct, img_preds, img_top3_preds = 0, 0, 0, [], []
            txt_running_loss, txt_correct, txt_top3_correct, txt_preds, txt_top3_preds = 0, 0, 0, [], []
            votting_correct, votting_top3_correct, votting_preds, votting_top3_preds = 0, 0, [], []

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, data in pbar:
                labels = data.pop("category").to(self.device)
                for key in data.keys():
                    data[key] = data[key].to(self.device)

                img_outputs, txt_outputs = self.model(data)
                img_outputs, txt_outputs = torch.squeeze(img_outputs), torch.squeeze(txt_outputs)
                img_loss, txt_loss = self.img_criterion(img_outputs, labels), self.txt_criterion(txt_outputs, labels)

                votting_outputs = torch.div(img_outputs + txt_outputs, 2)

                self.img_optimizer.zero_grad()
                self.txt_optimizer.zero_grad()

                img_loss.backward()
                txt_loss.backward()

                self.img_scheduler.step()
                self.txt_scheduler.step()

                self.img_optimizer.step()
                self.txt_optimizer.step()

                _, img_pred = torch.max(img_outputs, 1)
                _, txt_pred = torch.max(txt_outputs, 1)
                _, votting_pred = torch.max(votting_outputs, 1)

                _, img_top3_pred = torch.topk(img_outputs, 3, largest=True, sorted=True)
                a = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - img_top3_pred), dim=1).to(torch.bool)
                a = a.to(torch.int8)
                img_top3_pred = a * labels + (1-a) * img_top3_pred[:,0]
                
                _, txt_top3_pred = torch.topk(txt_outputs, 3, largest=True, sorted=True)
                b = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - txt_top3_pred), dim=1).to(torch.bool)
                b = b.to(torch.int8)
                txt_top3_pred = b * labels + (1-b) * txt_top3_pred[:,0]

                _, votting_top3_pred = torch.topk(votting_outputs, 3, largest=True, sorted=True)
                c = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - votting_top3_pred), dim=1).to(torch.bool)
                c = c.to(torch.int8)
                votting_top3_pred = c * labels + (1-c) * votting_top3_pred[:,0]


                total += labels.size(0)
                img_correct += (img_pred == labels).sum().item()
                img_top3_correct += (img_top3_pred == labels).sum().item()
                txt_correct += (txt_pred == labels).sum().item()
                txt_top3_correct += (txt_top3_pred == labels).sum().item()
                votting_correct += (votting_pred == labels).sum().item()
                votting_top3_correct += (votting_top3_pred == labels).sum().item()

                gt += labels.to("cpu").tolist()
                img_preds += img_pred.to("cpu").tolist()
                img_top3_preds += img_top3_pred.to("cpu").tolist()
                txt_preds += txt_pred.to("cpu").tolist()
                txt_top3_preds += txt_top3_pred.to("cpu").tolist()
                votting_preds += votting_pred.to("cpu").tolist()
                votting_top3_preds += votting_top3_pred.to("cpu").tolist()

                img_running_loss += img_loss.item()
                txt_running_loss += txt_loss.item()
                
                img_accuracy = img_correct / total
                img_top3_accuracy = img_top3_correct / total
                img_f1 = f1_score(y_true=gt, y_pred=img_preds, labels=label_list, average="macro", zero_division=0)
                img_top3_f1 = f1_score(y_true=gt, y_pred=img_top3_preds, labels=label_list, average="macro", zero_division=0)

                txt_accuracy = txt_correct / total
                txt_top3_accuracy = txt_top3_correct / total
                txt_f1 = f1_score(y_true=gt, y_pred=txt_preds, labels=label_list, average="macro", zero_division=0)
                txt_top3_f1 = f1_score(y_true=gt, y_pred=txt_top3_preds, labels=label_list, average="macro", zero_division=0)

                votting_accuracy = votting_correct / total
                votting_top3_accuracy = votting_top3_correct / total
                votting_f1 = f1_score(y_true=gt, y_pred=votting_preds, labels=label_list, average="macro", zero_division=0)
                votting_top3_f1 = f1_score(y_true=gt, y_pred=votting_top3_preds, labels=label_list, average="macro", zero_division=0)

                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] | (Image) "
                    f"Loss: {(img_running_loss / (batch + 1)):.3f} "
                    f"Acc: {img_accuracy * 100:.2f}% "
                    f"F1: {img_f1:.2f} | (Text) "
                    f"Loss: {(txt_running_loss / (batch + 1)):.3f} "
                    f"Acc: {txt_accuracy * 100:.2f}% "
                    f"F1: {txt_f1:.2f}"
                )

                if batch%100 == 0:
                    img_cm = wandb.plot.confusion_matrix(
                        y_true=gt,
                        preds=img_preds,
                        class_names=list(_dict_num_to_category.values())
                        )
                    txt_cm = wandb.plot.confusion_matrix(
                        y_true=gt,
                        preds=txt_preds,
                        class_names=list(_dict_num_to_category.values())
                        )
                    votting_cm = wandb.plot.confusion_matrix(
                        y_true=gt,
                        preds=votting_preds,
                        class_names=list(_dict_num_to_category.values())
                        )
                    wandb.log({
                        "train/img/loss": img_running_loss/(batch+1),
                        "train/img/accuracy": img_accuracy,
                        "train/img/top3_accuracy": img_top3_accuracy,
                        "train/img/f1": img_f1,
                        "train/img/top3_f1": img_top3_f1,
                        "train/txt/loss": txt_running_loss/(batch+1),
                        "train/txt/accuracy": txt_accuracy,
                        "train/txt/top3_accuracy": txt_top3_accuracy,
                        "train/txt/f1": txt_f1,
                        "train/txt/top3_f1": txt_top3_f1,
                        "train/votting/accuracy": votting_accuracy,
                        "train/votting/top3_accuracy": votting_top3_accuracy,
                        "train/votting/f1": votting_f1,
                        "train/votting/top3_f1": votting_top3_f1,
                        "train/img/cm": img_cm,
                        "train/txt/cm": txt_cm,
                        "train/votting/cm": votting_cm,
                    })

            pbar.close()

            val_img, val_txt, val_votting = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                "validation/img/accuracy": val_img[0],
                "validation/img/top3_accuracy": val_img[1],
                "validation/img/f1": val_img[2],
                "validation/img/top3_f1": val_img[3],
                "validation/img/cm": val_img[4],
                "validation/txt/accuracy": val_txt[0],
                "validation/txt/top3_accuracy": val_txt[1],
                "validation/txt/f1": val_txt[2],
                "validation/txt/top3_f1": val_txt[3],
                "validation/txt/cm": val_txt[4],
                "validation/votting/accuracy": val_votting[0],
                "validation/votting/top3_accuracy": val_votting[1],
                "validation/votting/f1": val_votting[2],
                "validation/votting/top3_f1": val_votting[3],
                "validation/votting/cm": val_votting[4],
            })

            if best_val_f1 > val_votting[2]:
                continue
            best_val_acc = val_votting[0]
            best_val_top3_acc = val_votting[1]
            best_val_f1 = val_votting[2]
            best_val_top3_f1 = val_votting[3]
            print(f"Model saved. Current best test f1: {best_val_f1:.3f}, top3_f1: {best_val_top3_f1:.3f}")
            torch.save(self.model.state_dict(), f=self.model_path)

        return best_val_acc, best_val_f1, best_val_top3_acc, best_val_top3_f1

    @torch.no_grad()
    def test(
        self,
        model: nn.Module,
        test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns data

        Returns:
            loss, f1, accuracy
        """
        total = 0
        gt = []
        img_running_loss, img_correct, img_top3_correct, img_preds, img_top3_preds = 0, 0, 0, [], []
        txt_running_loss, txt_correct, txt_top3_correct, txt_preds, txt_top3_preds = 0, 0, 0, [], []
        votting_correct, votting_top3_correct, votting_preds, votting_top3_preds = 0, 0, [], []
        label_list = [i for i in range(self.num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, data in pbar:
            labels = data.pop("category").to(self.device)
            for key in data.keys():
                data[key] = data[key].to(self.device)

            img_outputs, txt_outputs = model(data)

            img_outputs, txt_outputs = torch.squeeze(img_outputs), torch.squeeze(txt_outputs)

            img_outputs, txt_outputs = torch.squeeze(img_outputs), torch.squeeze(txt_outputs)
            img_loss, txt_loss = self.img_criterion(img_outputs, labels), self.txt_criterion(txt_outputs, labels)

            votting_outputs = torch.div(img_outputs + txt_outputs, 2)

            _, img_pred = torch.max(img_outputs, 1)
            _, txt_pred = torch.max(txt_outputs, 1)
            _, votting_pred = torch.max(votting_outputs, 1)

            _, img_top3_pred = torch.topk(img_outputs, 3, largest=True, sorted=True)
            a = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - img_top3_pred), dim=1).to(torch.bool)
            a = a.to(torch.int8)
            img_top3_pred = a * labels + (1-a) * img_top3_pred[:,0]
            
            _, txt_top3_pred = torch.topk(txt_outputs, 3, largest=True, sorted=True)
            b = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - txt_top3_pred), dim=1).to(torch.bool)
            b = b.to(torch.int8)
            txt_top3_pred = b * labels + (1-b) * txt_top3_pred[:,0]

            _, votting_top3_pred = torch.topk(votting_outputs, 3, largest=True, sorted=True)
            c = ~torch.prod(input = torch.abs(labels.unsqueeze(1) - votting_top3_pred), dim=1).to(torch.bool)
            c = c.to(torch.int8)
            votting_top3_pred = c * labels + (1-c) * votting_top3_pred[:,0]
            
            total += labels.size(0)
            img_correct += (img_pred == labels).sum().item()
            img_top3_correct += (img_top3_pred == labels).sum().item()
            txt_correct += (txt_pred == labels).sum().item()
            txt_top3_correct += (txt_top3_pred == labels).sum().item()
            votting_correct += (votting_pred == labels).sum().item()
            votting_top3_correct += (votting_top3_pred == labels).sum().item()

            gt += labels.to("cpu").tolist()
            img_preds += img_pred.to("cpu").tolist()
            img_top3_preds += img_top3_pred.to("cpu").tolist()
            txt_preds += txt_pred.to("cpu").tolist()
            txt_top3_preds += txt_top3_pred.to("cpu").tolist()
            votting_preds += votting_pred.to("cpu").tolist()
            votting_top3_preds += votting_top3_pred.to("cpu").tolist()

            img_running_loss += img_loss.item()
            txt_running_loss += txt_loss.item()

            pbar.update()
            pbar.update()
            pbar.set_description(
                f"Val: | (Voting) "
                f"Acc: {(votting_correct / total) * 100:.2f}% "
                f"F1: {f1_score(y_true=gt, y_pred=votting_preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
            
        img_accuracy = img_correct / total
        img_top3_accuracy = img_top3_correct / total
        img_f1 = f1_score(y_true=gt, y_pred=img_preds, labels=label_list, average="macro", zero_division=0)
        img_top3_f1 = f1_score(y_true=gt, y_pred=img_top3_preds, labels=label_list, average="macro", zero_division=0)

        txt_accuracy = txt_correct / total
        txt_top3_accuracy = txt_top3_correct / total
        txt_f1 = f1_score(y_true=gt, y_pred=txt_preds, labels=label_list, average="macro", zero_division=0)
        txt_top3_f1 = f1_score(y_true=gt, y_pred=txt_top3_preds, labels=label_list, average="macro", zero_division=0)

        votting_accuracy = votting_correct / total
        votting_top3_accuracy = votting_top3_correct / total
        votting_f1 = f1_score(y_true=gt, y_pred=votting_preds, labels=label_list, average="macro", zero_division=0)
        votting_top3_f1 = f1_score(y_true=gt, y_pred=votting_top3_preds, labels=label_list, average="macro", zero_division=0)

        # img_cm = wandb.plot.confusion_matrix(
        #     y_true=gt,
        #     preds=img_preds,
        #     class_names=list(_dict_num_to_category.values())
        #     )
        # txt_cm = wandb.plot.confusion_matrix(
        #     y_true=gt,
        #     preds=txt_preds,
        #     class_names=list(_dict_num_to_category.values())
        #     )
        # votting_cm = wandb.plot.confusion_matrix(
        #     y_true=gt,
        #     preds=votting_preds,
        #     class_names=list(_dict_num_to_category.values())
        #     )
        return (
            [img_accuracy, img_top3_accuracy, img_f1, img_top3_f1, _,],
            [txt_accuracy, txt_top3_accuracy, txt_f1, txt_top3_f1, _,],
            [votting_accuracy, votting_top3_accuracy, votting_f1, votting_top3_f1, _]
        )
