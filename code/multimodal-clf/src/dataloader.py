"""Tune Model.

- Author: Junghoon Kim, Jongkuk Lim, Jimyeong Kim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com, wlaud1001@snu.ac.kr
- Reference
    https://github.com/j-marple-dev/model_compression
"""
import glob
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder, VisionDataset
from PIL import Image
import pandas as pd
import pickle


with open('dict_label_to_num', 'rb') as f:
    _dict_label_to_num = pickle.load(f)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict[str, Any], dataframe, tokenized_txt):
        dataset_name=config["DATASET"]
        img_size=config["IMG_SIZE"]
        transform_train=config["AUG_TRAIN"]
        transform_train_params=config["AUG_TRAIN_PARAMS"]
        if not transform_train_params:
            transform_train_params = dict()

        self.dataframe = dataframe

        # preprocessing policies
        self.transform_train = getattr(
            __import__("src.augmentation.policies", fromlist=[""]),
            transform_train,
        )(dataset=dataset_name, img_size=img_size, **transform_train_params)

        self.tokenized_txt = tokenized_txt
        self.labels = list(self.dataframe["category"].apply(lambda x: _dict_label_to_num[str(x)]))


    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_txt.items()}
        
        image = Image.open(self.dataframe.loc[idx, "img"]).convert('RGB')
        image = self.transform_train(image)
        item["img"] = image

        labels = torch.tensor(self.labels[idx])
        return item, labels

    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict[str, Any], dataframe, tokenized_txt):
        dataset_name=config["DATASET"]
        img_size=config["IMG_SIZE"]
        transform_test=config["AUG_TEST"]
        transform_test_params=config.get("AUG_TEST_PARAMS")
        if not transform_test_params:
            transform_test_params = dict()

        self.dataframe = dataframe

        # preprocessing policies
        self.transform_test= getattr(
            __import__("src.augmentation.policies", fromlist=[""]),
            transform_test,
        )(dataset=dataset_name, img_size=img_size, **transform_test_params)

        self.tokenized_txt = tokenized_txt
        self.labels = list(self.dataframe["category"].apply(lambda x: _dict_label_to_num[str(x)]))


    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_txt.items()}
        
        image = Image.open(self.dataframe.loc[idx, "img"]).convert('RGB')
        image = self.transform_test(image)
        item["img"] = image

        labels = torch.tensor(self.labels[idx])
        return item, labels

    def __len__(self):
        return len(self.labels)


def get_dataloader(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloader for training and testing."""

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=10,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=5
    )

    return train_loader, valid_loader
