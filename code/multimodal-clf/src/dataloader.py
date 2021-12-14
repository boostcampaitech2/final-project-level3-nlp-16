from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict
from transformers import AutoTokenizer

from PIL import Image
from io import BytesIO


def get_dataloader(
    datasets: DatasetDict,
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get dataloader for train, validation, test"""

    # Load image transforms
    transform_train_params=data_config["AUG_TRAIN_PARAMS"]
    if not transform_train_params:
        transform_train_params = dict()

    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TRAIN"],
    )(img_size=data_config["IMG_SIZE"], **transform_train_params)
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(img_size=data_config["IMG_SIZE"])

    # Load text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["txt_backbone"])

    def text_to_token(text):
        return tokenizer(text, return_tensors="pt", max_length=32, padding="max_length", truncation="only_first")
        
    def apply_transform_train(example_batch):
        # apply image transform
        example_batch["pixel_values"] = [
            transform_train(bytes_to_pil(b)) for b in example_batch.pop("img_bytes")
        ]
        # apply text tokenizer
        for key, val in text_to_token(example_batch.pop("title")).items():
            example_batch[key] = val
        return example_batch

    def apply_transform_test(example_batch):
        # apply image transform
        example_batch["pixel_values"] = [
            transform_test(bytes_to_pil(b)) for b in example_batch.pop("img_bytes")
        ]
        # apply text tokenizer
        for key, val in text_to_token(example_batch.pop("title")).items():
            example_batch[key] = val
        return example_batch

    # Create dataloader
    train_loader = DataLoader(
        dataset=datasets["train"].with_transform(apply_transform_train),
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=data_config["TRAIN_BATCH_SIZE"],
        num_workers=10,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=datasets["valid"].with_transform(apply_transform_test),
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=data_config["VALID_BATCH_SIZE"],
        num_workers=5,
    )
    test_loader = DataLoader(
        dataset=datasets["test"].with_transform(apply_transform_test),
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=data_config["VALID_BATCH_SIZE"],
        num_workers=5,
    )

    return train_loader, valid_loader, test_loader

def bytes_to_pil(b):
    return Image.open(BytesIO(b)).convert('RGB')
