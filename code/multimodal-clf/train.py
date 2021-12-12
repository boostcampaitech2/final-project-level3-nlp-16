"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union
import yaml
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict


from src.model import MultimodalCLF
from src.trainer import TorchTrainer
from src.utils.common import read_yaml, seed_everything
from src.utils.torch_utils import check_runtime, model_info


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""

    # Save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    # Load model
    model = MultimodalCLF(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    model.to(device)

    # Load dataset
    dataset = load_dataset(data_config["DATA_PATH"], use_auth_token=True)
    dataset = dataset["train"].filter(lambda x: x["img_bytes"] != b"") # 이미지 없는 데이터 제외
    dataset = dataset.remove_columns(["pid","description","tag"])

    train_valid = dataset.train_test_split(test_size=0.2) # train valid split
    datasets = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": train_valid["test"],
        }
    )

    # Load image transforms
    transform_train_params=data_config["AUG_TRAIN_PARAMS"]
    if not transform_train_params:
        transform_train_params = dict()
    train_transform = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TRAIN"],
    )(img_size=data_config["IMG_SIZE"], **transform_train_params)

    valid_transform = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(img_size=data_config["IMG_SIZE"])

    # Load text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["txt_backbone"])

    def bytes_to_pil(b):
        return Image.open(BytesIO(b)).convert('RGB')

    # # TODO: max_length 추가하기
    def text_to_token(text):
        return tokenizer(text, return_tensors="pt", max_length=32, padding="max_length", truncation="only_first")
        
    def apply_train_transforms(example_batch):
        # apply image transform
        example_batch["pixel_values"] = [
            train_transform(bytes_to_pil(b)) for b in example_batch.pop("img_bytes")
        ]

        # apply text tokenizer
        for key, val in text_to_token(example_batch.pop("title")).items():
            example_batch[key] = val
        return example_batch

    def apply_valid_transforms(example_batch):
        # apply image transform
        example_batch["pixel_values"] = [
            valid_transform(bytes_to_pil(b)) for b in example_batch.pop("img_bytes")
        ]

        # apply text tokenizer
        for key, val in text_to_token(example_batch.pop("title")).items():
            example_batch[key] = val
        return example_batch
    
    # Create dataloader
    train_loader = DataLoader(
        dataset=datasets["train"].with_transform(apply_train_transforms),
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=data_config["TRAIN_BATCH_SIZE"],
        num_workers=10,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=datasets["valid"].with_transform(apply_valid_transforms),
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=data_config["TRAIN_BATCH_SIZE"],
        num_workers=5,
    )

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=data_config["INIT_LR"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_loader),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        num_classes=data_config["NUM_CLASSES"],
        verbose=1,
    )

    # WandB logging
    wandb.init(
        entity="nlprime",
        project="final_project",
    )

    # Training
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_loader,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=valid_loader,
    )

    # Evaluation with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc, test_top3_f1, test_top3_acc = trainer.test(
        model=model, test_dataloader=valid_loader
    )
    return test_loss, test_f1, test_acc, test_top3_f1, test_top3_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3_kluebert.yaml",
        type=str,
        help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/secondhand-goods.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_dir = "exp/latest"
    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    test_loss, test_f1, test_acc, test_top3_f1, test_top3_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
    
    print(f"Test Result // loss: {test_loss:.3f}, f1: {test_f1:.3f}, acc: {test_acc:.3f}, top3_f1: {test_top3_f1:.3f}, top3_acc: {test_top3_acc:.3f}")

