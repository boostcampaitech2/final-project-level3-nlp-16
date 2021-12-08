"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataloader import TrainDataset, TestDataset, get_dataloader
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
    
    # Tokenize Titles
    df = pd.read_csv(data_config["DATA_PATH"], sep="\t")
    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["category"])
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    
    # TODO: max_length 추가하기
    tokenizer = AutoTokenizer.from_pretrained(model_config["txt_backbone"])
    train_tokenized_titles = tokenizer(
        list(train_df["title"]),
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        truncation="only_first"
        )
    valid_tokenized_titles = tokenizer(
        list(valid_df["title"]),
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        truncation="only_first"
        )
    
    # Create dataset
    train_dataset = TrainDataset(data_config, train_df, train_tokenized_titles)
    valid_dataset = TestDataset(data_config, valid_df, valid_tokenized_titles)

    # Create dataloader
    train_dl, valid_dl = get_dataloader(train_dataset, valid_dataset, data_config["BATCH_SIZE"])

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
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
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=valid_dl,
    )

    # Evaluation with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc, test_top3_f1, test_top3_acc = trainer.test(
        model=model, test_dataloader=valid_dl
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
        "--data", default="configs/data/bunjang.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.path.join(data_config["DATA_PATH"], "data.tsv")
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

