import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union
import yaml

import torch
import torch.nn as nn

import wandb
from datasets import load_dataset, DatasetDict

from src.model import MultimodalCLF
from src.trainer import TorchTrainer
from src.utils.common import read_yaml, seed_everything, model_info
from src.dataloader import get_dataloader


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    device: torch.device,
) -> Tuple[float, float, float, float]:

    # Save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    # Load model
    model = MultimodalCLF(model_config)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    # model_info(model, verbose=False)
    model.to(device)

    # Load dataset
    dataset = load_dataset(data_config["DATA_PATH"], use_auth_token=True)
    dataset = dataset["train"].filter(lambda x: x["img_bytes"] != b"") # filtering non-image data
    dataset = dataset.remove_columns(["pid","description","tag"])

    # Split dataset to train, validation, test
    train_valid = dataset.train_test_split(test_size=0.2)
    valid_test = train_valid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": valid_test["test"],
            "test": valid_test["test"],
        }
    )
    
    # Load dataloder
    train_loader, valid_loader, test_loader = get_dataloader(datasets, data_config, model_config)

    # Create optimizer, scheduler, criterion for img_model and txt_model
    img_optimizer = torch.optim.AdamW(
        model.img_model.parameters(), lr=data_config["IMG_INIT_LR"]
    )
    txt_optimizer = torch.optim.AdamW(
        model.txt_model.parameters(), lr=data_config["TXT_INIT_LR"]
    )
    img_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=img_optimizer,
        max_lr=data_config["IMG_INIT_LR"],
        steps_per_epoch=len(train_loader),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    txt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=txt_optimizer,
        max_lr=data_config["TXT_INIT_LR"],
        steps_per_epoch=len(train_loader),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    img_criterion = nn.CrossEntropyLoss()
    txt_criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=(img_criterion, txt_criterion),
        optimizer=(img_optimizer, txt_optimizer),
        scheduler=(img_scheduler, txt_scheduler),
        device=device,
        model_path=model_path,
        num_classes=data_config["NUM_CLASSES"],
    )

    # WandB logging
    wandb.init(
        entity="nlprime",
        project="final_project",
    )

    # Training
    best_acc, best_f1, best_top3_acc, best_top3_f1 = trainer.train(
        train_dataloader=train_loader,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=valid_loader,
    )

    # Evaluation with test dataset
    model.load_state_dict(torch.load(model_path))
    test_acc, test_f1, test_top3_acc, test_top3_f1, test_cm = trainer.test(
        model=model, test_dataloader=test_loader
    )
    wandb.log({
        "test/accuracy": test_acc,
        "test/top3_accuracy": test_top3_acc,
        "test/f1": test_f1,
        "test/top3_f1": test_top3_f1,
        "test/confusion_matrix": test_cm,
    })
    return test_f1, test_acc, test_top3_f1, test_top3_acc


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
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/data.yml'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    test_f1, test_acc, test_top3_f1, test_top3_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        device=device,
    )
    print(f"Test Result | Acc: {test_acc:.3f}, F1: {test_f1:.3f}, Top3_Acc: {test_top3_acc:.3f}, Top3_F1: {test_top3_f1:.3f}")
    