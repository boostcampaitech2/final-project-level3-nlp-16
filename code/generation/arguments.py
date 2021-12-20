from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="skt/kogpt2-base-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/opt/ml/code/generation/",
        metadata={"help": "The name of the dataset to use."},
    )

    dataset_type: str = field(
        default="cosine",
        metadata={"help": "metric to extract description [normal, cosine]"},
    )

    cosine_rate: float = field(
        default=0.3, metadata={"help": "rate to compare with cosine similarity"}
    )


@dataclass
class TrainingArguments(TrainingArguments):

    output_dir: str = field(default="/opt/ml/code/generation/output")
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Define the number of epoch to run during training"},
    )
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    overwrite_output_dir: bool = field(default=True)
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "Select evaluation strategy[no, steps, epoch]"},
    )
    save_strategy: str = field(default="epoch")
    save_steps: int = field(default=3000)
    eval_steps: int = field(default=3000)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="loss")
