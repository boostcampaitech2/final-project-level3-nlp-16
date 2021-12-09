from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    GPT2LMHeadModel,
    Trainer,
    set_seed,
)
from arguments import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)
from data import CustomDataset
from preprocess import (
    create_data_normal,
    create_data_cosine,
)


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
        )
    )
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()

    print(f"backbone model is {model_args.model_name_or_path}")
    print(f"data is {data_args.dataset_name}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="<sep>",
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )
    model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.train()

    if data_args.dataset_type == "cosine":
        tag_dataset = create_data_cosine(data_args.dataset, data_args.cosine_rate)
    else:
        tag_dataset = create_data_normal(data_args.dataset_name)

    dataset = CustomDataset(tag_dataset["data"].tolist(), tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
