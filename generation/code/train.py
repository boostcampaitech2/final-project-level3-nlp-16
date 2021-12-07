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
    create_tagwholeset,
    create_data_with_passage,
    create_data_for_all_data,
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
    tag_dataset = create_data_with_passage(data_args.dataset_name)
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
