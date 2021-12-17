from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenized_sentence = self.tokenizer(
            dataset,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

    def __getitem__(self, idx):
        encoded = {key: val[idx] for key, val in self.tokenized_sentence.items()}
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    def __len__(self):
        return len(self.dataset)
