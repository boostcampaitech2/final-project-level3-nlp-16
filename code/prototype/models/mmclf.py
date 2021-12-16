import io
from typing import List, Dict, Any, Type, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from efficientnet_pytorch import EfficientNet
from transformers import AutoConfig, AutoTokenizer, BertModel


class MultimodalCLF(nn.Module):
    """Model class."""

    def __init__(self, num_classes: int = 15) -> None:
        super().__init__()
        self.img_model = EfficientNet.from_name("efficientnet-b0")

        self.txt_config = AutoConfig.from_pretrained("klue/bert-base")
        self.txt_model = BertModel(self.txt_config)

        self.img_classifier = FCLayer(self.img_model._fc.out_features, num_classes, 0, use_activation=False)
        self.txt_classifier = FCLayer(self.txt_config.hidden_size, num_classes, 0, use_activation=False)

    def forward(
        self,
        item: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_output = self.img_model(item["pixel_values"])
        txt_outputs = self.txt_model(
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
            token_type_ids=item["token_type_ids"],
        )
        txt_output = txt_outputs.pooler_output

        img_logits = self.img_classifier(img_output)
        txt_logits = self.txt_classifier(txt_output)

        return img_logits, txt_logits

class FCLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float=0., use_activation: bool=True) -> None:
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


def get_model(model_path: str = "models/best.pt") -> MultimodalCLF:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCLF(num_classes=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, "constant")


def _transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def _tokenize_title(title: str):
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    tokenized_title = tokenizer(
        title,
        return_tensors = "pt",
        max_length = 32,
        padding = "max_length",
        truncation = "only_first"
    )
    return {key: val.clone().detach() for key, val in tokenized_title.items()}


@torch.no_grad()
def predict_from_multimodal(model: MultimodalCLF, image_bytes: bytes, title: str, config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item = _tokenize_title(title)
    transformed_image = _transform_image(image_bytes)
    item["pixel_values"] = transformed_image
    for key in item.keys():
        item[key] = item[key].to(device)
    
    model.eval()
    img_logits, txt_logits = model.forward(item)
    logits = torch.div(img_logits+txt_logits, 2)
    _, top3_pred = torch.topk(logits, 3, largest=True, sorted=True)
    return [config["classes"][pred] for pred in top3_pred.tolist()[0]]


def get_config(config_path: str = "models/config.yaml"):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
