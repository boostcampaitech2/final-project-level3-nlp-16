from typing import Dict, List, Type, Union, Tuple

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from transformers import AutoConfig, BertModel


class MultimodalCLF(nn.Module):
    """Model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]],
    ) -> None:
        super().__init__()
        self.img_model = EfficientNet.from_pretrained(cfg["img_backbone"])
        # self.img_model = EfficientNet.from_name(cfg["img_backbone"])

        self.txt_config = AutoConfig.from_pretrained(cfg["txt_backbone"])
        self.txt_model = BertModel(self.txt_config)

        self.img_classifier = FCLayer(self.img_model._fc.out_features, 15, 0, use_activation=False)
        self.txt_classifier = FCLayer(self.txt_config.hidden_size, 15, 0, use_activation=False)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_output = self.img_model(data["pixel_values"])
        txt_outputs = self.txt_model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"],
        )
        txt_output = txt_outputs.pooler_output

        img_logits = self.img_classifier(img_output) # (batch_size, 15)
        txt_logits = self.txt_classifier(txt_output) # (batch_size, 15)

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