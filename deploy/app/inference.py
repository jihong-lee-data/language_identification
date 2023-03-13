from typing import List, Union, Dict
from app.tool import load_json
import platform
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np

processor = platform.processor().lower()

config = load_json("model/config.json")


class Inference(nn.Module):
    def __init__(self, device="cpu"):
        """
        Initialize the Inference class.

        Args:
        - device (str): the device where the model will run (default: "cpu")
        """
        super().__init__()
        self.device = torch.device(device)
        MODEL_PATH = f"model/lang_id_{processor}_{self.device.type}.pt"
        self.model = torch.jit.load(MODEL_PATH, map_location=self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("model", use_fast=True)
        self.label_dict = dict(
            zip(config["label2id"].values(), config["label2id"].keys())
        )

    def _encode(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Encode the input text.

        Args:
        - text (Union[str, List[str]]): the input text

        Returns:
        - encoding (Dict[str, torch.Tensor]): the encoding of the input text
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        return encoding

    def forward(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Compute the output of the model.

        Args:
        - text (Union[str, List[str]]): the input text

        Returns:
        - output (Dict[str, torch.Tensor]): the output of the model
        """
        encoding = self._encode(text)
        output = self.model(
            encoding["input_ids"], attention_mask=encoding["attention_mask"]
        )["logits"]
        torch.cuda.empty_cache()
        return output

    def predict(self, text: Union[str, List[str]], n: int = 1) -> Dict:
        """
        Make a prediction based on the input text.

        Args:
        - text (Union[str, List[str]]): the input text
        - n (int): the number of predictions to return (default: 1)

        Returns:
        - predictions (Dict): the predicted labels and their corresponding probabilities
        """
        logits = self(text)
        probabilities = logits.softmax(dim=-1)

        col_id = (-logits).argsort(dim=-1)[:, :n].detach().cpu().numpy().tolist()
        row_id = np.repeat(np.arange(len(col_id)), n).reshape(len(col_id), n)

        return [
            {self.label_dict.get(idx): prob for idx, prob in zip(indice, probs)}
            for indice, probs in zip(
                col_id, probabilities[row_id, col_id].detach().cpu().numpy().tolist()
            )
        ]
