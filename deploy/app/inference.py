from app.tool import load_json
import platform
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Union
import numpy as np

processor = platform.processor().lower()
if 'arm' in  processor:
    backend= 'qnnpack'
else:
    backend= 'fbgemm'

torch._C._set_graph_executor_optimize(False)

config= load_json('model/config.json')
label_dict = dict(zip(config['label2id'].values(), config['label2id'].keys()))

class Inference(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device= torch.device(device)
        MODEL_PATH = f'model/lang_id_{processor}_{self.device.type}.pt'
        self.model = torch.jit.load(MODEL_PATH, map_location=self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("model", use_fast=True)

    def _encode(self, text: Union[str, List[str]]) -> dict[torch.Tensor]:
        encoding= self.tokenizer(text, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt').to(self.device)
        return encoding

    def forward(self, text: Union[str, List[str]]) -> dict[torch.Tensor]:
        encoding= self._encode(text)
        output =self.model(encoding['input_ids'], attention_mask=encoding['attention_mask'])['logits']
        torch.cuda.empty_cache()
        return output        
    
    def predict(self,
                text: Union[str, List[str]],
                n: int= 3):
        logits= self(text)
        probabilities= logits.softmax(dim=-1)
        
        col_id = (-logits).argsort(dim=-1)[:, :n].detach().cpu().numpy().tolist()
        row_id = np.repeat(np.arange(len(col_id)), n).reshape(len(col_id), n)
        
        
        # [{label_dict.get(idx): logits.softmax(dim=-1)} for idx in indice] for indice in indice_n]

        return [{label_dict.get(idx): prob for idx, prob in zip(indice, probs)} for indice, probs in zip(col_id, probabilities[row_id, col_id].detach().cpu().numpy().tolist())]
            # probs= logits.softmax(dim=-1)[:, indice_n].detach().cpu().numpy().tolist()
        #     return dict(zip(preds, probs))
        # elif isinstance(text, list):
        #     return [{label_dict.get(idx): logits.softmax(dim=-1)} for idx in indice] for indice in indice_n]
            
            