import os
import sys
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer
from module.tool import load_json
import torch
import time
import platform

device = torch.device('cpu')
print("Device: ", device)

processor = platform.processor().lower()
if 'x86' in  processor:
    backend= 'fbgemm'
elif 'arm' in  processor:
    backend= 'qnnpack'
torch.backends.quantized.engine = backend

torch._C._set_graph_executor_optimize(False)

config= load_json('model/best_model/config.json')
label_dict = dict(zip(config['label2id'].values(), config['label2id'].keys()))


torch.cuda.empty_cache()

MODEL_PATH = f'model/traced/lang_id_{processor}_{device.type}.pt'

class Inference(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.jit.load(MODEL_PATH, map_location=device)
        # self.model = torch.jit.script(self.model)
        # self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("model/best_model", use_fast=True)

    def forward(self, x):
        x= self.tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt').to(self.device)
        logits =self.model(x['input_ids'], attention_mask=x['attention_mask'])['logits']
        pred = label_dict.get(logits.argmax().item())
        
        return pred        


def main():
    
    clf = Inference()
    
    while True:
        try:
            text= input('Enter text: ')
            start = time.time()
            pred = clf(text)
            end = time.time()
            print(pred)
            print(f"Inference time: ({end - start:.5f} sec)", end = '\n'*2)
            torch.cuda.empty_cache()
        except EOFError:
            print('Bye!')
            break

if __name__ == '__main__':
    
    main()