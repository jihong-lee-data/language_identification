import os
import sys
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

config= load_json('model/config.json')
label_dict = dict(zip(config['label2id'].values(), config['label2id'].keys()))

torch.cuda.empty_cache()

MODEL_PATH = f'model/lang_id_{processor}_{device.type}.pt'

class Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.jit.load(MODEL_PATH, map_location=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("model", use_fast=True)

    def _forward(self, x):
        x= self.tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt').to(device)
        output =self.model(x['input_ids'], attention_mask=x['attention_mask'])['logits']
        return output        
    
    def predict(self, text, n = 3):
        logits= self._forward(text).squeeze()
        
        indice_n = (-logits).argsort()[:n].detach().cpu().numpy().tolist()

        preds= [label_dict.get(idx) for idx in indice_n]
        probs= logits.softmax(dim=-1)[indice_n].detach().cpu().numpy().tolist()
        return dict(zip(preds, probs))
        
        
        


def main():
    
    clf = Inference()
    
    while True:
        try:
            text= input('Enter text: ')
            start = time.time()
            pred = clf.predict(text)
            end = time.time()
            print(pred)
            print(f"Inference time: ({end - start:.5f} sec)", end = '\n'*2)
            torch.cuda.empty_cache()
        except EOFError:
            print('Bye!')
            break

if __name__ == '__main__':
    
    main()