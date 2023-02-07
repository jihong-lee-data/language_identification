import os
import sys
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig
import torch
import time

LABELS = ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'sw', 'th', 'tl', 'tr', 'uk', 'vi', 'zh_cn', 'zh_tw']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()

label_dict = dict(zip(range(len(LABELS)), LABELS))

MODEL_PATH = 'traced_model.pt'


class Classifier:
    def __init__(self, label_dict = label_dict):
        self.model = load_model()
        self.label_dict = label_dict

    def classify(self, text):
        probs, logits = self.model(text)
        return probs[0], logits[0]


    def get_max_n(self, values: list, n = 3):
        max_n_idx = (-values).argsort()[:n]
        max_n_labels, max_n_values = [], []
        for idx in max_n_idx:
            max_n_values.append(values[idx])
            max_n_labels.append(self.label_dict[idx])
        return max_n_labels, max_n_values



class load_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.jit.load(MODEL_PATH, map_location=device)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            
    
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,   
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)
        output = self.model(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        logits = output[0]
        probs = torch.softmax(logits, -1)
        torch.cuda.empty_cache()
        return probs.detach().cpu().numpy(), logits.detach().cpu().numpy()


def main():
    

    clf = Classifier(label_dict)
    
    while True:
        try:
            text= input('Enter text: ')
            start = time.time()
            probs, logits = clf.classify(text)
            preds_n, probs_n = clf.get_max_n(probs, n = 3)
            end = time.time()
            print(f"time: ({end - start:.5f} sec)", end = '\n'*2)
            print(dict(zip(preds_n, probs_n)))

            torch.cuda.empty_cache()
        except EOFError:
            print('Bye!')
            break

if __name__ == '__main__':
    
    main()