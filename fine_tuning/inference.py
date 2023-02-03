import os
import sys
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
import torch

LABELS = ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'sw', 'th', 'tl', 'tr', 'uk', 'vi', 'zh_cn', 'zh_tw']

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

label_dict = dict(zip(range(len(LABELS)), LABELS))

LOCAL_PATH = "test_trainer/checkpoint-500"
FILE_NAME = "pytorch_model.bin"
LOCAL_DATA_PATH = os.path.join(LOCAL_PATH, FILE_NAME)

# print(LOCAL_DATA_PATH)

config = RobertaConfig.from_json_file(os.path.join(LOCAL_PATH, "config.json"))

class Classifier:
    def __init__(self, label_dict = label_dict):
        self.model = load_model()
        self.label_dict = label_dict

    def classify(self, text):
        probs, logits = self.model(text)
        return probs[0].detach().cpu().numpy(), logits[0].detach().cpu().numpy()        


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
        self.roberta = RobertaModel.from_pretrained("xlm-roberta-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 30).to(device)
        
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
        output = self.roberta(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:,0,:]
        logits = self.classifier(output)
        probs = torch.softmax(logits, -1)
        torch.cuda.empty_cache()
        
        return probs, logits

def main():
    clf = Classifier(label_dict)

    while True:
        try:
            text= input('Enter text: ')
            probs, logits = clf.classify(text)

            preds_3, probs_3 = clf.get_max_n(probs, n = 3)
            torch.cuda.empty_cache()
            print(dict(zip(preds_3, probs_3)))

        except EOFError:
            print('Bye!')
            break

if __name__ == '__main__':
    main()