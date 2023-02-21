import os
import sys
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

LABELS = ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'sw', 'th', 'tl', 'tr', 'uk', 'vi', 'zh_cn', 'zh_tw']

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

label_dict = dict(zip(range(len(LABELS)), LABELS))

LOCAL_PATH = "../fine_tuning/test_trainer/checkpoint-96000"
FILE_NAME = "pytorch_model.bin"
LOCAL_W_PATH = os.path.join(LOCAL_PATH, FILE_NAME)

# print(LOCAL_DATA_PATH)

config = RobertaConfig.from_json_file(os.path.join(LOCAL_PATH, "config.json"))

class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RobertaForSequenceClassification(config).to(device)
        self.model.load_state_dict(torch.load(LOCAL_W_PATH, map_location=device))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    
    def forward(self, text:str):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,   
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)
        logits = self.model(encoding["input_ids"], attention_mask=encoding["attention_mask"])[0]
        torch.cuda.empty_cache()
        return logits.detach().cpu().numpy()
        # return self.get_max_n(logits.detach().cpu().numpy())

    def predict(self, texts):
        logits = self.forward(texts)
        return self.id2label(logits.argmax(-1))

    def id2label(self, ids):
        return np.vectorize(lambda x: label_dict.get(x))(ids)


def main():
    data_path = "data/model_comparison.csv"
    test_data = pd.read_csv(data_path)

    clf = Classifier()
    start = time.time()
    test_data['xlm-roberta-finetune'] = [clf.predict(text)[0] for text in test_data['text']]
    # size = test_data.shape[0]
    # step_size = 10
    # indice = np.linspace(0, size, step_size, dtype= int)
    # batch_list= [(indice[i], indice[i+1]) for i in range(len(indice)-1)]
    # preds = []
    # for lb, ub  in tqdm(batch_list):
    #     preds.extend(clf.predict(test_data.loc[lb:ub, 'text'].tolist()).tolist())

    end = time.time()
    # test_data['xlm-roberta-finetune'] = preds
    test_data.to_csv(data_path)
    print(f"Done! ({end - start:.5f} sec)", end = '\n'*2)

if __name__ == "__main__":
    main()
