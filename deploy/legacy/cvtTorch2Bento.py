
import bentoml
import torch
from torch import nn
from torch.nn import functional as F
import os
import torch
import time
import warnings
warnings.filterwarnings(action='ignore')
from pathlib import Path
from module.tool import load_json


MODEL_DIR = Path('model/best_model')


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model= torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'model/best_model/')
        self.tokenizer= torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'model/best_model/')
        self.model.eval()

    def _encode(self, text):
        encoding = self.tokenizer(text, add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')
        return encoding
    
    def _forward(self, text):
        encoding = self._encode(text)
        output = self.model(**encoding)
        return output

    def predict(self, text, n=5):
        logits= self._forward(text)["logits"].squeeze()
        probs = logits.softmax(dim= -1).detach().cpu().numpy()
        indice_n = (-probs).argsort()[:n]
        pred_n = [self.model.config.id2label.get(idx) for idx in indice_n]
        prob_n = probs[indice_n]
        return dict(zip(pred_n, prob_n))

classifier = Classifier()    



print(classifier.predict('안녕하세요', n=3))

# bentoml.pytorch.save(
#     classifier,
#     "xlm-roberta-finetune",
#     signatures={"__call__": {"batchable": True, "batch_dim": 0},
#                 "predict": {"batchable": True, "batch_dim": 0},
#                 },
# )

# bentoml.transformers.save_model(
#     name= "xlm-roberta-finetune",
#     pipeline=classifier,
#     task_name= TASK_NAME,
#     signatures={
#         "__call__": {
#             "batchable": True,
#             "batch_dim": 0,
#         }
#     }
#     )


# loaded = bentoml.transformers.load_model("xlm-roberta-finetune:latest")

# loaded("안녕하세요", n=3)