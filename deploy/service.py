import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import wordnet
import bentoml
from bentoml.io import JSON, NumpyNdarray
import torch
from module.tool import load_json, save_json
LABEL= ["ar", "cs", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "ko", "mn", "ms", "nl", "pl", "pt", "ru", "sv", "sw", "th", "tl", "tr", "uk", "vi", "zh_cn", "zh_tw"]

label_dict = {int(id):label for id, label in enumerate(LABEL)}

class Classifier():
    def __init__(self, runner):
        self.tokenizer= torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', '../deploy/model/best_model/')
        self.runner= runner

    async def __call__(self, text, n=5):
        encoding = self._encode(text)
        logits = await self.runner.run.async_run(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(torch.Tensor(logits.squeeze()), -1).detach().cpu().numpy()
        indice_n = (-probs).argsort()[:n]
        result = {self._id2label(idx): probs[idx] for idx in indice_n}
        return result
    
    def _encode(self, text):
        encoding = self.tokenizer(text, add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')
        return encoding
    
    def _id2label(self, id):
         return label_dict.get(id)


providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

runner= bentoml.onnx.get('lang_id:latest').with_options(providers=providers).to_runner()

svc = bentoml.Service('language_identification', runners=[runner])

# runner.init_local()

classifier = Classifier(runner)

@svc.api(input=JSON(), output=JSON())
async def predict(input_json) -> JSON:
     text = input_json['text']
     resp_json = await classifier(text, 3)
     
     return resp_json