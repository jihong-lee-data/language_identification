import numpy as np
import pandas as pd
# import seaborn as sns
import os
import time
import re
import pickle
import joblib
import json
from pprint import pprint
import gzip

import warnings
warnings.filterwarnings(action='ignore')


def get_time(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print(f"Done ({end - start:.5f} sec)", end = '\n'*2)
        return result
    return wrapper

def tokenizer(text):
    text = str(text)
    # remove special characters
    text = re.sub(r'[!@#$(),，\n"%^*?？:;~`0-9&\[\]\。\/\.\=\-]', ' ', text)
    text = re.sub(r'[\s]{2,}', ' ', text.strip())

    text = text.lower().strip()
    
    hira_chars = ("\u3040-\u309f")
    kata_chars = ("\u30a0-\u30ff")
    zh_chars = ("\u2e80-\u2fff\u31c0-\u31ef\u3200-\u32ff\u3300-\u3370"           
               "\u33e0-\u33fe\uf900-\ufaff\u4e00-\u9fff") 
    
    tokenized = text.split()
    
    ja_exist = re.findall(f'[{hira_chars}{kata_chars}]+', text)
    zh_exist = re.findall(f'[{zh_chars}]+', text)
    
    if ja_exist:
        ja_tokens = []
        for token in tokenized:
            ja_tokens.extend(re.findall(f'[{zh_chars}]+|[{hira_chars}]+|[{kata_chars}]+', token))
            ja_tokens.extend(re.findall(f'[^{zh_chars}{hira_chars}{kata_chars}]+', token))
        return ja_tokens
    elif zh_exist:
        zh_tokens = []
        for token in tokenized:
            zh_tokens.extend(re.findall(f'[{zh_chars}]', token))
            zh_tokens.extend(re.findall(f'[^{zh_chars}]+', token))
        return zh_tokens
    else:
        char_tokens = []
        for token in tokenized:
            char_tokens.extend(re.findall('.', token))
        return tokenized + char_tokens   

class ISO():
    def __init__(self):
        with open('../model_development/resource/iso.json', 'r') as f:
            self.iso_dict = json.load(f)
        self.search_list = [[i[0]] + i[1] for i in self.iso_dict["en_to_iso"].items()]   

    def search(self, text, tol = 2):
        results = []
        for en_id_pair in self.search_list:
            if any((self._word_validation(text, target, tol) for target in en_id_pair)):
                results.append(en_id_pair)
        return results


    def _word_validation(self, test:str, target:str, tol = 2):
        if tol == 0:
            return test in [target]
        elif tol == 1:
            return test.lower() in [target.lower()]
        elif tol == 2:
            return test.lower() in target.lower()

class Model():
    def __init__(self, model_name, model = None):
        model_dir = "model"
        self.model_path = os.path.join(model_dir, model_name, "model.pkl")
        if os.path.exists(self.model_path):
            try:
                self.model = self.load_model()
                self.labels = self.model.classes_
                self._int2label_dict = dict(zip(range(len(self.labels)), self.labels))
                self._label2int_dict = dict(zip(self.labels, range(len(self.labels))))
                self.int2label= np.vectorize(self.int2label)
                self.label2int= np.vectorize(self.label2int)
            except:
                self.model = model
        else:
            self.model = model            
        


    def load_model(self):
        with gzip.open(self.model_path, 'rb') as f:
            model = pickle.loads(joblib.load(self.model_path))

        # print(f"This model is loaded from {self.model_path}.")
        return model


    def predict(self, text, n = 3):
        probs = self.model.predict_proba([text])
        preds = probs.argsort()[0, ::-1][:n]
        return preds, probs[0, preds]

    
    def int2label(self, value):
        return self._int2label_dict.get(value)

    @np.vectorize
    def label2int(self, value):
        return self._label2int_dict.get(value)
            
            

        
