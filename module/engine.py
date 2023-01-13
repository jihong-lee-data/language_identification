import numpy as np
import os
import time
import re
import pickle
import joblib
import json
import datasets
from datasets import load_dataset, load_from_disk
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')


with open('resource/iso_639_1.json', 'r') as f:
    iso_dict = json.load(f)


def get_time(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print(f"Done ({end - start:.5f} sec)", end = '\n'*2)
        return result
    return wrapper


# @get_time
# def load_hf_dataset(dataset_path:str, save_to_disk: bool = False, **kwargs):
#     if not os.path.isdir(dataset_path):
#         print('Downloading dataset...')
#         dataset = load_dataset(dataset_path, kwargs)
#         if save_to_disk:
#             dataset.save_to_disk(dataset_path)
#         print('The dataset is saved and loaded.')
#     else: 
#         print('The dataset already exists.')
#         dataset = load_from_disk(dataset_path)
#         print('The dataset is loaded from disk.')
#     return dataset


@np.vectorize
def _rm_spcl_char(text):
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9\[\]]', ' ', text)
    text = re.sub(r'[\s]{2,}', ' ', text.strip())
    text = text.lower()
    return text


def preprocessor(text:iter):
    return Pipeline([('rm_spcl_char', FunctionTransformer(_rm_spcl_char))]).transform(text)
    


def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        joblib.dump(pickle.dumps(model), f)
        print(f"This model is saved at {model_path}.")


def load_model(model_path):
    print(f"This model is loaded from {model_path}.")
    return pickle.loads(joblib.load(model_path))


def save_result(result, result_path):
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f"Model results are saved at {result_path}.")