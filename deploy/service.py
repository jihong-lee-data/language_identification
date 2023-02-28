from module.engine import Model, tokenizer
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import wordnet
import bentoml
from bentoml.io import JSON, NumpyNdarray


loaded_model = bentoml.sklearn.load_model("mnnb_v16:latest")
clf_runner = bentoml.sklearn.get("mnnb_v16:latest").to_runner()
label = loaded_model.classes_
label_dict = dict(zip(range(len(label)), label))

svc = bentoml.Service("language_identification", runners=[clf_runner])
@svc.api(input=JSON(), output=JSON())
def predict(input_json) -> np.ndarray:
     text = input_json['text']
     probs = clf_runner.predict_proba.run([text])[0]
     
     max_n_id = (-probs).argsort()[:3]
     preds= [label_dict[i] for i in max_n_id]
     probs= [round(probs[i], 4) for i in max_n_id]
     
     return dict(zip(preds, probs))
