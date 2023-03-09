from app.inference import Inference
from app.tool import rm_spcl_char
from fastapi import FastAPI
import nltk
from nltk.corpus import wordnet
from pydantic import BaseModel
from typing import Union, List
from fastapi.encoders import jsonable_encoder
import numpy as np

class Request(BaseModel):
    text: Union[str, List[str]]
    n: int= 1
    

app= FastAPI()

app.model= Inference(device='cpu')

nltk.download('wordnet')


@app.get('/')
async def root():
    return 'Language Identification API'


@app.post('/api/langid')
async def predict(request: Request):
    request_json = jsonable_encoder(request)
    text = request_json['text']
    n = request_json['n']

    if isinstance(text, str):
        text = [text]
    
    lang_pred = np.array([
                          {} if not rm_spcl_char(tmp_text) else ({'en': 1.00} if tmp_text.strip() in wordnet.words() else None)
                           for tmp_text in text
                        ])
    
    text_for_model = np.array(text)[lang_pred == None].tolist()

    if text_for_model:
        model_pred = app.model.predict(text=text_for_model, n=n)
        lang_pred[lang_pred == None] = model_pred

    result = dict(result=lang_pred.tolist())
    return jsonable_encoder(result)
