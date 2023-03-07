from module.inference import Inference
from module.tool import rm_spcl_char

from fastapi import FastAPI, Request, HTTPException
import numpy as np
import json
import nltk
from nltk.corpus import wordnet
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

class JSON(BaseModel):
    text: str

    
app= FastAPI()

app.model= Inference()

nltk.download('wordnet')

@app.get('/')
async def root():
    return 'Language Identification API'
    
@app.post('/api/langid')
async def predict(request:JSON):
    request_json= jsonable_encoder(request)
    
    text= request_json['text']
    if not rm_spcl_char(text):
        lang_pred_dict= {}
    elif text.strip() in wordnet.words():
        lang_pred_dict= {'en': 1.00}
    else:    
        lang_pred_dict= app.model.predict(text, n= 3)
    if lang_pred_dict is not None:
        return jsonable_encoder(lang_pred_dict)
    
