from fastapi import FastAPI, Request, HTTPException
from module.engine import Model, tokenizer
import pickle
import numpy as np
import json
import nltk
from nltk.corpus import wordnet
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder


class JSON(BaseModel):
    text: str

    
app = FastAPI()

model = Model("best_estimator")

nltk.download('wordnet')


@app.post('/api/langid')
async def api_predict(request:JSON):
    # request_json = await request.json()
    request_json = jsonable_encoder(request)
    if request_json is None:
        return HTTPException(400)
    if 'text' not in request_json:
        return HTTPException(400)        
    text= request_json['text']
    if not tokenizer(text):
        lang_pred_dict = {}
    elif text.strip() in wordnet.words():
        lang_pred_dict = {'en': 1.00}
    else:    
        preds_id, probs = model.predict(text, n = 3)

        preds = model.int2label(preds_id)

        lang_pred_dict = dict(zip(preds, probs.round(4)))
    
    if lang_pred_dict is not None:
    #     json_str = json.dumps(lang_pred_dict)
        return jsonable_encoder(lang_pred_dict)
    
