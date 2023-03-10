import numpy as np
from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import nltk
from nltk.corpus import wordnet
from app.inference import Inference
from app.tool import rm_spcl_char, ISO

app = FastAPI()
app.model = Inference(device='cpu')
app.iso = ISO()
nltk.download('wordnet')


class LangIdRequest(BaseModel):
    text: str
    n: int= 3

# class LangIdRequest(BaseModel):
#     text: Union[str, List[str]]
#     n: int = 1

class ISOSearchRequest(BaseModel):
    query: str
    tol: int = 2
    code: str = ''


class ISOConvertRequest(BaseModel):
    src: str
    src_code: str = ''
    dst_code: str = '1'


@app.get('/')
async def root():
    return 'Language Identification API'


@app.post('/api/langid')
async def predict(request: LangIdRequest):
    request_json = jsonable_encoder(request)
    text = request_json['text']
    n = request_json['n']
    
    
    text_prepped= rm_spcl_char(text)
    
    lang_pred = {} if not text_prepped else ({'en': 1.00} if text_prepped.strip() in wordnet.words() else None)
    
    if not isinstance(lang_pred, dict):
        lang_pred= app.model.predict(text_prepped, n= n)

    return jsonable_encoder(lang_pred) 
    
# ToDo - batch inference & new response formatting
@app.post('/api/langidb')
async def predict_b(request: LangIdRequest):
    request_json = jsonable_encoder(request)
    text = request_json['text']
    n = np.clip(request_json['n'], a_min=1, a_max=len(app.model.label_dict))
    dst_code = request_json['type']

    if isinstance(text, str):
        text = [text]

    if len(text) > 1000:
        return jsonable_encoder(dict(result=[]))

    lang_pred = np.array([
        {} if not rm_spcl_char(tmp_text) else ({'en': 1.00} if tmp_text.strip() in wordnet.words() else None)
        for tmp_text in text
    ])

    text_for_model = np.array(text)[lang_pred == None].tolist()

    if text_for_model:
        model_pred = app.model.predict(text=text_for_model, n=n)
        lang_pred[lang_pred == None] = model_pred

    result = dict(result=lang_pred.tolist()) if len(text) > 1 else lang_pred[0]

    if dst_code:
        dst_code = dst_code[0]


@app.post('/api/iso/search')
async def search(request: ISOSearchRequest):
    request_json = jsonable_encoder(request)
    query = request_json['query']
    code = request_json['code']
    tol = request_json['tol']

    return jsonable_encoder(dict(result=app.iso(query=query, code=code, tol=tol)))


@app.post('/api/iso/convert')
async def convert(request: ISOConvertRequest):
    request_json = jsonable_encoder(request)
    src = request_json['src']
    src_code = request_json['src_code']
    dst_code = request_json['dst_code']
    try:
        result= jsonable_encoder(dict(result=app.iso.convert(src=src, src_code=src_code, dst_code=dst_code)))
    except ValueError: 
        result= jsonable_encoder(dict(result=[]))

    return result