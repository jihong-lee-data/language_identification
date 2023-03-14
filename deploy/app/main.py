import numpy as np
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import nltk
from nltk.corpus import wordnet
from app.inference import Inference
from app.tool import remove_character, ISO

app = FastAPI()
app.model = Inference(device="cpu")
app.iso = ISO()
nltk.download("wordnet")


class LangIdRequest(BaseModel):
    text: str
    n: int = 3


class LangIdBatchRequest(BaseModel):
    text: Union[str, List[str]]
    n: int = 1


class LangIdBatchResponse(BaseModel):
    result: List[List[Dict[str, Any]]]


class ISOSearchRequest(BaseModel):
    query: str
    tol: int = 2
    code: str = ""


class ISOSearchResponse(BaseModel):
    result: List[Dict[str, Any]]


class ISOConvertRequest(BaseModel):
    src: str
    src_code: str = ""
    dst_code: str = "1"


class ISOConvertResponse(BaseModel):
    result: str


@app.get("/")
async def root():
    """
    The home page of the API.
    """
    return "Language Identification API"


@app.post("/api/langid/predict")
async def predict(request: LangIdRequest):
    """
    Predicts the language of a given text.

    Args:
        request (LangIdRequest): The request object containing the text to be predicted and other parameters.

    Returns:
        JSON: A JSON object containing the predicted language and its probability.
    """
    request_json = jsonable_encoder(request)
    text = request_json["text"]
    n = request_json["n"]
    try:
        text_prepped = remove_character(text, target="sde")
        lang_pred = (
            {}
            if not text_prepped
            else (
                {"en": 1.00}
                if text_prepped.strip() in wordnet.words()
                else None
            )
        )
        if not isinstance(lang_pred, dict):
            lang_pred = app.model.predict(text_prepped, n=n).pop(0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return jsonable_encoder(lang_pred)


@app.post("/api/langid/predictb")
async def predict_batch(request: LangIdBatchRequest):
    """Predicts the language of a batch of texts.

    Args:
        request (LangIdBatchRequest): The request object containing the texts to be predicted and other parameters.

    Returns:
        JSON: A JSON object containing the predicted languages and their probabilities.
    """
    try:
        request_json = jsonable_encoder(request)
        text = request_json["text"]
        n = np.clip(request_json["n"], a_min=1, a_max=len(app.model.label_dict))

        if isinstance(text, str):
            text = [text]

        if len(text) > 1000:
            return jsonable_encoder({"result": []})

        text_prepped = [
            remove_character(sample, target="sde") for sample in text
        ]

        lang_pred = np.array(
            [
                [{}]
                if not tmp_text
                else [
                    {
                        "predicion": "en",
                        "probability": 1.00,
                        "rank": 1,
                        "english": "English",
                        "korean": "영어",
                    }
                ]
                if tmp_text.strip() in wordnet.words()
                else [None]
                for tmp_text in text_prepped
            ]
        )

        text_for_model = (
            np.array(text)[(lang_pred == None).squeeze()].squeeze().tolist()
        )

        if text_for_model:
            model_pred = app.model.predict(text=text_for_model, n=n)

            model_result = np.array(
                [
                    [
                        {
                            "prediction": pred,
                            "probability": prob,
                            "rank": idx + 1,
                            "english": app.iso.convert(
                                src=pred, src_code="1", dst_code="e"
                            ),
                            "korean": app.iso.convert(
                                src=pred, src_code="1", dst_code="k"
                            ),
                        }
                        for idx, (pred, prob) in enumerate(row.items())
                    ]
                    for row in model_pred
                ]
            ).tolist()

            result = {
                "result": [
                    model_result.pop(0) if row == [None] else row.tolist()
                    for row in lang_pred
                ]
            }
        else:
            result = {"result": lang_pred.tolist()}

    except Exception as e:
        result = {"error": str(e)}

    return jsonable_encoder(result)


@app.post("/api/iso/search")
async def search(request: ISOSearchRequest):
    """Searches for a given query within a given ISO code table.

    Args:
        request (ISOSearchRequest): The request object containing the query, ISO code table, and tolerance level.

    Returns:
        JSON: A JSON object containing the search result or an error message.
    """
    try:
        request_json = jsonable_encoder(request)
        query = request_json["query"]
        code = request_json["code"]
        tol = request_json["tol"]
        result = app.iso(query=query, code=code, tol=tol)
        return jsonable_encoder({"result": result})
    except Exception as e:
        return jsonable_encoder({"error": str(e)})


@app.post("/api/iso/convert")
async def convert(request: ISOConvertRequest):
    """Converts a given text from one language code to another.

    Args:
        request (ISOConvertRequest): The request object containing the text, source code, and destination code.

    Returns:
        JSON: A JSON object containing the converted text.
    """
    request_json = jsonable_encoder(request)
    src = request_json["src"]
    src_code = request_json["src_code"]
    dst_code = request_json["dst_code"]

    if dst_code:
        dst_code = dst_code[0]

    try:
        result = jsonable_encoder(
            {
                "result": app.iso.convert(
                    src=src, src_code=src_code, dst_code=dst_code
                )
            }
        )
    except ValueError:
        result = jsonable_encoder({"result": []})

    return result
