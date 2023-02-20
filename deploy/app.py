from module.engine import Model, tokenizer
from flask import Flask, request, Response, abort
import pickle
import numpy as np
import json
import nltk
from nltk.corpus import wordnet

app = Flask(__name__)

# http://localhost:5000/api_predict
model = Model("best_estimator")

nltk.download('wordnet')

@app.route('/api/langid', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        msg = """
        Please send Post Request
        JSON
        { "text" : "text to identify"}

        """
        return msg
    elif request.method == "POST":
        request_json = request.get_json(silent=True, cache=False, force=True)
        if request_json is None:
            return abort(400)
        if 'text' not in request_json:
            return abort(400)        

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
            json_str = json.dumps(lang_pred_dict)
            return Response(json_str, mimetype='application/json')
       
        

app.debug = True    
app.run(host='0.0.0.0', port=8000)

