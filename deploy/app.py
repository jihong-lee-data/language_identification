from module.engine import Model
from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# http://localhost:5000/api_predict
model = Model("best_estimator")

@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        msg = """
        Please send Post Request
        JSON
        { "text" : "text to identify"}

        """
        return msg
    elif request.method == "POST":
        data = request.get_json()
        
        text = data['text']

        preds_id, probs = model.predict(data['text'], n = 3)

        preds = model.int2label(preds_id)
       
        return str([dict(zip(pred, prob.round(4))) for pred, prob in zip(preds, probs)])

app.debug = True    
app.run(host='0.0.0.0')

