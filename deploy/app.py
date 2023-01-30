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
        return "Please send Post Request"
    elif request.method == "POST":
        data = request.get_json()
        
        text = data['text']
        
        prediction = model.predict([text])
        
        return str(prediction)
    
app.run(host='0.0.0.0')


import requests

url = "http://0.0.0.0:5000/api_predict"
data = {
        "text": "안녕하세요"
        }

r = requests.post(url, json = data)
print(r)
print(r.text)


