from module.engine import Model, tokenizer
import requests
import numpy as np
import timeit
import pandas as pd
import time 
print('Start')
def request_lang_id(url, text):
    response =  requests.post(url, json={'text': text})
    return response.json()

old_model_url = 'http://127.0.0.1:5000/api/langid'
new_model_url = 'http://127.0.0.1:3000/api/langid'

# test_data = pd.read_csv("data/test_data/lang_detect_test.csv")
test_data = pd.read_excel("../model_development/data/test_data/lang_detection_short_texts.xlsx")


start = time.time()
test_data['old'] = [request_lang_id(old_model_url, text) for text in test_data['lang_code']]
end = time.time()
print(f"Old model done! ({end - start:.5f} sec)", end = '\n'*2)

start = time.time()
test_data['new'] = [request_lang_id(new_model_url, text) for text in test_data['lang_code']]
end = time.time()
print(f"New model done! ({end - start:.5f} sec)", end = '\n'*2)

test_data.to_csv("model_comparison.csv", index = False)

print('Done')