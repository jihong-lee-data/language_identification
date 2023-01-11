from module.engine import load_hf_dataset, rm_spcl_char
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import json

dataset = load_hf_dataset('papluca/language-identification/')


# feature preprocessing
data_dict = dict()
for key in dataset.keys():
    data_list = []
    for text in dataset[key]['text']:
        
        data_list.append(rm_spcl_char(text))
    data_dict[key] = data_list
    

# feature vectorizing & label encoding
from sklearn.feature_extraction.text import CountVectorizer
    
cv = CountVectorizer()
cv.fit(data_dict['train'])

le = LabelEncoder()
le.fit(dataset['train']['labels'])

x = dict()
y = dict()

for key in dataset.keys():
    x[key] = cv.transform(data_dict[key])
    y[key] = le.transform(dataset[key]['labels'])



# fitting model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x['train'], y['train'])


from sklearn.metrics import accuracy_score, confusion_matrix


results = dict()

for key in dataset.keys():
    y_pred = model.predict(x[key])
    results[key] = dict()
    results[key]['acc'] = accuracy_score(y[key], y_pred)
    # results[key]['cm'] = confusion_matrix(y[key], y_pred)

with open('mnnb_results.json', 'w') as f:
    json.dump(results, f)
