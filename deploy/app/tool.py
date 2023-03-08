import json
import os
import re

def load_json(path): 
    return json.load(open(path, "r"))

def save_json(file, path): 
    json.dump(file, open(path, "w"))

def mk_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    
def rm_spcl_char(text):
    # remove special characters
    text = re.sub(r'[!@#$(),，\n"%^*?？:;~`0-9&\[\]\。\/\.\=\-]', ' ', text)
    text = re.sub(r'[\s]{2,}', ' ', text.strip())
    text = text.lower().strip()
    
    return text