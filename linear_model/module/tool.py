import json
import os

def load_json(path): 
    return json.load(open(path, "r"))

def save_json(file, path): 
    json.dump(file, open(path, "w"))

def mk_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    