import json
import os
import re
from pathlib import Path

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

class ISO():
    def __init__(self):
        self.dict_path= Path('resource/iso.json')
        if not self.dict_path.exists():
            self.dict_path.parent.mkdir(parents=True, exist_ok=True)
            os.system(f"curl https://raw.githubusercontent.com/jihong-lee-data/gadgetbox/main/{str(self.dict_path)} > {str(self.dict_path)}")
        self.iso_dict= load_json(str(self.dict_path))
        self.search_list = [[i[0]] + i[1] for i in self.iso_dict["en_to_iso"].items()]   

    def __call__(self, text, tol= 2):
        return self._search(text, tol)

    def _search(self, text, tol = 2):
        results = []
        for en_id_pair in self.search_list:
            if any((self._word_validation(text, target, tol) for target in en_id_pair)):
                results.append(en_id_pair)
        return results


    def _word_validation(self, test:str, target:str, tol = 2):
        if tol == 0:
            return test in [target]
        elif tol == 1:
            return test.lower() in [target.lower()]
        elif tol == 2:
            return test.lower() in target.lower()