import numpy as np
import pandas as pd
import seaborn as sns
import os
import time
import json
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint
import gzip

import warnings
warnings.filterwarnings(action='ignore')


def get_time(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print(f"Done ({end - start:.5f} sec)", end = '\n'*2)
        return result
    return wrapper

def save_results(result, result_path):
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f"Model results are saved at {result_path}.")


def mk_path(path):
    if not os.path.isdir(path):
        os.makedirs (path)


def mk_confusion_matrix(save_path=None, y_true=None, y_pred=None, labels=None, figsize = (35, 30)):
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    
    plt.figure(figsize = figsize)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap = "OrRd", cbar = False)  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
     

def save_inference(save_path, x, y_true, y_pred):
    df_results = pd.DataFrame(np.column_stack([x, y_true, y_pred]), columns = ['text', 'label_true', 'label_pred'])
    df_results.to_csv(save_path, index = False)

class ISO():
    def __init__(self):
        with open('resource/iso.json', 'r') as f:
            self.iso_dict = json.load(f)
        self.search_list = [[i[0]] + i[1] for i in self.iso_dict["en_to_iso"].items()]   

    def search(self, text, tol = 2):
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

