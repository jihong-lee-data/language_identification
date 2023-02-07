from module.engine import *
from nltk.corpus import words
import time

model = Model("best_estimator")

while True:
    try:
        text= input('text: ')
        start = time.time()
        if not tokenizer(text):
            lang_pred_dict = {}
        elif text.strip() in words.words():
            lang_pred_dict = {'en': 1.00}
        else:    
            preds_id, probs = model.predict(text, n = 3)

            preds = model.int2label(preds_id)

            lang_pred_dict = dict(zip(preds, probs.round(4)))

        if lang_pred_dict is not None:
            json_str = json.dumps(lang_pred_dict)
            print(json_str)
        end = time.time()
        print(f"Inference time: ({end - start:.5f} sec)", end = '\n'*2)
    except EOFError:
        print("End!")
        break
