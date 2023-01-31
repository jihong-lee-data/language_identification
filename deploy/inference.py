from module.engine import *

model = Model("best_estimator")


preds_id, probs = model.predict(["안녕하세요", 'greatest love of all'], n = 3)

preds = model.int2label(preds_id)

print([dict(zip(pred, prob.round(4))) for pred, prob in zip(preds, probs)])



