import pandas as pd
from module.engine import *
from inference import *
from datasets import load_from_disk
from tqdm import tqdm
from torch.utils.data import DataLoader

dataset = load_from_disk('../model_development/data/wortschartz_30/')['test']
train_dataloader = DataLoader(dataset, num_workers = 20)
train_gen = iter(train_dataloader)

print('data loaded.')
clf = Classifier()

print('running inference...')
result = []
for data in tqdm(train_gen):
    result.append(dict(text = data['text'][0], labels= label_dict[data['labels'][0].detach().cpu().numpy().tolist()], pred = label_dict[clf.classify(data['text'][0])[0].argmax()]))
pd.DataFrame(result).to_csv("test_preds.csv")
print('Saved')