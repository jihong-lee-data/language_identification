import pandas as pd
from module.engine import *
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
import numpy as np

raw_data_path = "data/os_data_51.tsv"
output_dir = raw_data_path.split('.')

raw = pd.read_csv(raw_data_path)

data = raw.copy()

label_list = sorted(data['labels'].unique().tolist())

valid_size = 51 * 10000
test_size = 51 * 10000




ft = Features({'text': Value('string'), 'labels': ClassLabel(num_classes=51, names=label_list, id = label_list)})
raw_dataset = load_dataset("csv", data_files = "data/os_data_51.tsv", split = 'all', features = ft)

trainvalid_test = raw_dataset.train_test_split(test_size=test_size, shuffle= True, stratify_by_column= 'labels', generator = np.random.seed(42))
train_valid = trainvalid_test['train'].train_test_split(test_size=valid_size, shuffle=True, stratify_by_column= 'labels', generator = np.random.seed(42))


ttv_ds = datasets.DatasetDict({'train': train_valid['train'], 'validation': train_valid['test'], 'test': trainvalid_test['test']})

# save to disk
ttv_ds.save_to_disk(output_dir)
