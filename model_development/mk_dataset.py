import pandas as pd
from module.engine import *
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
import numpy as np

raw_data_path = "data/wortschartz_30.parquet"
output_dir = raw_data_path.split('.parquet')[0]

mk_path(output_dir)

raw = pd.read_parquet(raw_data_path)
num_classes = raw['labels'].nunique()
data = raw.copy()

label_list = sorted(data['labels'].unique().tolist())

valid_size = num_classes * 10000
test_size = num_classes * 10000

print('size')
pprint(dict(train = data.shape[0] - valid_size - test_size, valid = valid_size, test = test_size))



ft = Features({'text': Value('string'), 'labels': ClassLabel(num_classes=num_classes, names=label_list, id = label_list)})
raw_dataset = load_dataset("parquet", data_files = raw_data_path, split = 'all', features = ft)

trainvalid_test = raw_dataset.train_test_split(test_size=test_size, shuffle= True, stratify_by_column= 'labels', generator = np.random.seed(42))
train_valid = trainvalid_test['train'].train_test_split(test_size=valid_size, shuffle=True, stratify_by_column= 'labels', generator = np.random.seed(42))


ttv_ds = datasets.DatasetDict({'train': train_valid['train'], 'validation': train_valid['test'], 'test': trainvalid_test['test']})

# save to disk
ttv_ds.save_to_disk(output_dir)
