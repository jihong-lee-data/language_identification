from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import numpy as np
import re
from module.engine import *
import pyprind


dataset = load_from_disk("data/wortschartz_30")

n_steps = 10
n_train_data = len(dataset['train'])
n_valid_data = len(dataset['validation'])
n_train_batch = int(n_train_data / n_steps)
n_valid_batch = int(n_valid_data / n_steps)

train_sampler = BatchSampler(RandomSampler(dataset['train']), batch_size = n_train_batch, drop_last = False)
valid_sampler = BatchSampler(RandomSampler(dataset['validation']), batch_size = n_valid_batch, drop_last = False)

train_dataloader = DataLoader(dataset['train'], batch_sampler = train_sampler, num_workers = 4)
valid_dataloader = DataLoader(dataset['validation'], batch_sampler = valid_sampler, num_workers = 4)



vectorizer = HashingVectorizer(alternate_sign=False, decode_error='ignore',
                                    n_features=2**22,
                                preprocessor=None,
                                tokenizer=tokenizer,
                                ngram_range= (1, 1)
                                )

classifier = MultinomialNB()

    
pipeline = Pipeline([('vect', vectorizer),
                    ('clf', classifier)])


model = pipeline


labels = dataset['train'].features['labels'].names



### mini batch 방식으로 모델 학습
pbar = pyprind.ProgBar(n_steps)

train_gen = iter(train_dataloader)
valid_gen = iter(valid_dataloader)
for _ in range(n_steps):
    print('1')
    current_train_batch = next(train_gen)
    current_valid_batch = next(valid_gen)
    
    print('2')
    
    X_train = model['vect'].partial_fit(current_train_batch['text']).transform(current_train_batch['text'])
    y_train = current_train_batch['labels']
    
    print('3')
    X_valid = model['vect'].transform(current_valid_batch['text'])
    y_valid = current_valid_batch['labels']

    print('4')
    model['clf'].partial_fit(X_train, y_train, classes = labels)
    
    print('5')
    print(f"train acc:{round(model['clf'].score(X_train, model['clf'].classes_[y_train]), 4)}")
    print(f"valid acc: {round(model['clf'].score(X_valid, model['clf'].classes_[y_valid]), 4)}")

    pbar.update()
    