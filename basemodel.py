from module.engine import *
from pprint import pprint
import warnings
warnings.filterwarnings(action='ignore')

# config
dataset_dir = 'data/os_data_51'

model_name = "mnnb_os_51_wo_prep"
model_pkl_path = f"model/obj/{model_name}.pkl"
result_json_path = f"model/result/{model_name}.json"

vectorizer = CountVectorizer()
classifier = MultinomialNB()


# load dataset
dataset = load_from_disk(dataset_dir)

# feature preprocessing
x = dict()
y = dict()
for key in dataset.keys():
    x[key] = dataset[key]['text']
    # x[key] = preprocessor(dataset[key]['text'])
    y[key] = dataset[key]['labels']

# pipeline: feature vectorizing & model fitting
pipeline = Pipeline([('vect', vectorizer),
                     ('clf', classifier)])

pipeline.fit(x['train'], y['train'])


# calc model score & save result
results = dict()
for key in dataset.keys():
    y_pred = pipeline.predict(x[key])
    results[key] = dict()
    results[key]['acc'] = accuracy_score(y[key], y_pred)


save_model(pipeline, model_pkl_path)
save_result(results, result_json_path)
pprint(results)

print('Done!')
