from module.engine import *
from pprint import pprint

# config
dataset_dir = 'papluca/language-identification/'

model_name = "dt"
model_pkl_path = f"model/obj/{model_name}.pkl"
result_json_path = f"model/result/{model_name}.json"

vectorizer = CountVectorizer()
classifier = DecisionTreeClassifier()


# load dataset
dataset = load_hf_dataset(dataset_dir)

# feature preprocessing
x, y = preprocessor(dataset)

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
