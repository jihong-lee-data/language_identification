from module.engine import *
from pprint import pprint
import warnings
warnings.filterwarnings(action='ignore')

@get_time
def main():
    # config
    dataset_dir = 'data'
    dataset_name = "wortschartz_idms"
    dataset_path = os.path.join(dataset_dir, dataset_name)
   

    # vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range = (2, 13))
    vectorizer = Pipeline([('vect', HashingVectorizer(alternate_sign=False, analyzer='char_wb', n_features= 2**22, ngram_range=(2, 10))), ('trans', TfidfTransformer())])
    classifier = XGBClassifier(n_estimators = 100, objective = 'multi:softprob',
                           max_depth = 3, learning_rate = 0.1, n_jobs = -1)


    model_version = 'v1'
    model_name= f"xgb_{dataset_name}_{model_version}"

    model_dir = os.path.join("model", model_name)
    result_dir = os.path.join(model_dir, "result")

    mk_path(model_dir)
    mk_path(result_dir)

    model_path = os.path.join(model_dir, f"model.pkl")
    config_path = os.path.join(result_dir, f"config.json")
    result_path = os.path.join(result_dir, f"results.csv")
    cm_path = os.path.join(result_dir, f"cm.png")


    configs = dict(dataset = dict(name = dataset_name, path = dataset_path),
            model = dict(name = model_name, path = model_path, result_path = result_path, cm_result_path = cm_path),
            vect = str(vectorizer),
            clf = str(classifier)
    )
    print('model configuration:')
    pprint(configs)

    # load dataset
    print('Loading dataset...')
    dataset = load_from_disk(dataset_path)
    print('Done.')

    # feature preprocessing
    print('Preprocessing features...')
    x = dict()
    y = dict()
    for key in dataset.keys():
        x[key] = dataset[key]['text']
        x[key] = preprocessor(dataset[key]['text'])
        y[key] = dataset[key].features['labels'].int2str(dataset[key]['labels'])
    print('Done.')

    # pipeline: feature vectorizing & model fitting
    print('Fitting model...')
    
    pipeline = Pipeline([('vect', vectorizer),
                        ('clf', classifier)])
    
    model = Model(model_name, model = pipeline)

    model.fit(x['train'], y['train'])

    print('Done.')
    
    
    # calc model score & save result
    print('Evaluating model...')
    configs['results'] = dict(acc = dict())
    y_pred = dict()
    for key in dataset.keys():
        y_pred[key] = model.predict(x[key])
        configs['results']['acc'][key] = accuracy_score(y[key], y_pred[key])

    pprint(configs['results'])


    print('Saving results...')
    # save_model(model, model_path)
    model.save_model()

    save_results(configs, config_path)
    save_inference(result_path, dataset['test']['text'], y['test'], y_pred['test'])
    mk_confusion_matrix(cm_path, y['test'], y_pred['test'], labels = dataset['test'].features['labels'].names)



if __name__ == '__main__':
    main()