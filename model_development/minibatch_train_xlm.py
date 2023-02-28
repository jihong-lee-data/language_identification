from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import numpy as np
import re
from module.engine_xlm import *
from tqdm import tqdm


def main():
    # config
    dataset_dir = 'data'
    dataset_name = "wortschartz_31"
    clf_type = "xlm_gnb"
    model_version = 'v1'
    dataset_path = os.path.join(dataset_dir, dataset_name)
    
    classifier = GaussianNB()

    model_name= f"{clf_type}_{dataset_name}_{model_version}"
    model = Model(model_name)
    
    model_dir = os.path.join("model", model_name)
    result_dir = os.path.join(model_dir, "result")

    mk_path(model_dir)
    mk_path(result_dir)

    model_path = os.path.join(model_dir, f"model.pkl")
    config_path = os.path.join(result_dir, f"config.json")
    metric_path = os.path.join(result_dir, f"metric.json")
    result_path = os.path.join(result_dir, f"results.csv")
    cm_path = os.path.join(result_dir, f"cm.png")


    configs = dict(dataset = dict(name = dataset_name, path = dataset_path),
                    model = dict(name = model_name, path = model_path, result_path = result_path, cm_result_path = cm_path, embedding = embedding_model_name),
                    
                    clf = str(classifier)
                    )

    # load dataset
    dataset = load_from_disk(dataset_path)

    configs['train_info'] = dict(
                                epochs= 10,
                                n_steps = 500,
                                n_logs = 10,
                                n_train_data = len(dataset['train']),
                                n_valid_data = len(dataset['validation']),
                                )
    configs['train_info'].update(dict(
                                n_train_batch = int(configs['train_info']['n_train_data'] / configs['train_info']['n_steps']),
                                n_valid_batch = int(configs['train_info']['n_valid_data'] / configs['train_info']['n_steps'])
                                ))
                                
    print('model configuration:')
    pprint(configs)

    
    train_sampler = BatchSampler(RandomSampler(dataset['train'], generator = np.random.seed(42)), batch_size = configs['train_info']['n_train_batch'], drop_last = False)
    valid_sampler = BatchSampler(RandomSampler(dataset['validation'], generator = np.random.seed(42)), batch_size = configs['train_info']['n_valid_batch'], drop_last = False)

    train_dataloader = DataLoader(dataset['train'], batch_sampler = train_sampler, num_workers = 4)
    valid_dataloader = DataLoader(dataset['validation'], batch_sampler = valid_sampler, num_workers = 4)

    ### mini batch 방식으로 모델 학습
    pipeline = Pipeline([('clf', classifier)])

    model.model=pipeline

    model.labels = dataset['train'].features['labels'].names

    int2label = dataset['train'].features['labels'].int2str
    
    print('Fitting model...')
    
    for epoch in range(1, configs['train_info']['epochs']+1):
        train_gen = iter(train_dataloader)
        valid_gen = iter(valid_dataloader)
        print("---------- epoch %d ----------" % epoch)
        for step in tqdm(range(1, configs['train_info']['n_steps']+1)):

            # crt_train_batch = dataset['train'][next(train_gen)]
            # crt_valid_batch = dataset['validation'][next(valid_gen)]
            crt_train_batch = next(train_gen)
            crt_valid_batch =next(valid_gen)
                    
            X_train = pass_embedding(crt_train_batch['text'])
            y_train = int2label(crt_train_batch['labels'])        
        
            
            model.model['clf'].partial_fit(X_train, y_train, classes = model.labels)
            
            if (step % (configs['train_info']['n_steps'] // configs['train_info']['n_logs'])) == 0:
                X_valid = pass_embedding(crt_valid_batch['text'])
                y_valid = int2label(crt_valid_batch['labels'])
                
                configs['train_info'][f'step_{step}'] = dict(train_acc = model.model['clf'].score(X_train, y_train), 
                                                            valid_acc = model.model['clf'].score(X_valid, y_valid))
                print('')
                print(f"train acc:{round(configs['train_info'][f'step_{step}']['train_acc'], 4)}")
                print(f"valid acc: {round(configs['train_info'][f'step_{step}']['valid_acc'], 4)}")
        
            
    model.save_model()

    print('Done.')


    # # calc model metric & save result
    # print('Evaluating model...')
    # metric = dict(acc = dict(), report = dict())
    # y_pred = dict()
    # for key in dataset.keys():
    #     y_pred[key] = model.model.predict(pass_embedding(dataset[key]['text']))
    #     metric['acc'][key] = accuracy_score(int2label(dataset[key]['labels']), y_pred[key])
    #     metric['report'][key] = classification_report(int2label(dataset[key]['labels']), y_pred[key])
    
    # pprint(metric['acc'])

    # print('Saving results...')

    # save_results(configs, config_path)
    # save_results(metric, metric_path)

    # save_inference(result_path, dataset['test']['text'], int2label(dataset['test']['labels']), y_pred['test'])
    # mk_confusion_matrix(cm_path, int2label(dataset['test']['labels']), y_pred['test'], labels = model.labels)
        
if __name__ == "__main__":
    main()
