from module.engine import *
from sklearn.random_projection import SparseRandomProjection

datasets = load_from_disk("data/wortschartz_30/")

model_name = 'vect_wortschartz_30_v5'

preprocessor = Pipeline(steps=[('vect', HashingVectorizer(alternate_sign=True, decode_error='ignore',
                                                        n_features=2**19,
                                                        preprocessor=None,
                                                        tokenizer=tokenizer,
                                                        ngram_range=(1, 7)
                                                        )),
                                ('trans', TfidfTransformer())])

configs= dict(preprocessor = str(preprocessor))

mk_path(f'model/{model_name}')

model = Model(model_name, preprocessor)
model.save_model()
save_results(configs, f'model/{model_name}/config.json')


print('fitting vectorizer')
start = time.time()
model.model.fit(datasets['train']['text'])
end = time.time()
print(f"took ({end - start:.5f} sec)", end = '\n'*2)

print('saving model structure')
model.save_model()

print('mk_dtm')
example = 'apple is not a fruit.'
start = time.time()
dtm = model.model['vect'].transform([example])
end = time.time()
print(f'dtm shape = {dtm.shape}')
print(f"took ({end - start:.5f} sec)", end = '\n'*2)

# start = time.time()
# rdc_dtm = model.model['dimrdc'].transform(dtm)
# end = time.time()
# print(f'reduced shape = {rdc_dtm.shape}')
# print(f"took ({end - start:.5f} sec)", end = '\n'*2)

print('done')


