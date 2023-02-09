from module.engine import *
from sklearn.random_projection import SparseRandomProjection

print('load best model')
model = Model('mnnb_wortschartz_30_v16')

datasets = load_from_disk("data/wortschartz_30/")
print('done')

print('mk_dtm')
X_dtm= model.model['vect']['vect'].transform(datasets['train']['text'])
print('done')

print('dtm size', X_dtm.shape)

srp = SparseRandomProjection(n_components='auto', eps= 0.1, random_state=42, dense_output = True)

print('reducing size')
srp.fit(X_dtm)
print('done')

print('saving model structure')
model_name = 'gnb_wortschartz_30_v1'

pipeline = Pipeline([('prep',
                     Pipeline([('vect',
                                model.model['vect']['vect']),
                                ('dimrdc',
                                 srp)])),
                    ('clf', GaussianNB())])

gnb_model = Model(model_name, pipeline)


gnb_model.save_model()

print('done')