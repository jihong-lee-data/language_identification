from module.engine import *
from scipy.special import softmax

def soft_voting(probs:list, weight:list=None):
    n_list = len(probs)
    if not weight:
        weight = [1] * n_list
    
    averaging = np.sum([softmax(prob, axis= 1) * w/np.sum(weight) for prob, w in zip(probs, weight)], axis = 0)
    
    return averaging.argmax(axis = 1)


def main():
    print('Loading models...')
    model_for_all = Model("mnnb_wortschartz_30_v13/")
    model_for_idms = Model("mnnb_wortschartz_idms_v2/")
    print('Done')
    iso_dict = ISO().iso_dict
    while True:
        X = [input("input text: ")]

        
        pred_id, prob = model_for_all.predict(X, prob = True)
        pred_lang = model_for_all.labels[pred_id]
        pred_lang

        if not pred_lang in ['id', 'ms']:
            response_id = pred_lang[0]
        else:
        
            _, prob_id_idms = model_for_idms.predict(X, prob = True)
            response_id = model_for_idms.labels[soft_voting([prob[:, [12, 16]], prob_id_idms], weight = [95, 97])][0]
        
        print(iso_dict['id'][response_id].split(';')[0])


if __name__ == '__main__':
    main()