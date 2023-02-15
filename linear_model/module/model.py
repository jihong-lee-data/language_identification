import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from module.engine import load_json

config = load_json("model_config.json")

_embedding_model = AutoModel.from_pretrained(config['model']["embedding_model"])
_tokenizer = AutoTokenizer.from_pretrained(config['model']["embedding_model"], use_fast=True)


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__(),
        self.model= nn.Sequential()
        self.device= device
        self.layers = (
          ('embedding', nn.Embedding.from_pretrained(_embedding_model.embeddings.word_embeddings.weight).to(device)),
          ('pool', nn.AvgPool2d(kernel_size=(512, 1))),
          ('flat', nn.Flatten()),
          ('fc', _stack_fc(layer_io= _n_unit(config["model"]['fc']['n_layers'],
                                             config["model"]['fc']['n_input'],
                                             config["model"]['fc']['n_output'],
                                             config["model"]['fc']['n_max'],
                                             config["model"]['fc']['n_inc']),
                            dropouts= _gen_dropout(config["model"]['dropout']['n_layers'],
                                                   config["model"]['dropout']['n_dropout'],
                                                    config["model"]['dropout']['rates']),
                            device=device))
        )
        for name, module in self.layers:
            self.model.add_module(name, module)
        
      # x는 데이터를 나타냅니다.
    def forward(self, x):
        x= _tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt')['input_ids'].to(self.device)
        logits =self.model(x)
        output = F.softmax(logits, dim=1)
        return output

def _n_unit(n_layers, n_input, n_output, n_max, n_inc=0):
    n_dec = n_layers - n_inc
    if n_inc >= n_layers:
        raise ValueError("n_inc must be less than n_layer")
    if n_layers == 1:
        return [(n_input, n_output)]
    if n_inc == 0:
        n_max = n_input
    inc_layers = np.int64(np.round(np.exp2(np.linspace(np.log2(n_input), np.log2(n_max), n_inc+1))))     
    dec_layers = np.int64(np.round(np.exp2(np.linspace(np.log2(n_max), np.log2(n_output), n_dec+1))))
    io_list = np.hstack([inc_layers, dec_layers[1:]])
    return [(io_list[i], io_list[i+1]) for i in range(n_layers)]


def _gen_dropout(n_layers=1, n_dropout=0, rates:(float or list) = 0.2):
    if not n_dropout:
        return None
    layer2attach = np.linspace(1, n_layers, n_dropout+2, dtype = np.int32)[1:-1].tolist()
    if isinstance(rates, float):
        rates = [rates] * n_dropout
    
    return [layer2attach, [nn.Dropout(rates[i]) for i in range(n_dropout)]]


def _stack_fc(layer_io, dropouts=None, activ_func=nn.ReLU(), device=None):
    model = nn.Sequential()
    n_layer = len(layer_io)
    for idx, io in enumerate(layer_io):
        layer_id = idx + 1
        layer= nn.Sequential()
        if layer_id == n_layer:
            name, module = 'ouput', nn.Linear(io[0], io[1], device = device)
        else:
            components = [('lin', nn.Linear(io[0], io[1], device = device)), ('activ', activ_func)]
            if dropouts:
                if layer_id in dropouts[0]:
                    components.append(('dropout', dropouts[1][dropouts[0].index(layer_id)]))
            for c_name, c_module in components:
                layer.add_module(c_name, c_module)
            name, module = f'fc{layer_id}', layer                
        model.add_module(name, module)
    return model