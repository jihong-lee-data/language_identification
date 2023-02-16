import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AutoTokenizer, AutoModel

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience= patience
        self.delta= delta
        self.counter= 0
        self.best_score= -np.inf
        self.early_stop= False
    def __call__(self, score):
        if score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score= score
            self.counter= 0
        if self.counter >= self.patience:
            self.early_stop = True            

def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(config):
    model= Net(embedding_model= AutoModel.from_pretrained(config['model']["embedding_model"]),
                tokenizer= AutoTokenizer.from_pretrained(config['model']["embedding_model"], use_fast=True),
                config= config,
                )
    return model

def load_trainer(model, config):
    loss_fn, optimizer, scheduler= None, None, None
    if config['trainer']['loss_fn'] == 'CrossEntropyLoss':
        loss_fn= nn.CrossEntropyLoss()
    if config['trainer']['optimizer'] == 'AdamW':
        optimizer= torch.optim.AdamW(model.parameters(), lr=config['trainer']['learning_rate'])
    if config['trainer']['scheduler'] == 'LambdaLR':
        scheduler= torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda= lambda epoch: config['trainer']['lr_lambda'] ** epoch,
                                                last_epoch=-1,
                                                verbose=False)
    return loss_fn, optimizer, scheduler

def get_dataloader(dataset, batch_size, num_workers= 4, seed= 42):
        batch_sampler= BatchSampler(RandomSampler(dataset, generator= np.random.seed(seed)), batch_size= batch_size, drop_last= False)
        return DataLoader(dataset, batch_sampler= batch_sampler,  num_workers= num_workers)


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, wandb):
    size= len(dataloader.dataset)
    n_step= int(np.ceil(size / dataloader.batch_sampler.batch_size))
    with tqdm(total= n_step) as pbar:
        for idx, data in enumerate(dataloader):
            batch= idx + 1
            X= data['text']
            y= torch.tensor(data['labels']).to(device)
            # 예측(prediction)과 손실(loss) 계산
            pred= model(X)
            loss= loss_fn(pred, y)
        
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, trained_size= loss.item(), batch * len(X)
            log_dict= dict(batch_loss= loss, trained_size= trained_size)
            wandb.log(log_dict, step= batch)  
            
            pbar.update(1)

    return loss
    

def test_loop(dataloader, model, loss_fn, device):
    size= len(dataloader.dataset)
    num_batches= len(dataloader)
    loss, correct= 0, 0
    with torch.no_grad():
        with tqdm(total= num_batches) as pbar:
            for data in dataloader:
                X= data['text']
                y= torch.tensor(data['labels']).to(device)
                pred= model(X)
                loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                pbar.update(1)

    loss /= num_batches
    correct /= size
    accuracy= 100*correct
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, accuracy

class Net(nn.Module):
    def __init__(self, embedding_model, tokenizer, config):
        super(Net, self).__init__(),
        self.embedding_layer= nn.Embedding.from_pretrained(embedding_model.embeddings.word_embeddings.weight)
        self.tokenizer= tokenizer
        self.device= torch.device(config['device'])
        self.model= nn.Sequential()
        self.layers= (
                        ('embedding', self.embedding_layer.to(self.device)),
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
                                            device=self.device))
                        )
        for name, module in self.layers:
            self.model.add_module(name, module)
        
        del self.embedding_layer
        

    def forward(self, x):
        x= self.tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt')['input_ids'].to(self.device)
        logits =self.model(x)
        output= F.softmax(logits, dim=1)
        return output


def _n_unit(n_layers, n_input, n_output, n_max, n_inc=0):
    n_dec= n_layers - n_inc
    if n_inc >= n_layers:
        raise ValueError("n_inc must be less than n_layer")
    if n_layers == 1:
        return [(n_input, n_output)]
    if n_inc == 0:
        n_max= n_input
    inc_layers= np.int64(np.round(np.exp2(np.linspace(np.log2(n_input), np.log2(n_max), n_inc+1))))     
    dec_layers= np.int64(np.round(np.exp2(np.linspace(np.log2(n_max), np.log2(n_output), n_dec+1))))
    io_list= np.hstack([inc_layers, dec_layers[1:]])
    return [(io_list[i], io_list[i+1]) for i in range(n_layers)]


def _gen_dropout(n_layers=1, n_dropout=0, rates:(float or list)= 0.2):
    if not n_dropout:
        return None
    layer2attach= np.linspace(1, n_layers, n_dropout+2, dtype= np.int32)[1:-1].tolist()
    if isinstance(rates, float):
        rates= [rates] * n_dropout
    
    return [layer2attach, [nn.Dropout(rates[i]) for i in range(n_dropout)]]


def _stack_fc(layer_io, dropouts=None, activ_func=nn.ReLU(), device=None):
    model= nn.Sequential()
    n_layer= len(layer_io)
    for idx, io in enumerate(layer_io):
        layer_id= idx + 1
        layer= nn.Sequential()
        if layer_id == n_layer:
            name, module= 'ouput', nn.Linear(io[0], io[1], device= device)
        else:
            components= [('lin', nn.Linear(io[0], io[1], device= device)), ('activ', activ_func)]
            if dropouts:
                if layer_id in dropouts[0]:
                    components.append(('dropout', dropouts[1][dropouts[0].index(layer_id)]))
            for c_name, c_module in components:
                layer.add_module(c_name, c_module)
            name, module= f'fc{layer_id}', layer                
        model.add_module(name, module)
    return model
