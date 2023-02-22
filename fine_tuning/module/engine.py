import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AutoTokenizer, RobertaForSequenceClassification

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

def save_state(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(config):
    model= FineTuning(config= config)
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


def train_loop(dataloader, model, loss_fn, optimizer, device, config):
    model.train()
    size= len(dataloader.dataset)
    n_steps= int(np.ceil(size / dataloader.batch_sampler.batch_size))
    n_logs= config['trainer']['n_logs']
    with tqdm(total= n_steps) as pbar:
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
            
            if batch % (n_steps // n_logs)== 0:
                print(f'===\nbatch {batch}\ntrained size: {trained_size}, train loss: {loss}\n===')
                save_state(model, config['model']['path']['checkpoint']['model'])
                save_state(optimizer, config['model']['path']['checkcpoint_dir'] / f"model_checkpoint_{trained_size}.pt")
            pbar.update(1)
    
    return loss
    

def test_loop(dataloader, model, loss_fn, device):
    size= len(dataloader.dataset)
    num_batches= len(dataloader)
    loss, correct= 0, 0
    model.eval()
    with torch.inference_mode():
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

class FineTuning(nn.Module):
    def __init__(self, config):
        super(FineTuning, self).__init__(),
        self.model= RobertaForSequenceClassification.from_pretrained("model/xlm-roberta-base")
        self.tokenizer= AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
        self.device= torch.device(config['device'])
        self.model.to(self.device)

    def forward(self, x):
        x= self.tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt').to(self.device)
        logits =self.model(x['input_ids'], attention_mask=x['attention_mask'])[0]
        output= F.softmax(logits, dim=1)
        return output
