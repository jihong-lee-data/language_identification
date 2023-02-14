from datasets import load_from_disk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings(action='ignore')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__(),
      self.model = nn.Sequential(
        nn.Linear(50, 512),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Dropout(0.25),
        # nn.Linear(2048, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 8192),
        # nn.ReLU(),
        # nn.Linear(8192, 4096),
        # nn.ReLU(),
        # nn.Dropout(0.25),
        # nn.Linear(4096, 2048),
        # nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 30)
       ).to(device)
      
      # x는 데이터를 나타냅니다.
    def forward(self, x):
      logits =self.model(x)
      output = F.log_softmax(logits, dim=1)
      return output

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)

def tokenize_function(text, device):
    return tokenizer(text, max_length= 50, padding="max_length", truncation= True, return_tensors='pt').to(device)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    with tqdm(total=size) as pbar:
        for batch, data in enumerate(dataloader):
            X = tokenize_function(data['text'], device)['input_ids'].type(torch.float32)
            y = data['labels'].to(device)
            # 예측(prediction)과 손실(loss) 계산
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            pbar.update(len(X))


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        with tqdm(total=num_batches) as pbar:
            for data in dataloader:
                X = tokenize_function(data['text'], device)['input_ids'].type(torch.float32)
                y = data['labels'].to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                pbar.update(1)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():

    

    print("device: ", device)
    torch.cuda.empty_cache()
    # loading dataset
    print("Loading dataset...")

    dataset = load_from_disk("../model_development/data/wortschartz_30")

    
    print("Done")

    train_size = len(dataset["train"])
    valid_size = len(dataset["validation"])

    print("Dataset size(train): ", train_size)
    print("Dataset size(validation): ", valid_size)


    model = Net()
    learning_rate = 1e-1
    batch_size = 128
    epochs = 3


    train_dataloader = DataLoader(dataset['train'], batch_size = batch_size, shuffle=True, num_workers = 4)
    valid_dataloader = DataLoader(dataset['validation'], batch_size = batch_size, shuffle= True, num_workers = 4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(valid_dataloader, model, loss_fn, device)
    print("Done!")

    
    torch.cuda.empty_cache()	
if __name__ == "__main__":
    main()

