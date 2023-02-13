from datasets import load_from_disk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AutoTokenizer
import os
import warnings

warnings.filterwarnings(action='ignore')

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__(),
      self.model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2024),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(2024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 30)
        )
      # x는 데이터를 나타냅니다.
    def forward(self, x):
      logits =self.model(x)
      output = F.softmax(logits, dim=1)
      return output

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        X = data['input_ids']
        y = data['labels']
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)
    
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            X = data['input_ids']
            y = data['labels']
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    torch.cuda.empty_cache()
    # loading dataset
    print("Loading dataset...")

    if os.path.exists("../fine_tuning/data/tokenized/wortschartz_30/"):
        dataset = load_from_disk("../fine_tuning/data/tokenized/wortschartz_30/")
    else:
        print("tokenizing dataset...")
        raw_dataset = load_from_disk("../model_development/data/wortschartz_30/")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation= True)

        dataset = raw_dataset.map(tokenize_function, batched=True)
        dataset.save_to_disk('data/tokenized/wortschartz_30')

    print("Done")

    train_size = len(dataset["train"])
    valid_size = len(dataset["validation"])

    print("Dataset size(train): ", train_size)
    print("Dataset size(validation): ", valid_size)


    model = Net()
    learning_rate = 1e-3
    batch_size = 64
    epochs = 1


    train_sampler = BatchSampler(RandomSampler(dataset['train'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)
    valid_sampler = BatchSampler(RandomSampler(dataset['validation'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)

    train_dataloader = DataLoader(dataset['train'], batch_sampler= train_sampler,  num_workers = 4)
    valid_dataloader = DataLoader(dataset['validation'], batch_sampler= valid_sampler, num_workers = 4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
    print("Done!")

    
    torch.cuda.empty_cache()	
if __name__ == "__main__":
    main()

