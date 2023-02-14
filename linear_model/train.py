from datasets import load_from_disk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AutoTokenizer, pipeline, AutoModel
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

BASE_DIR = 'model'
if not os.path.exists(BASE_DIR):
    os.system(f'mkdir {BASE_DIR}')


device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

embedding_model = AutoModel.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__(),
        self.model = nn.Sequential()
        self.layers = (
          ('embedding', nn.Embedding.from_pretrained(embedding_model.embeddings.word_embeddings.weight).to(device)),
          ('pool', nn.AvgPool2d(kernel_size=(512, 1))),
          ('flat', nn.Flatten()),
          ('fc1', nn.Linear(768, 1024, device = device)),
          ('activ1', nn.ReLU()),
          ('fc2', nn.Linear(1024, 2024, device = device)),
          ('activ2', nn.ReLU()),
          ('dropout1', nn.Dropout(0.25)),
          ('fc3', nn.Linear(2024, 1024, device = device)),
          ('activ3', nn.ReLU()),
          ('fc4', nn.Linear(1024, 512, device = device)),
          ('activ4', nn.ReLU()),
          ('dropout2', nn.Dropout(0.5)),
          ('fc5', nn.Linear(512, 256, device = device)),
          ('activ5', nn.ReLU()),
          ('fc6', nn.Linear(256, 128, device = device)),
          ('activ6', nn.ReLU()),
          ('ouput', nn.Linear(128, 30, device = device))
        )
        for name, module in self.layers:
            self.model.add_module(name, module)
      # x는 데이터를 나타냅니다.
    def forward(self, x):
        x= tokenizer(x, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt')['input_ids'].to(device)
        logits =self.model(x)
        output = F.softmax(logits, dim=1)
        return output

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    n_step = int(np.ceil(size / dataloader.batch_sampler.batch_size))
    with tqdm(total= n_step) as pbar:
        for batch, data in enumerate(dataloader):
            X = data['text']
            y = torch.tensor(data['labels']).to(device)
            # 예측(prediction)과 손실(loss) 계산
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch % (n_step // 2) == 0) and (batch != 0):
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                check_point_dir = os.path.join(BASE_DIR, f'checkpoint-{epoch}-{batch}')
                if not os.path.exists(check_point_dir):
                    os.system(f'mkdir {check_point_dir}')
                torch.save(model.state_dict(), os.path.join(check_point_dir, 'model.bin'))
                torch.save(optimizer.state_dict(), os.path.join(check_point_dir, 'optimizer.pt'))
            
            pbar.update(1)

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(total= num_batches) as pbar:
            for data in dataloader:
                X = data['text']
                y = torch.tensor(data['labels']).to(device)
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

    dataset = load_from_disk("../model_development/data/wortschartz_30/")

    print("Done")

    train_size = len(dataset["train"])
    valid_size = len(dataset["validation"])

    print("Dataset size(train): ", train_size)
    print("Dataset size(validation): ", valid_size)


    model = Net()
    learning_rate = 1e-5
    batch_size = 128
    epochs = 5


    train_sampler = BatchSampler(RandomSampler(dataset['train'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)
    valid_sampler = BatchSampler(RandomSampler(dataset['validation'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)

    train_dataloader = DataLoader(dataset['train'], batch_sampler= train_sampler,  num_workers = 4)
    valid_dataloader = DataLoader(dataset['validation'], batch_sampler= valid_sampler, num_workers = 4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t+1)
        test_loop(valid_dataloader, model, loss_fn)

    print("Done!")

    
    torch.cuda.empty_cache()	
if __name__ == "__main__":
    main()

