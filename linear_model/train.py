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

BASE_DIR = 'FC_3_v1'
if not os.path.exists(BASE_DIR):
    os.system(f'mkdir {BASE_DIR}')

def n_unit(n_layer, n_input, n_output, n_max, n_inc=0):
    n_dec = n_layer - n_inc
    if n_inc >= n_layer:
        raise ValueError("n_inc must be less than n_layer")
    if n_layer == 1:
        return [(n_input, n_output)]
    if n_inc == 0:
        n_max = n_input
    inc_layers = np.int64(np.round(np.exp2(np.linspace(np.log2(n_input), np.log2(n_max), n_inc+1))))     
    dec_layers = np.int64(np.round(np.exp2(np.linspace(np.log2(n_max), np.log2(n_output), n_dec+1))))
    io_list = np.hstack([inc_layers, dec_layers[1:]])
    return [(io_list[i], io_list[i+1]) for i in range(n_layer)]

def gen_dropout(n_layers, n_dropout, rates:(float or list) = 0.2):
    layer2attach = np.linspace(1, n_layers, n_dropout+2, dtype = np.int32)[1:-1].tolist()
    if isinstance(rates, float):
        rates = [rates] * n_dropout
    
    return [layer2attach, [nn.Dropout(rates[i]) for i in range(n_dropout)]]

def stack_fc(layer_io, dropouts=None, activ_func=nn.ReLU(), device=None):
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
          ('fc', stack_fc(layer_io= n_unit(1, 768, 30, 768, 0), dropouts= gen_dropout(3, 1, 0.2), device=device))
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
    learning_rate = 1e-1
    batch_size = 128
    epochs = 10
    print(model)

    train_sampler = BatchSampler(RandomSampler(dataset['train'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)
    valid_sampler = BatchSampler(RandomSampler(dataset['validation'], generator = np.random.seed(42)), batch_size = batch_size, drop_last = False)

    train_dataloader = DataLoader(dataset['train'], batch_sampler= train_sampler,  num_workers = 4)
    valid_dataloader = DataLoader(dataset['validation'], batch_sampler= valid_sampler, num_workers = 4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t+1)
        test_loop(valid_dataloader, model, loss_fn)
        scheduler.step()
    print("Done!")

    
    torch.cuda.empty_cache()	
if __name__ == "__main__":
    main()

