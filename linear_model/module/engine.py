import tqdm
import torch
import numpy as np
import os
import json

def load_json(path): 
    return json.load(open(path, "r"))

def save_json(file, path): 
    json.dump(file, open(path, "w"))

def train_loop(dataloader, model, loss_fn, optimizer, epoch, device):
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
