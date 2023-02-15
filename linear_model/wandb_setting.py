from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch
import numpy as np
from tqdm import tqdm
import os
from module.engine import train_loop, test_loop, load_json
from module.model import Net


import warnings
warnings.filterwarnings(action='ignore')

import wandb
import random
import json
# start a new wandb run to track this script

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = 'FC_1_v1'
if not os.path.exists(BASE_DIR):
    os.system(f'mkdir {BASE_DIR}')

#TODO
##refactoring

def main():
    config = load_json("model_config.json")

    wandb.init(
        # set the wandb project where this run will be logged
        project="lang_id",
        entity = "jihongleejihong",
        save_code = True,
        group = "DNN",
        job_type = "train",
        notes="test",
        tags= [config['model_name'], config['architecture'], config['dataset']],
        id = config["model_name"],
        sync_tensorboard = True,
        resume = False,
        config = config,
        name = config["model_name"]
    )

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

if __name__ == "main":
    main()






# simulate training
epochs = 1000
offset = random.random() / 5
best_val_acc = 0
best_epoch = 0
wandb.define_metric("train_loss", summary="min")
wandb.define_metric("val_acc", summary="max")
for epoch in range(2, epochs):
    val_acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
    # log metrics to wandb
    wandb.log({'train_loss': loss, 'val_acc': val_acc}, step=epoch)
    
    wandb.run.summary['best_val_acc'] = best_val_acc
    wandb.run.summary['best_epoch'] = best_epoch
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()