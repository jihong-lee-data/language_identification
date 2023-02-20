import os
import numpy as np
import torch
from torch import nn
from datasets import load_from_disk
import wandb
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore')

from module.engine import get_dataloader, load_model, load_trainer, train_loop, test_loop, EarlyStopping, save_checkpoint
from module.tool import load_json, save_json, mk_dir


def main():
    device= torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # loading model configuration
    INIT_CONFIG_PATH= "model_config.json"
    config= load_json(INIT_CONFIG_PATH)
    config['device']= device.type

    # setting variout paths
    MODEL_DIR= 'model'
    SAVE_DIR= os.path.join(MODEL_DIR, config['model_name'])
    MODEL_CONFIG_PATH= os.path.join(SAVE_DIR, "model_config.json")


    DATA_DIR= "../model_development/data/"
    DATA_PATH= os.path.join(DATA_DIR, config['dataset'])

    CP_DIR= os.path.join(SAVE_DIR, "checkpoint")
    CP_MODEL_PATH= os.path.join(CP_DIR, "model.pt")
    CP_OPTIM_PATH= os.path.join(CP_DIR, "optimizer.pt")
    CP_SCHDLR_PATH= os.path.join(CP_DIR, "scheduler.pt")
    
    for path in [MODEL_DIR, SAVE_DIR, CP_DIR]:
        mk_dir(path)
    
    # initiating wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="lang_id",
        entity= "jihongleejihong",
        save_code= True,
        group= "DNN",
        job_type= "train",
        notes="test",
        tags= [config['model_name'], config['architecture'], config['dataset']],
        id= config["model_name"] + f"_{datetime.now().strftime('%y%m%d%H%M%S')}",
        sync_tensorboard= False,
        resume= 'allow',
        config= config,
        name= config["model_name"]
    )

    # loading dataset
    print("Loading dataset...")
    dataset= load_from_disk(DATA_PATH)
    print("Dataset size(train): ", len(dataset["train"]))
    print("Dataset size(validation): ", len(dataset["validation"]))
    train_dataloader= get_dataloader(dataset['train'], batch_size= config['trainer']['batch_size'], num_workers= 4, seed= 42)
    valid_dataloader= get_dataloader(dataset['validation'], batch_size= config['trainer']['batch_size'], num_workers= 4, seed= 42)
    print("Done")

    # loading model and training modules
    model= load_model(config)
    
    # loading pretrained weight only when base model is declared
    if config['trainer'].get('base_model'):
        BM_PATH = os.path.join(MODEL_DIR, config['trainer']['base_model'], "checkpoint", "model.pt")
        model.load_state_dict(torch.load(BM_PATH, map_location=device))
    
    loss_fn, optimizer, scheduler= load_trainer(model, config)
    early_stopping= EarlyStopping(patience= config['trainer']['early_stop']['patience'], delta= config['trainer']['early_stop']['delta'])

    # training
    best_val_acc, best_epoch= 0, 0

    for crt_epoch in range(1, config['trainer']['epochs']+1):  
        print(f"Epoch {crt_epoch}\n-------------------------------")
        # train 
        train_loss= train_loop(train_dataloader, model, loss_fn, optimizer, device)
        # validation
        val_loss, val_acc= test_loop(valid_dataloader, model, loss_fn, device)
                    

        log_dict= dict(train_loss= train_loss,
                        val_loss= val_loss,
                        val_acc= val_acc)
        print(log_dict)
        wandb.log(log_dict, step=crt_epoch)

        torch.cuda.empty_cache()	
        
        if val_acc > best_val_acc:
            best_val_acc= val_acc
            best_epoch= crt_epoch
    
        # check early stopping condition
        early_stopping(score= -val_loss)
        
        if early_stopping.early_stop:
            break
        elif not early_stopping.counter:
            save_checkpoint(model, CP_MODEL_PATH)
            save_checkpoint(optimizer, CP_OPTIM_PATH)
            save_checkpoint(scheduler, CP_SCHDLR_PATH)
            save_json(config, MODEL_CONFIG_PATH)

        scheduler.step()
        
    torch.cuda.empty_cache()
    
    # wrapping up & finishing wandb
    wandb.run.summary['best_epoch']= best_epoch
    wandb.run.summary['best_val_acc']= best_val_acc
    wandb.run.summary['stop_epoch']= crt_epoch
    wandb.run.summary['stop_val_loss']= val_loss
    wandb.run.summary['early_stop']= early_stopping.early_stop
    
    wandb.finish()
    
    print("Done!")

if __name__== "__main__":
    main()
