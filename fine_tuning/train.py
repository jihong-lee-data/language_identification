import os
import numpy as np
import torch
from torch import nn
from datasets import load_from_disk
import wandb
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings(action='ignore')
from module.engine import get_dataloader, load_model, load_trainer, train_loop, test_loop, EarlyStopping, save_state
from module.tool import load_json, save_json


def main():
    device= torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # loading model configuration
    INIT_CONFIG_PATH= "model_config.json"
    config= load_json(INIT_CONFIG_PATH)
    config['device']= device.type

    # setting variout paths
    MODEL_DIR= Path('model')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR= MODEL_DIR / config['model_name']
    MODEL_CONFIG_PATH= SAVE_DIR / "model_config.json"
    
    DATA_DIR= Path("data/")
    DATA_PATH= DATA_DIR / config['dataset']
    
    CP_DIR= SAVE_DIR / "checkpoint"
    
    
    
    BEST_DIR= SAVE_DIR / "best_epoch"
    BEST_MODEL_PATH= BEST_DIR / "model.pt"
    BEST_OPTIM_PATH= BEST_DIR / "optimizer.pt"
    BEST_SCHDLR_PATH= BEST_DIR / "scheduler.pt"
    
    
    PRE_MODEL_PATH = MODEL_DIR / config['model']['base_model'] / "config.json"
    for path in [MODEL_DIR, SAVE_DIR, CP_DIR, BEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    

    config['model']['config'] = load_json(PRE_MODEL_PATH)
    config['model']['path'] = dict(checkcpoint_dir=CP_DIR,                                                
                                    best_epoch=dict(model=BEST_MODEL_PATH,
                                    optimizer=BEST_OPTIM_PATH,
                                    scheduler=BEST_SCHDLR_PATH
                                                )
    )

    # initiating wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="lang_id",
        entity= "jihongleejihong",
        save_code= True,
        group= "finetune",
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
    model=load_model(config)
    
    
    loss_fn, optimizer, scheduler= load_trainer(model, config)
    early_stopping= EarlyStopping(patience= config['trainer']['early_stop']['patience'], delta= config['trainer']['early_stop']['delta'])

    # training
    best_val_acc, best_epoch= 0, 0

    for crt_epoch in range(1, config['trainer']['epochs']+1):  
        print(f"Epoch {crt_epoch}\n-------------------------------")
        # train 
        train_loss= train_loop(train_dataloader, model, loss_fn, optimizer, device, config)
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
            save_state(model, BEST_MODEL_PATH)
            save_state(optimizer, BEST_MODEL_PATH)
            save_state(scheduler, BEST_MODEL_PATH)
            save_json(config, MODEL_CONFIG_PATH)
            # remove checkpoints generated during an epoch
            for path in CP_DIR.glob('*.pt'):
                path.unlink()
        
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
