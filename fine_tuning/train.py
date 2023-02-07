from datasets import load_from_disk
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import os
import warnings
warnings.filterwarnings(action='ignore')


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    torch.cuda.empty_cache()
    # loading dataset
    print("Loading dataset...")

    if os.path.exists("data/tokenized/wortschartz_30/"):
        tokenized_datasets = load_from_disk("data/tokenized/wortschartz_30/")
    else:
        print("tokenizing dataset...")
        datasets = load_from_disk("../model_development/data/wortschartz_30/")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation= True)

        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        tokenized_datasets.save_to_disk('data/tokenized/wortschartz_30')

    print("Done")

    train_size = len(tokenized_datasets["train"])
    valid_size = len(tokenized_datasets["validation"])

    print("Dataset size(train): ", train_size)
    print("Dataset size(validation): ", valid_size)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # loading base model
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30).to(device)


    default_args = {
        "output_dir": "test_trainer",
        "num_train_epochs": 1,
        "log_level": "error",
        "per_device_train_batch_size":25,
        "per_device_eval_batch_size":25,
        "label_names": tokenized_datasets['train'].features['labels'].names,
        "logging_steps": 1000,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "dataloader_num_workers": 20,
        "seed": 42,
    }

    train_n_step = train_size / default_args["per_device_train_batch_size"]
    valid_n_step = valid_size / default_args["per_device_eval_batch_size"]

    default_args["save_steps"]= train_n_step // 5
    
    if device == 'mps':
        default_args['use_mps_device']= True
        default_args["dataloader_num_workers"]= 20

    training_args = TrainingArguments(**default_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    torch.cuda.empty_cache()	
if __name__ == "__main__":
    main()
