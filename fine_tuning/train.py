from datasets import load_from_disk
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import os




def main():
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # loading dataset
    print("Loading dataset...")
    
    if os.path.exists("data/tokenized/wortschartz_30/"):
        tokenized_datasets = load_from_disk("data/tokenized/wortschartz_30/")
    else:
        datasets = load_from_disk("../model_development/data/wortschartz_30/")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base").to(device)
        print("tokenizer device: ", tokenizer.device)

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation= True)

        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        tokenized_datasets.save_to_disk('data/tokenized/wortschartz_30')


    print("Done")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        global metric
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # loading base model
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30).to(device)
    print("model device: ", model.device)

    default_args = {
        "output_dir": "test_trainer",
        "num_train_epochs": 1,
        "log_level": "error",
        "per_device_train_batch_size":16,
        "per_device_eval_batch_size":16,
        "logging_steps": 200,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "use_mps_device": True,
        "seed": 42
    }

    training_args = TrainingArguments(**default_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()