from datasets import load_from_disk
import numpy as np
import torch
import evaluate
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
# from transformers import AutoTokenizer

# dataset = load_from_disk("../model_development/data/wortschartz_30/")
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation= True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# tokenized_datasets.save_to_disk('data/tokenized/wortschartz_30')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    # loading dataset
    print("Loading dataset...")
    dataset = load_from_disk("data/tokenized/wortschartz_30/")
    print("Done")

    # loading base model
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30)
    model.to(device)
    
    print("device: ", model.device)


    metric = evaluate.load("accuracy")

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
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()