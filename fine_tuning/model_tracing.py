import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

device = torch.device('cpu')

script_model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30, torchscript=True)
script_model.load_state_dict(torch.load('test_trainer/checkpoint-96000/pytorch_model.bin', map_location=device))

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

example = 'flitto is the best company in Korea'

encoding = tokenizer.encode_plus(
    example,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,   
    return_attention_mask=True,
    return_tensors='pt',
).to(device)

traced_model = torch.jit.trace(script_model, [encoding["input_ids"], encoding["attention_mask"]])

torch.jit.save(traced_model, "traced_model.pt")

