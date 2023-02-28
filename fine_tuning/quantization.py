from transformers import RobertaForSequenceClassification, AutoTokenizer
from module.engine import load_model
from module.tool import load_json
import torch
import os 
from pathlib import Path

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

model = RobertaForSequenceClassification.from_pretrained('model/xlm-roberta-base')
model.load_state_dict(torch.load("model/xlm-roberta-finetune_v4_ep1/best_epoch/model.pt", map_location=torch.device('cpu')))

SAVE_PATH = Path("model/traced")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

backend = "qnnpack"
torch.backends.quantized.engine = backend


quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear,}, dtype=torch.qint8
)
quantized_model = torch.quantization.quantize_dynamic(
    quantized_model, {torch.nn.Embedding,}, dtype=torch.quint8
)

print_size_of_model(model)
print_size_of_model(quantized_model)

device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

example = 'Flitto is the best company in Korea'

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

traced_model = torch.jit.trace(quantized_model,  [encoding["input_ids"], encoding["attention_mask"]], strict=False)
torch.jit.save(traced_model, SAVE_PATH / "xlm-roberta-finetune_v4.pt")
