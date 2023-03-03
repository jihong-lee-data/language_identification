from transformers import RobertaForSequenceClassification, AutoTokenizer
from module.engine import load_model
from module.tool import load_json
import torch
import os 
from pathlib import Path
import platform

device = torch.device('cuda')

processor = platform.processor().lower()
if 'x86' in  processor:
    backend= 'fbgemm'
elif 'arm' in  processor:
    backend= 'qnnpack'

MODEL_FILENAME = f"xlm-roberta-finetune_v4_{processor}_{device.type}.pt"

print(MODEL_FILENAME)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

model = RobertaForSequenceClassification.from_pretrained('model/xlm-roberta-base').to(device)
model.load_state_dict(torch.load("model/xlm-roberta-finetune_v4_ep1/best_epoch/model.pt", map_location=device))

SAVE_PATH = Path("model/traced")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# backend = "qnnpack"
torch.backends.quantized.engine = backend


# model.qconfig = torch.quantization.get_default_qconfig(backend)
# model.
# torch.backends.quantized.engine = backend
# model_static_quantized = torch.quantization.prepare(model, inplace=False)
# quantized_model = torch.quantization.convert(model_static_quantized, inplace=False)

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear,}, dtype=torch.qint8
)
quantized_model = torch.quantization.quantize_dynamic(
    quantized_model, {torch.nn.Embedding,}, dtype=torch.quint8
)

print_size_of_model(model)
print_size_of_model(quantized_model)


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
torch.jit.save(traced_model, SAVE_PATH / MODEL_FILENAME)
