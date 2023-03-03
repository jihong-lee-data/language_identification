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

model.save_pretrained("model/best_model")