import os
import torch
import time
import warnings
warnings.filterwarnings(action='ignore')
from pathlib import Path
from module.engine import load_model
from module.tool import load_json

device_available = dict(cuda= torch.cuda.is_available(), mps= torch.backends.mps.is_available(), cpu= True)

MODEL_DIR = Path('model')
model_name = 'xlm-roberta-finetune_v2'
MODEL_PATH = MODEL_DIR / model_name / "checkpoint" / "model_checkpoint_1488000.pt"
MODEL_PATH = "temp.p"
def inference(model, text):
    model.eval()
    with torch.no_grad():
        probs= model(text)

    if isinstance(text, str):
        return model.model.config.id2label.get(probs.argmax().detach().cpu().numpy().tolist())
    return [model.model.config.id2label.get(prob.argmax().detach().cpu().numpy().tolist()) for prob in probs]


    
def main():
    chosen_device = input(f"Enter device (available: {[key for key in device_available.keys() if device_available[key]]}, default: cpu)\n")
    if not chosen_device:
        chosen_device = 'cpu'
    
    device = torch.device(chosen_device)    

    print("Device: ", device)

    torch.cuda.empty_cache()

    # model_config= load_json(os.path.join(MODEL_DIR, 'model_config.json'))
    model_config= load_json('model_config.json')
    model_config['device'] = device
    model = load_model(model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    
    while True:
        try:
            text= input('Enter text: ')
            start = time.time()
            pred = inference(model, text)
            end = time.time()
            print(pred)
            print(f"Inference time: ({end - start:.5f} sec)", end = '\n'*2)
            torch.cuda.empty_cache()
        except EOFError:
            print('Bye!')
            break

if __name__ == '__main__':
    
    main()