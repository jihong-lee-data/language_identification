import os
import torch
import time
import warnings
warnings.filterwarnings(action='ignore')

from module.engine import load_model
from module.tool import load_json

LABELS = ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'sw', 'th', 'tl', 'tr', 'uk', 'vi', 'zh_cn', 'zh_tw']

device_available = dict(cuda= torch.cuda.is_available(), mps= torch.backends.mps.is_available(), cpu= True)

label_dict = dict(zip(range(len(LABELS)), LABELS))

model_name = 'FC4_v1'
MODEL_DIR= os.path.join('model', model_name)
MODEL_PATH = os.path.join(MODEL_DIR, 'checkpoint', 'model.pt')

def inference(model, text):
    model.eval()
    with torch.no_grad():
        probs= model(text)

    if isinstance(text, str):
        return label_dict[probs.argmax().detach().cpu().numpy().tolist()]
    return [label_dict[prob.argmax().detach().cpu().numpy().tolist()] for prob in probs]


    
def main():
    chosen_device = input(f"Enter device (available: {[key for key in device_available.keys() if device_available[key]]}, default: cpu)\n")
    if not chosen_device:
        chosen_device = 'cpu'
    
    device = torch.device(chosen_device)    

    print("Device: ", device)

    torch.cuda.empty_cache()

    model_config= load_json(os.path.join(MODEL_DIR, 'model_config.json'))
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