import onnx
import onnxruntime as ort
import warnings
import torch
import time
warnings.filterwarnings("ignore")

# LABELS = ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'sw', 'th', 'tl', 'tr', 'uk', 'vi', 'zh_cn', 'zh_tw']
# label_dict = dict(zip(range(len(LABELS)), LABELS))

# def get_max_n(values, n = 3):
#     max_n_idx = (-values).argsort()[:n]
#     max_n_labels, max_n_values = [], []
#     for idx in max_n_idx:
#         max_n_values.append(values[idx])
#         max_n_labels.append(label_dict[idx])
#     return max_n_labels, max_n_values

providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("onnx/model.onnx", providers=providers)
tokenizer=  torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'model/best_model/')

def main():
    while True:
        try:
            text= input('Enter text: ')
            start = time.time()
            encoding = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=512,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,   
                        return_attention_mask=True,
                        return_tensors='pt',
                    )

            logits = ort_session.run(
                None,
                {'input_ids': encoding.get('input_ids').detach().cpu().numpy(), 'attention_mask':  encoding.get('attention_mask').detach().cpu().numpy()})[0]
            
            probs = torch.softmax(torch.Tensor(logits), -1).detach().cpu().numpy()[0]
            
            # preds_n, probs_n = get_max_n(probs, n = 3)
            
            end = time.time()
            # print(dict(zip(preds_n, probs_n)))
            print(logits.argmax())
            print(f"Inference time: ({end - start:.5f} sec)", end = '\n'*2)
        except EOFError:
            print('Bye!')
            break
        
if __name__ == '__main__':
        main()

