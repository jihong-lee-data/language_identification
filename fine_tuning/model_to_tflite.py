import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch

onnx_model_path = "onnx/model.onnx"
tf_model_path = 'tf'
tflite_model_path = 'tflite'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and PyTorch weights
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast = True)
pt_model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30).to(device)
pt_model.load_state_dict(torch.load('test_trainer/checkpoint-96000/pytorch_model.bin', map_location=device))

# Save to disk
tokenizer.save_pretrained("local-pt-checkpoint")
pt_model.save_pretrained("local-pt-checkpoint")

os.system('sh cvt2onnx.sh')

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)

tf_rep.export_graph(tf_model_path)

#Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print('done')