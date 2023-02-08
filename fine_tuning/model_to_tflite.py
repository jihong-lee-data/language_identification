import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
import warnings
from onnxruntime.quantization import quantize_dynamic, quantize_static

warnings.filterwarnings("ignore")

onnx_model_path = "onnx/model.onnx"
q_onnx_model_path = "onnx/quant_model.onnx"
tf_model_path = 'tf'
tflite_model_path = 'tflite/model.tflite'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("local-pt-checkpoint"):

    # Load tokenizer and PyTorch weights
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast = True)
    pt_model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=30).to(device)
    pt_model.load_state_dict(torch.load('test_trainer/checkpoint-96000/pytorch_model.bin', map_location=device))

    # Save to disk
    tokenizer.save_pretrained("local-pt-checkpoint")
    pt_model.save_pretrained("local-pt-checkpoint")
else:
    print('pretrained model already exists in local-pt-checkpoint')

# torch2onnx
if not os.path.exists(onnx_model_path):
    os.system('sh cvt2onnx.sh')
else:
    print('onnx model already exists')

# quantization
quantized_model = quantize_dynamic(onnx_model_path, q_onnx_model_path)
# quantize_model = quantize_static(onnx_model_path, q_onnx_model_path)

# Load the ONNX model
q_onnx_model = onnx.load(q_onnx_model_path)
# Check that the IR is well formed
onnx.checker.check_model(q_onnx_model)

tf_rep = prepare(q_onnx_model)

tf_rep.export_graph(tf_model_path)


#Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print('done')
