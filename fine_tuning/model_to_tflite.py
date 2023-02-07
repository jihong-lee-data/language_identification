import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

os.command('sh cv2onnx.sh')

tf_model_path = 'tf'
tflite_model_path = 'tflite'
# Load the ONNX model
onnx_model = onnx.load("onnx/model.onnx")

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)

tf_rep.export_graph("model.tf")

#Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)