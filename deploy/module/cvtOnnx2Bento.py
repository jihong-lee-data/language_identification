
import bentoml
import torch
import torch
import warnings
warnings.filterwarnings(action='ignore')
import onnx

# ort_session = ort.InferenceSession("onnx/model.onnx")
onnx_model= onnx.load("onnx/model.onnx")

tokenizer=  torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'model/best_model/')

signatures = {
    "run": {"batchable": True},
}

bentoml.onnx.save_model("lang_id", onnx_model, signatures=signatures)