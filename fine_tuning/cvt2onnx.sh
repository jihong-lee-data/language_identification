#!/bin/bash
 python -m transformers.onnx --model=local-pt-checkpoint --feature=sequence-classification onnx/  
