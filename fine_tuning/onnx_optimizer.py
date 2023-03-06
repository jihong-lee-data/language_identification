from onnxruntime.transformers import optimizer

optimized_model = optimizer.optimize_model(
                                            "onnx/model.onnx",
                                            model_type='bert',
                                            num_heads=0,
                                            hidden_size=0,
                                            opt_level=2,
                                            use_gpu=True,
                                            )

optimized_model.convert_float_to_float16()
optimized_model.save_model_to_file("model_fp16.onnx")
