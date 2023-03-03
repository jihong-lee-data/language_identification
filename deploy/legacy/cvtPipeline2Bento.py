
import bentoml
import torch
from torch import nn
from torch.nn import functional as F
import os
import torch
import time
import warnings
warnings.filterwarnings(action='ignore')
from pathlib import Path
from module.tool import load_json
from module.pipeline_for_bento import LanguageIdentificationPipeline


MODEL_DIR = Path('model/best_model')


from transformers.pipelines import SUPPORTED_TASKS

from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, pipeline, AutoTokenizer


TASK_NAME = "langugage-identification"


PIPELINE_REGISTRY.register_pipeline(
    TASK_NAME,
    pipeline_class=LanguageIdentificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
    type= "text",
)


# TASK_DEFINITION = {
#     "impl": LanguageIdentificationPipeline,
#     "tf": (),
#     "pt": (RobertaForSequenceClassification,),
#     "default": {},
#     "type": "text",
# }
# SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

classifier = pipeline(
    task=TASK_NAME,
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_DIR),
    tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
)

print(classifier('안녕하세요', n=3))

bentoml.transformers.save_model(
    name= "xlm-roberta-finetune",
    pipeline=classifier,
    task_name= TASK_NAME,
    signatures={
        "__call__": {
            "batchable": True,
            "batch_dim": 0,
        }
    }
    )


loaded = bentoml.transformers.load_model("xlm-roberta-finetune:latest")

loaded("안녕하세요", n=3)