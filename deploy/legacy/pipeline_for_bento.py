
from transformers import Pipeline

class LanguageIdentificationPipeline(Pipeline):
    def _sanitize_parameters(self, n = 1, **kwargs):
        postprocess_kwargs= dict(n=n)
        return {}, {}, postprocess_kwargs

    def preprocess(self, text):
        encoding = self.tokenizer(text, add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        return encoding

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs, n=3):
        logits = model_outputs["logits"].squeeze()
        probs = logits.softmax(dim= -1).detach().cpu().numpy()
        indice_n = (-probs).argsort()[:n]
        pred_n = [self.model.config.id2label.get(idx) for idx in indice_n]
        prob_n = probs[indice_n]
        return dict(zip(pred_n, prob_n))