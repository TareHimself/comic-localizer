import evaluate
from transformers.trainer_utils import EvalPrediction


class Metrics:
    def __init__(self, processor):
        self.cer_metric = evaluate.load("cer")
        self.processor = processor
        self.idx = 0

    def compute_metrics(self, pred: EvalPrediction):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids = pred_ids.copy()

        invalid = (pred_ids < 0) | (pred_ids >= len(self.processor.tokenizer))
        pred_ids[invalid] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        with open("./metrics_print.txt", "a", encoding="utf-8") as f:
            print(f"{self.idx}:: {cer} :: [{pred_str}] [{label_str}]", file=f)

        return {"cer": cer}
