# from transformers import logging

# logging.set_verbosity_error()
from typing import Optional
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    ProcessorMixin,
)
from .dataset import OcrDataset
from .metrics import Metrics
from .model import get_model
import torch


class DataColator:
    def __init__(self, max_length: int, processor: ProcessorMixin):
        self.max_length = max_length
        self.processor = processor

    def __call__(self, data: list[dict]):
        images = torch.stack([x["pixel_values"] for x in data]).float()

        all_labels = [x["labels"] for x in data]
        labels_batch = self.processor.tokenizer(
            all_labels,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels = labels_batch.masked_fill(
            labels_batch == self.processor.tokenizer.pad_token_id, -100
        )
        # for x, length in zip(data, sizes):
        #     ids = x["labels"]  # treat labels as the raw token ids sequence
        #     # pad decoder inputs with pad_token_id
        #     ids_padded = F.pad(
        #         ids, (0, pad_to - length), value=self.processor.tokenizer.pad_token_id
        #     )
        #     input_ids.append(ids_padded)

        #     # labels are same but pad with -100 to ignore loss
        #     lab_padded = F.pad(ids, (0, pad_to - length), value=-100)
        #     labels.append(lab_padded)

        # #input_ids = torch.stack(input_ids).long()  # [B, T]
        # labels = torch.stack(labels).long()  # [B, T]

        # attention_mask: 1 where not pad, 0 where pad
        # attention_mask = (input_ids != self.tokenizer.pad_token_id).long()  # [B, T]

        return {
            "pixel_values": images,
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "labels": labels,
        }


def train(
    train_dataset_path,
    eval_dataset_path,
    checkpoint: Optional[str] = None,
    output_path="./output/ocr",
    run_name="debug",
    encoder_name="facebook/dino-vits8",
    decoder_name="FacebookAI/xlm-roberta-base",
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=8,
    fp16=True,
    max_length=300,
    test=False,
    remove_lang_token=False,
):
    # if checkpoint is not None:
    #     model = OcrModel(encoder_name,decoder_name)
    #     processor = model.processor

    #     processor.tokenizer.add_tokens(["<lang_en>", "<lang_ja>", "<lang_ko>"])

    #     # Then resize
    #     model.decoder.resize_token_embeddings(len(processor.tokenizer))

    #     model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    #     model.config.pad_token_id = processor.tokenizer.pad_token_id
    #     # Set Correct vocab size.
    #     model.config.vocab_size = model.config.decoder.vocab_size
    #     model.config.eos_token_id = processor.tokenizer.sep_token_id
    # else:
    #     OcrModel.from_pretrained()

    model, processor = get_model(
        encoder_name, decoder_name, max_length=max_length, num_decoder_layers=2
    )

    if not remove_lang_token:
        processor.tokenizer.add_tokens(["<lang_en>", "<lang_ja>", "<lang_ko>"])
        # Then resize
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Set Correct vocab size.
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4
    model.generation_config.max_new_tokens = max_length

    train_dataset = OcrDataset(
        train_dataset_path,
        max_length=max_length,
        processor=processor,
        max_samples=1 if test else -1,
        remove_lang_token=remove_lang_token,
    )

    # lengths = [len(train_dataset[i]["labels"]) for i in range(len(train_dataset))]
    # print("p50", np.percentile(lengths, 50))
    # print("p90", np.percentile(lengths, 90))
    # print("p95", np.percentile(lengths, 95))
    # print("max", max(lengths))

    # print(train_dataset[0])
    eval_dataset = (
        train_dataset
        if test
        else OcrDataset(
            eval_dataset_path,
            max_length=max_length,
            processor=processor,
            remove_lang_token=remove_lang_token,
        )
    )

    # p = train_dataset[0]
    # model.save_pretrained("./ocr-en-ko-test2")

    collator = DataColator(max_length, processor)

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=1,
        predict_with_generate=True,
        generation_num_beams=4,
        # generation_max_length=max_length,
        # torch_compile=True, needs triton which has no wheels for windows
        eval_strategy="epoch",
        save_strategy="best",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=8,
        output_dir=output_path,
        logging_steps=10,
        # save_steps=3,
        learning_rate=learning_rate,
        # weight_decay=0.05,
        # save_steps=20000,
        # eval_steps=20000,
        # max_grad_norm=1.0,  # ADDED gradient clipping
        num_train_epochs=num_epochs,
        run_name=f"name: {run_name}, encoder: {encoder_name}, decoder: {decoder_name}",
        report_to="wandb",
        # auto_find_batch_size=True
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
