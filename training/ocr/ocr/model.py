from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoModel,
    AutoConfig,
)


class OcrProcessor(TrOCRProcessor):
    def __init__(
        self,
        encoder_name: str = "google/vit-base-patch16-224",
        decoder_name: str = "ai-forever/mGPT",
    ):
        self.image_processor = AutoImageProcessor.from_pretrained(
            encoder_name, use_fast=True
        )
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            decoder_name, use_fast=True
        )
        super().__init__(self.image_processor, self.tokenizer)


# class OcrModel(VisionEncoderDecoderModel):
#     def __init__(self,encoder_name: str = "google/vit-base-patch16-224",decoder_name: str = "google-t5/t5-small"):
#         encoder = AutoModel.from_pretrained(encoder_name)
#         decoder = T5ForConditionalGeneration.from_pretrained(decoder_name)
#         super().__init__(encoder=encoder,decoder=decoder)
#         self.processor = OcrModelProcessor(encoder_name,decoder_name)
#         self.config.decoder_start_token_id = self.processor.tokenizer.pad_token_id if  self.processor.tokenizer.bos_token_id is None else  self.processor.tokenizer.bos_token_id
#         self.config.pad_token_id = self.processor.tokenizer.pad_token_id
#         self.config.eos_token_id = self.processor.tokenizer.eos_token_id

#         assert self.config.decoder_start_token_id is not None
#         assert self.config.eos_token_id is not None
#         assert self.config.pad_token_id is not None


def get_model(
    encoder_name: str = "google/vit-base-patch16-224",
    decoder_name: str = "FacebookAI/xlm-roberta-base",
    max_length=300,
    num_decoder_layers=None,
) -> tuple[VisionEncoderDecoderModel, OcrProcessor]:
    processor = OcrProcessor(encoder_name, decoder_name)
    # encoder
    encoder = AutoModel.from_pretrained(encoder_name)

    # decoder config: force the missing flags
    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    if num_decoder_layers is not None:
        decoder_config.num_hidden_layers = num_decoder_layers

    # decoder
    decoder = AutoModelForCausalLM.from_pretrained(decoder_name, config=decoder_config)

    # if num_decoder_layers is not None:
    #     print(decoder.roberta.encoder.layer)

    # combine
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.bos_token_id is None
        else processor.tokenizer.bos_token_id
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    assert model.config.decoder_start_token_id is not None
    assert model.config.eos_token_id is not None
    assert model.config.pad_token_id is not None
    return (model, processor)


if __name__ == "__main__":
    from transformers import logging

    logging.set_verbosity_error()
    model, processor = get_model()
    print(model.decoder.base_model)
    # encoded = processor(text="What are you doing here?",add_special_tokens=True)["input_ids"]
    # encoded2 = processor.tokenizer.encode()
    # print(processor.tokenizer.decode(encoded[0]),encoded)
