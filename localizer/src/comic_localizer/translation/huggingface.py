from comic_localizer.core.plugin import (
    LanguageStringArgument,
    PytorchDevicePluginArgument,
    Translator,
)
import torch
from transformers import pipeline
from comic_localizer.core.plugin import (
    TranslatorResult,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)
import asyncio
from comic_localizer.utils import (
    get_default_language,
    get_default_torch_device,
    standardize_language_code,
)


class HuggingFaceTranslator(Translator):
    """Translates using hugging face models"""

    def __init__(
        self,
        model_url: str = "Helsinki-NLP/opus-mt-ja-en",
        input_language: str = "ja",
        output_language: str = get_default_language(),
        device: torch.device = get_default_torch_device(),
    ) -> None:
        super().__init__()
        self.pipeline = pipeline(
            "translation",
            model=model_url,
            device=device,
        )
        # src_lang/tgt_lang are only meaningful for multilingual tokenizers
        # (mBART/M2M100/NLLB/etc.) that define _build_translation_inputs.
        # MarianMT (the default model) doesn't, and transformers silently
        # ignores them for it, so only pass them when the tokenizer supports it.
        self._supports_lang_pair = hasattr(
            self.pipeline.tokenizer, "_build_translation_inputs"
        )
        self.input_language = input_language
        self.output_language = standardize_language_code(output_language)
        # if torch.cuda.is_available():
        #     self.pipeline.cuda()
        # elif torch.backends.mps.is_available():
        #     self.pipeline.to('mps')

    def predict(self, batch: list[OcrResult]):
        with torch.inference_mode():
            extra = (
                {"src_lang": self.input_language, "tgt_lang": self.output_language}
                if self._supports_lang_pair
                else {}
            )
            results = self.pipeline([x.text for x in batch], **extra)
            return results

    async def translate(self, batch: list[OcrResult]):
        # return [print(y) for y in self.pipeline([x.text for x in batch])]
        results = await asyncio.to_thread(self.predict, batch)

        return [
            TranslatorResult(y["translation_text"], language=self.output_language)
            for y in results
        ]

    @staticmethod
    def get_name() -> str:
        return "Hugging Face"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:

        return [
            StringPluginArgument(
                id="model_url",
                name="Model",
                description="The Hugging Face translation model to use",
                default="Helsinki-NLP/opus-mt-ja-en",
            ),
            LanguageStringArgument(
                id="input_language",
                name="Input Language",
                description="The language to translate from",
                default="ja",
            ),
            LanguageStringArgument(
                id="output_language",
                name="Output Language",
                description="The language to translate to",
            ),
            PytorchDevicePluginArgument("device", "Device"),
        ]
