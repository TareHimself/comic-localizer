from transformers import (
    AutoTokenizer,
    GenerationMixin,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    BaseImageProcessor,
    PreTrainedTokenizer,
)
from .metrics import Metrics
import torch
import cv2
import albumentations as A


class _Model(VisionEncoderDecoderModel, GenerationMixin):
    pass


transform = A.Compose(
    [
        A.ToGray(),
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(min_height=224, min_width=224),
    ]
)


class MockPrediction:
    def __init__(self, label_ids, predictions):
        self.label_ids = label_ids
        self.predictions = predictions


def test(images: list[str], model_path: str):
    device = torch.device("cuda")
    feature_extractor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
        model_path, use_fast=True
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
    model = _Model.from_pretrained(model_path)
    model = model.to(device)

    input_images = [
        transform(image=cv2.imread(x, cv2.IMREAD_COLOR_RGB))["image"] for x in images
    ]
    processed = feature_extractor(input_images, return_tensors="pt")["pixel_values"].to(
        device
    )
    x = model.generate(processed, max_length=300).cpu()
    y = [tokenizer.decode(z, skip_special_tokens=False) for z in x]
    m = Metrics(tokenizer)
    print(y)
    a = tokenizer.encode("<lang_ja> Today is a good day")
    c = tokenizer.encode("<lang_ko> Today")
    print(tokenizer.batch_decode([a, c]))
    print(m.compute_metrics(MockPrediction(torch.tensor([a]), torch.tensor([c]))))
    b = tokenizer.decode(a)
    print(b)
    # mage = self.transform(image=image)["image"]
    # image = np.stack([image, image, image], axis=-1)  # uint8, HWC
    # features = self.feature_extractor(image,return_tensors="pt").pixel_values
    # trainer.train()
