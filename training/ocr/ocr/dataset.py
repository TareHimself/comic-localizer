import numpy as np
import os
import cv2
import albumentations as A
from torch.utils.data import Dataset


class OcrDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        max_length,
        processor,
        max_samples=-1,
        image_size=224,
        remove_lang_token=False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        files = os.listdir(dataset_path)
        self.images = list(sorted([x for x in files if x.endswith(".png")]))
        self.texts = list(sorted([x for x in files if x.endswith(".txt")]))
        if max_samples != -1:
            self.images = self.images[:max_samples]
            self.texts = self.texts[:max_samples]
        self.max_length = max_length
        self.processor = processor
        self.image_size = image_size
        self.remove_lang_token = remove_lang_token
        self.transform = A.Compose(
            [
                # A.InvertImg(p=0.05),
                # A.OneOf(
                #     [
                #         A.Downscale(
                #             (0.25, 0.5),
                #             interpolation_pair={
                #                 "downscale": cv2.INTER_LINEAR,
                #                 "upscale": cv2.INTER_LINEAR,
                #             },
                #         ),
                #         A.Downscale(
                #             (0.25, 0.5),
                #             interpolation_pair={
                #                 "downscale": cv2.INTER_NEAREST,
                #                 "upscale": cv2.INTER_NEAREST,
                #             },
                #         ),
                #     ],
                #     p=0.1,
                # ),
                # A.Blur(p=0.2),
                # A.Sharpen(p=0.2),
                # A.RandomBrightnessContrast(p=0.5),
                # A.GaussNoise(
                #     std_range=(0.1, 0.2), # 10-20% of max intensity
                #     per_channel=False,    # Faster, keeps noise "grayscale"
                #     p=0.2
                # ),
                # A.ImageCompression(
                #     quality_range=(40, 80), # 40-80 provides actual visible artifacts
                #     compression_type="jpeg",
                #     p=0.2
                # )
            ]
        )

    def __len__(self):
        return len(self.images)

    def get_image(self, index: int):
        image_item = self.images[index]
        image = cv2.imread(
            os.path.join(self.dataset_path, image_item), cv2.IMREAD_GRAYSCALE
        )
        image = self.transform(image=image)["image"]
        image = np.stack([image, image, image], axis=-1)  # uint8, HWC
        features = self.processor(image, return_tensors="pt").pixel_values
        return features.squeeze()

    def get_text(self, index: int):
        text_item = self.texts[index]

        with open(
            os.path.join(self.dataset_path, text_item), "r", encoding="utf-8"
        ) as f:
            text = f.read()
            if self.remove_lang_token:
                text = text[text.find(">") + 1 :].strip()
            return text

    def __getitem__(self, idx):
        image = self.get_image(idx)
        text = self.get_text(idx)

        encoding = {
            "pixel_values": image,
            "labels": text,
        }
        return encoding
