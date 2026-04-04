import numpy as np

from utils import (
    create_sample,
    Justification,
    Alignment,
    ColorGenerator,
    get_text_border_color_fast,
)
import cv2
import random
import os
from typing import Iterator, Optional
from faker import Faker
from tqdm import tqdm
from math import ceil

fake = Faker()


class Locale:
    def __init__(self, code: str, sep: Optional[str] = None):
        self.code = code
        self.sep = sep


class GeneratorConfig:
    def __init__(self, locales: list[Locale], font_files: list[str]):
        self.locales = locales
        self.font_files = font_files


class LocaleFontPair:
    def __init__(self, locale: Locale, font_file: str):
        self.locale = locale
        self.font_file = font_file


class LocaleDictValue:
    def __init__(self, locale: Locale, fonts: list[str]):
        self.locale = locale
        self.fonts = fonts
        self.index = 0

    def get(self):
        font = self.fonts[self.index]
        self.index = (self.index + 1) % len(self.fonts)
        return LocaleFontPair(self.locale, font)


class LocaleFontPairGenerator(Iterator[LocaleFontPair]):
    def __init__(self, configs: list[GeneratorConfig], count: int) -> None:
        self.count = count
        self.index = 0

        locale_dict: dict[str, LocaleDictValue] = {}
        for config in configs:
            for locale in config.locales:
                locale_dict[locale.code] = LocaleDictValue(locale, config.font_files)

        self.locale_dict = locale_dict
        self.locales = list(locale_dict.keys())
        self.len_locales = len(self.locales)

        if self.len_locales == 0:
            raise ValueError("At least one locale is required")

    def __iter__(self) -> "LocaleFontPairGenerator":
        return self

    def __next__(self) -> LocaleFontPair:
        if self.count is not None and self.index >= self.count:
            raise StopIteration

        locale_key = self.locales[self.index % self.len_locales]
        info = self.locale_dict[locale_key]
        self.index += 1
        return info.get()


class Background:
    def __init__(self, background: cv2.typing.MatLike):
        self.background = background

    def get(self, width: int, height: int, rand: random.Random):
        bh, bw, _ = self.background.shape
        if width > bw or height > bh:
            scale = max(float(width) / float(bw), height / float(bh))
            bh *= scale
            bw *= scale
            bh = ceil(bh)
            bw = ceil(bw)
            data = cv2.resize(self.background, (bw, bh))
        else:
            data = self.background

        diff_y = bh - height
        diff_x = bw - width
        offset_y = 0
        offset_x = 0

        if diff_y > 0:
            offset_y = rand.randrange(0, diff_y)

        if diff_x > 0:
            offset_x = rand.randrange(0, diff_x)

        return data[offset_y : offset_y + height, offset_x : offset_x + width]

    @staticmethod
    def file(file_path: str):
        return Background(cv2.imread(file_path, cv2.IMREAD_COLOR_BGR))

    @staticmethod
    def black(width=512, height=512, channels=3):
        return Background(np.full((height, width, channels), 0, dtype=np.uint8))

    @staticmethod
    def white(width=512, height=512, channels=3):
        return Background(np.full((height, width, channels), 255, dtype=np.uint8))


class SampleGenerator:
    def __init__(
        self,
        configs: Optional[list[GeneratorConfig]] = None,
        backgrounds: Optional[list[Background]] = None,
    ):
        self.configs = configs if configs is not None else []
        self.backgrounds = backgrounds if configs is not None else []

    def add_config(self, config: GeneratorConfig):
        self.configs.append(config)

    def create_distribution(self, count: int) -> LocaleFontPairGenerator:
        return LocaleFontPairGenerator(self.configs, count)

    def run(
        self,
        output_path: str,
        count: int = 200,
        seed: int = 0,
        max_image_size=(768, 768),
        min_image_size=(256, 256),
    ):
        white_bg = Background.white(max_image_size[0], max_image_size[1])
        black_bg = Background.black(max_image_size[0], max_image_size[1])
        working_backgrounds = [white_bg, black_bg] + self.backgrounds

        Faker.seed(seed)
        images_path = os.path.join(output_path, "images")
        labels_path = os.path.join(output_path, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        dist = self.create_distribution(count=count)

        fakers: dict[str, Faker] = {}

        for locale in set(dist.locales):
            fakers[locale] = Faker(locale=locale)

        rand = random.Random(seed)

        color_gen = ColorGenerator(seed)

        for i, item in tqdm(
            enumerate(dist), total=count, desc=f"Generating Samples => {output_path}"
        ):
            faker_inst = fakers[item.locale.code]
            justification = rand.choice(
                [Justification.Left, Justification.Right, Justification.Center]
            )
            alignment = rand.choice([Alignment.Top, Alignment.Center, Alignment.Bottom])
            while True:
                text = rand.choice(
                    [
                        lambda: faker_inst.sentence(),
                        lambda: faker_inst.address(),
                        lambda: faker_inst.company(),
                    ]
                )()

                image_x_size = rand.randrange(min_image_size[0], max_image_size[0])
                image_y_size = rand.randrange(min_image_size[1], max_image_size[1])

                image_x_size = image_x_size - (image_x_size % 8)
                image_y_size = image_y_size - (image_y_size % 8)

                background = rand.choice(working_backgrounds).get(
                    image_x_size, image_y_size, rand
                )

                text_color = rand.choice(
                    [
                        lambda: color_gen.next_bgr(),
                        lambda: (0, 0, 0),
                        lambda: (255, 255, 255),
                    ]
                )()
                # border_color = readable_background_from_bgr(text_color)
                border_color = get_text_border_color_fast(
                    text_color=text_color, draw_destination=background
                )
                if border_color is not None:
                    border_size = rand.randrange(1, 3)
                else:
                    border_size = 0

                # background_color = readable_background_from_bgr(text_color)
                sample = create_sample(
                    text,
                    font_file=item.font_file,
                    draw_area=background,
                    sep=item.locale.sep,
                    justification=justification,
                    alignment=alignment,
                    margin=rand.choice([0, 1, 2, 3, 4]),
                    text_color=text_color,
                    line_spacing=rand.randint(1, 3),
                    target_font_size=rand.randrange(20, 70),
                    outline_color=border_color,
                    outline_size=border_size,
                )

                if sample is None:
                    continue

                sample_name = f"{str(i).zfill(5)}"

                lines = []

                for contour in sample.contours:
                    contour = contour.squeeze(axis=1)

                    if len(contour.shape) != 2 or len(contour) < 3:
                        continue

                    coords = []
                    for x, y in contour:
                        coords.append(float(x) / image_x_size)
                        coords.append(float(y) / image_y_size)

                    line = "0 " + " ".join(f"{v:.6f}" for v in coords)
                    lines.append(line)

                cv2.imwrite(
                    os.path.join(images_path, f"{sample_name}.png"), sample.image
                )
                with open(
                    os.path.join(labels_path, f"{sample_name}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("\n".join(lines))

                break
