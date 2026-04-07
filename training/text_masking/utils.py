import colorsys
import cv2
from cv2.typing import MatLike
import numpy as np
from typing import (
    Optional,
    Sequence,
    Tuple,
)
from PIL import Image, ImageDraw, ImageFont
import pyphen
from enum import Enum
import functools


class WrappedLine:
    def __init__(self, words: list[str], offset: float, height: float = 0):
        self.words = words
        self.offset = offset
        self.height = height

    def add_word(self, word: str, word_height: float):
        self.words.append(word)
        self.height = max(self.height, word_height)


class WrapResult:
    def __init__(self, lines: list[WrappedLine], bounds: tuple[int, int]):
        self.lines = lines
        self.bounds = bounds


def bbox_to_rect(bbox: tuple[float, float, float, float]):
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


class LayoutCache:
    def __init__(self, font: ImageFont.FreeTypeFont):
        self.font = font
        self.cache = {}

    def get(self, text: str) -> tuple[float, float, float, float]:
        if text in self.cache:
            return self.cache[text]

        self.cache[text] = bbox_to_rect(self.font.getbbox(text))

        return self.cache[text]


class HyphenationCache:
    def __init__(
        self, hyphenator: pyphen.Pyphen, wrap: float, layout_cache: LayoutCache
    ):
        self.hyphenator = hyphenator
        self.cache = {}
        self.wrap = wrap
        self.layout_cache = layout_cache

    def add_dashes_to_hypenations(self, hyphenation: tuple[str, str]):
        return [
            (f"{hyphenation[0]}-", self.layout_cache.get(f"{hyphenation[0]}-")),
            (hyphenation[1], self.layout_cache.get(hyphenation[1])),
        ]

    def filter_out_impossible(self, hyphenations: list[list[str]]):
        # x = list(map(lambda a: list(map(lambda item: [item[0],item[1][2]],a)),hyphenations))
        return filter(
            lambda hyp: max(hyp, key=lambda item: item[1][2])[1][2] <= self.wrap,
            hyphenations,
        )

    def get(
        self, text: str
    ) -> list[list[tuple[str, tuple[float, float, float, float]]]]:
        if text in self.cache:
            return self.cache[text]

        self.cache[text] = list(
            self.filter_out_impossible(
                [
                    [(text, self.layout_cache.get(text))],
                    *map(self.add_dashes_to_hypenations, self.hyphenator(text)),
                ]
            )
        )

        return self.cache[text]


def has_white(image: np.ndarray):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0


def wrap_text_pure(
    text: str,
    font: ImageFont.FreeTypeFont,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    sep: Optional[str] = None,
) -> Optional[WrapResult]:
    layout_cache = LayoutCache(font=font)
    _, _, space_width, _ = layout_cache.get(" ")

    if sep is not None and " " not in sep:
        space_width = 0

    text_list = list(text) if sep == "" else text.split(sep=sep)
    text_bounds = list(map(lambda a: (a, layout_cache.get(a)), text_list))
    x_offset = 0
    # Text too big to fit on a line
    if any(map(lambda a: a[1][2] > wrap_width, text_bounds)):
        return None

    x_offset = 0
    line_idx = 0
    lines = [WrappedLine([], 0)]
    x_bounds = 0
    for word, bbox in text_bounds:
        x_end = x_offset + bbox[2]

        if x_end > wrap_width:
            last_line = lines[-1]
            lines.append(
                WrappedLine([], last_line.offset + last_line.height + line_spacing)
            )
            line_idx += 1

            x_bounds = max(x_bounds, x_offset)

            x_offset = 0
            x_end = bbox[2]

        lines[line_idx].add_word(word, bbox[3])
        x_offset = min(x_end + space_width, wrap_width)
        x_bounds = max(x_bounds, x_offset)

    last_line = lines[-1]
    return WrapResult(lines, (x_bounds, last_line.offset + last_line.height))


def compute_word_bounds_and_hyphens(
    word: str, font: ImageFont.FreeTypeFont, hyphenator: pyphen.Pyphen
):
    result = [[(word, bbox_to_rect(font.getbbox(word)))]]
    for hyphenated in hyphenator.iterate(word):
        result.append(list(map(lambda a: a, hyphenated)))


def wrap_text_with_hyphenator(
    text: str,
    font: ImageFont.FreeTypeFont,
    hyphenator: pyphen.Pyphen,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    sep: Optional[str] = None,
) -> Optional[WrapResult]:
    layout_cache = LayoutCache(font=font)
    hyphenation_cache = HyphenationCache(
        hyphenator=hyphenator, layout_cache=layout_cache, wrap=wrap_width
    )

    _, _, space_width, _ = layout_cache.get(" ")

    if sep is not None and " " not in sep:
        space_width = 0

    text_list = list(text) if sep == "" else text.split(sep=sep)
    all_word_versions = list(map(hyphenation_cache.get, text_list))

    # No versions means the word cant fit at this font size
    if any(map(lambda a: len(a) == 0, all_word_versions)):
        return None

    # we know one version of the word will fit, we just need to find the version
    def fit_best_version(
        lines: list[WrappedLine],
        versions: list[list[tuple[str, tuple[float, float, float, float]]]],
        x_offset: float,
        x_bounds: float,
    ):
        nonlocal wrap_width
        nonlocal space_width
        nonlocal line_spacing
        line_idx = len(lines) - 1

        selected_version = versions[0]
        version_part_index = 0
        # if we are at a new line we can skip this section
        if x_offset != 0:
            for version in versions:
                word_partial, bbox = version[version_part_index]
                x_end = x_offset + bbox[2]

                if x_end <= wrap_width:
                    lines[line_idx].add_word(word_partial, bbox[3])
                    x_bounds = max(x_bounds, x_end)
                    version_part_index += 1
                    selected_version = version
                    x_offset = x_end + space_width
                    break

        if version_part_index < len(selected_version):
            # now we start fitting a new line

            if len(lines[line_idx].words) > 0:
                last_line = lines[-1]
                lines.append(
                    WrappedLine([], last_line.offset + last_line.height + line_spacing)
                )
                line_idx += 1

            x_offset = 0

            for version_part in selected_version[version_part_index:]:
                word_partial, bbox = version_part

                x_end = x_offset + bbox[2]

                if x_end > wrap_width:
                    last_line = lines[-1]
                    lines.append(
                        WrappedLine(
                            [], last_line.offset + last_line.height + line_spacing
                        )
                    )
                    line_idx += 1

                    x_bounds = max(x_bounds, x_offset)

                    x_offset = 0
                    x_end = bbox[2]

                lines[line_idx].add_word(word_partial, bbox[3])
                x_offset = min(x_end + space_width, wrap_width)

                x_bounds = max(x_bounds, x_offset)

        return x_bounds, x_offset

    x_offset = 0
    x_bounds = 0
    lines = [WrappedLine([], 0)]

    for versions in all_word_versions:
        new_bounds, new_offset = fit_best_version(lines, versions, x_offset, x_bounds)
        x_bounds = new_bounds
        x_offset = new_offset

    last_line = lines[-1]
    return WrapResult(lines, (x_bounds, last_line.offset + last_line.height))


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    wrap_width: float = float("inf"),
    line_spacing: float = 2,
    sep: Optional[str] = None,
    hyphenator: Optional[pyphen.Pyphen] = None,
) -> Optional[WrapResult]:
    return (
        wrap_text_pure(
            text=text,
            font=font,
            wrap_width=wrap_width,
            line_spacing=line_spacing,
            sep=sep,
        )
        if hyphenator is None
        else wrap_text_with_hyphenator(
            text=text,
            font=font,
            wrap_width=wrap_width,
            line_spacing=line_spacing,
            hyphenator=hyphenator,
            sep=sep,
        )
    )


class FontFitResult:
    def __init__(self, font_size: float, wrap: WrapResult):
        self.font_size = font_size
        self.wrap = wrap


def find_next_test(min_size, max_size):
    return min_size + ((max_size - min_size) // 2)


@functools.lru_cache()
def load_font(file: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(file, size)


# @perf
def find_best_font_size(
    text: str,
    font_file: str,
    size: tuple[int, int],
    font_size=30,
    min_font_size=10,
    max_font_size=20,
    tolerance=1,
    line_spacing: float = 2,
    sep: Optional[str] = None,
    hyphenator: Optional[pyphen.Pyphen] = None,
    stop_if_fit: bool = True,
) -> Optional[FontFitResult]:
    current_size = min(font_size, max_font_size)
    current_max = max_font_size
    current_min = min_font_size
    best = None
    while current_min <= current_max:
        font = load_font(font_file, current_size)
        wrap_result = wrap_text(
            text=text,
            font=font,
            wrap_width=size[0],
            hyphenator=hyphenator,
            line_spacing=line_spacing,
            sep=sep,
        )
        if wrap_result is not None and wrap_result.bounds[1] <= size[1]:
            best = FontFitResult(current_size, wrap_result)
            current_min = current_size + 1
            next_font_size = find_next_test(current_min, current_max)

            if stop_if_fit or abs(best.font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size
        else:
            current_max = current_size - 1
            next_font_size = find_next_test(current_min, current_max)
            best_font_size = current_size if best is None else best.font_size

            if abs(best_font_size - next_font_size) < tolerance:
                break

            current_size = next_font_size

    return best


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def readable_background_from_bgr(
    bgr: Tuple[int, int, int],
    hue_shift: float = 0.5,  # 0.5 = 180 degrees
    min_lightness_gap: float = 0.45,
) -> Tuple[int, int, int]:
    """
    AI Made
    """
    b, g, r = bgr

    # BGR -> normalized RGB
    r_n = r / 255.0
    g_n = g / 255.0
    b_n = b / 255.0

    # RGB -> HLS
    # colorsys uses HLS, where:
    # H = hue [0,1], L = lightness [0,1], S = saturation [0,1]
    h, l, s = colorsys.rgb_to_hls(r_n, g_n, b_n)  # noqa: E741

    # Shift hue so background color varies instead of becoming black/white
    bg_h = (h + hue_shift) % 1.0

    # Force a strong lightness contrast
    if l >= 0.5:
        # text is light, make background dark
        bg_l = max(0.08, l - min_lightness_gap)
    else:
        # text is dark, make background light
        bg_l = min(0.92, l + min_lightness_gap)

    # Keep some color, but avoid oversaturated ugly pairings
    if s < 0.15:
        # grayscale-ish text: invent some background color
        bg_s = 0.35
    else:
        # slightly tame saturation
        bg_s = min(0.85, max(0.25, s * 0.8))

    # Back to RGB
    bg_r, bg_g, bg_b = colorsys.hls_to_rgb(bg_h, bg_l, bg_s)

    # RGB -> BGR 0-255
    return (
        int(round(bg_b * 255)),
        int(round(bg_g * 255)),
        int(round(bg_r * 255)),
    )


def get_text_border_color_fast(
    text_color: tuple[int, int, int],
    draw_destination: np.ndarray,
    min_channel_diff: int = 30,
    max_similar_fraction: float = 0.10,
    sample_stride: int = 6,
) -> Optional[tuple[int, int, int]]:
    """
    Fast border decision for complex backgrounds.

    Returns:
        None            if border is not needed
        (b, g, r) tuple if border is needed
    """

    sampled = draw_destination[::sample_stride, ::sample_stride]

    text_b = text_color[0]
    text_g = text_color[1]
    text_r = text_color[2]

    # int16 avoids uint8 underflow on subtraction
    sampled = sampled.astype(np.int16, copy=False)

    diff_b = np.abs(sampled[..., 0] - text_b)
    diff_g = np.abs(sampled[..., 1] - text_g)
    diff_r = np.abs(sampled[..., 2] - text_r)

    max_diff = np.maximum(diff_b, np.maximum(diff_g, diff_r))

    similar_fraction = np.count_nonzero(max_diff < min_channel_diff) / max_diff.size

    if similar_fraction <= max_similar_fraction:
        return None

    brightness = (text_b * 29 + text_g * 150 + text_r * 77) >> 8
    return (0, 0, 0) if brightness >= 128 else (255, 255, 255)


class Sample:
    def __init__(
        self, image: np.ndarray, mask: np.ndarray, contours: Sequence[MatLike]
    ):
        self.image = image
        self.mask = mask
        self.contours = contours


class Justification(Enum):
    Left = 0
    Center = 1
    Right = 2


class Alignment(Enum):
    Top = 0
    Center = 1
    Bottom = 2


class ColorGenerator:
    def __init__(self, seed: float = 0.0):
        self.h = seed

    def next_bgr(self) -> Tuple[int, int, int]:
        GOLDEN_RATIO = 0.618033988749895

        self.h = (self.h + GOLDEN_RATIO) % 1.0

        # keep saturation/value in good ranges
        s = 0.6
        v = 0.85

        r, g, b = colorsys.hsv_to_rgb(self.h, s, v)

        return (
            int(b * 255),
            int(g * 255),
            int(r * 255),
        )


def create_sample(
    text: str,
    font_file: str,
    draw_area: np.ndarray,
    text_color=(0, 0, 0),
    outline_color=(255, 255, 255),
    margin=3,
    line_spacing: float = 2,
    outline_size: int = 0,
    hyphenator: Optional[pyphen.Pyphen] = None,
    min_font_size=10,
    max_font_size=100,
    contours_min_thresh=200,
    target_font_size=20,
    sep: Optional[str] = None,
    justification: Justification = Justification.Center,
    alignment: Alignment = Alignment.Center,
) -> Optional[Sample]:

    dest = draw_area
    mask = np.zeros_like(dest)

    frame_h, frame_w, _ = draw_area.shape

    fit_result = find_best_font_size(
        text,
        font_file,
        (frame_w - (margin * 2), frame_h - (margin * 2)),
        target_font_size,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        line_spacing=line_spacing + (outline_size * 2),
        hyphenator=hyphenator,
        sep=sep,
    )

    if fit_result is None:
        return None

    as_pil = cv2_to_pil(dest)
    as_pil_mask = cv2_to_pil(mask)

    pen = ImageDraw.Draw(as_pil)

    pen_mask = ImageDraw.Draw(as_pil_mask)

    font = load_font(font_file, size=fit_result.font_size)
    text_bounds = np.array(fit_result.wrap.bounds)
    available_space_y = frame_h
    centering_offset_y = (available_space_y - text_bounds[1]) / 2

    for i in range(len(fit_result.wrap.lines)):
        line = fit_result.wrap.lines[i]

        text = (" " if sep is None else sep).join(line.words)

        x1, y1, x2, y2 = font.getbbox(text)

        w = x2 - x1

        centering_offset_x = (frame_w - w) / 2
        x_pos = centering_offset_x
        y_pos = centering_offset_y

        if justification == Justification.Left:
            x_pos = margin
        elif justification == Justification.Right:
            x_pos = (frame_w - w) - margin

        if alignment == Justification.Left:
            y_pos = margin
        elif alignment == Justification.Right:
            y_pos = (frame_h - fit_result.wrap.bounds[1]) - margin

        y_pos += line.offset

        pen.text(
            (
                x_pos,
                y_pos,
            ),
            str(text),
            fill=(*text_color, 255),
            font=font,
            stroke_width=outline_size,
            stroke_fill=((*outline_color, 255) if outline_size > 0 else None),
        )

        # text mask for compositing
        pen_mask.text(
            (
                x_pos,
                y_pos,
            ),
            str(text),
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=outline_size + 2,  # 2 pixel for segmentation
            stroke_fill=(255, 255, 255, 255),
        )

    dest = pil_to_cv2(as_pil)
    mask = pil_to_cv2(as_pil_mask)
    _, thresh = cv2.threshold(
        cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),
        contours_min_thresh,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return Sample(dest, mask, contours)
