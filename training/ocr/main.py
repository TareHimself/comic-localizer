import os
import simple_parsing
from dataclasses import dataclass
from ocr.train import train
from typing import Optional


@dataclass
class Config:
    output: str = os.path.join(os.getcwd(), "trained_ocr")  # trained model output path
    device: str = "cuda:0"  # pytorch device to train with
    ckpt: Optional[str] = None
    train: str = os.path.join(
        os.getcwd(), "data_ocr", "train"
    )  # Path to the train dataset folder (folder with image and text pairs)
    eval: str = os.path.join(
        os.getcwd(), "data_ocr", "val"
    )  # Path to the validation dataset folder (folder with image and text pairs)
    batch: int = 32
    half: bool = False
    epochs: int = 8
    lr: float = 1e-5
    test: bool = True
    no_lang: bool = False


# python main.py --test False --batch 64 --half True --no_lang True --epochs 150
# facebook/dino-vits8

if __name__ == "__main__":
    config: Config = simple_parsing.parse(Config)

    train(
        train_dataset_path=config.train,
        eval_dataset_path=config.eval,
        output_path=config.output,
        batch_size=config.batch,
        fp16=config.half,
        num_epochs=config.epochs,
        test=config.test,
        learning_rate=config.lr,
        checkpoint=config.ckpt,
        remove_lang_token=config.no_lang,
    )
