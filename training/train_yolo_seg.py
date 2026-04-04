from ultralytics import YOLO
import torch
import os
import simple_parsing
from dataclasses import dataclass


@dataclass
class Config:
    dataset: str  # Path to the dataset 'data.yaml'
    output: str = os.path.join(".", "trained_segment.pt")  # trained model output path
    model: str = "yolo11n-seg.yaml"  # The checkpoint to start training from
    device: str = "cuda:0"  # pytorch device to train with
    patience: int = 10
    resume: bool = False


if __name__ == "__main__":
    config: Config = simple_parsing.parse(Config)
    device = torch.device(config.device)
    # Load a pretrained YOLO11n model
    model = YOLO(model=config.model, task="segment")

    results = model.train(
        data=config.dataset,
        patience=config.patience,
        imgsz=640,
        batch=0.8,
        device=device,
        epochs=20000,
        project="Manga Translator Segmentation",
        resume=config.resume,
    )

    model.save(config.output)
