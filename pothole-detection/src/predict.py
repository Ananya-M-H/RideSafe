"""Runs the trained model on a single image.

Usage:
    python src/predict.py path/to/image.jpg
"""
import argparse
from pathlib import Path

import torch

from dataset import CLASSES, eval_transform
from model import PotholeCNN
from train import get_device
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def predict(image_path: str, checkpoint: str) -> None:
    device = get_device()
    model = PotholeCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    pred_idx = probs.argmax().item()
    print(f"Prediction: {CLASSES[pred_idx]} (confidence={probs[pred_idx]:.3f})")
    for cls, p in zip(CLASSES, probs.tolist()):
        print(f"  {cls}: {p:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a road image")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "best_model.pt"))
    args = parser.parse_args()
    predict(args.image, args.checkpoint)