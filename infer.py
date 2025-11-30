# infer.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from PIL import Image
from predict_utils import load_model, preprocess_image, class_idx_to_name
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--checkpoint", default="checkpoints/best.pth")
args = parser.parse_args()

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(checkpoint_path=args.checkpoint, device=device)
    img = Image.open(args.image).convert("RGB")
    t = preprocess_image(img).unsqueeze(0).to(device)
    logits = model(t, None)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = probs.argmax()
    print("Pred:", class_idx_to_name(pred, classes))
    print("Probs:", probs.tolist())

if __name__ == "__main__":
    run()
