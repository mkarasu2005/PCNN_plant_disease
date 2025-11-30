# utils.py
import random
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, 'best.pth')
        torch.save(state, best_path)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            # model supports env optional
            logits = model(imgs, None)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            preds.extend(pred.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    return {"accuracy": acc, "macro_f1": f1}
