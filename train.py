# train.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import make_dataloaders
from model import FusionModel
from utils import seed_everything, save_checkpoint, evaluate_model

def train_main(images_root=r"D:\Final year project PCNN\PlantVillage", epochs=12, batch_size=32, image_size=224, lr=2e-4, out_dir="checkpoints"):
    seed_everything(42)
    train_loader, val_loader, classes = make_dataloaders(images_root=images_root, batch_size=batch_size, image_size=image_size)
    num_classes = len(classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Classes:", num_classes, "Device:", device)
    model = FusionModel(num_classes=num_classes, env_in_dim=0, image_embed_dim=512, transformer_dim=256)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_f1 = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        running_loss = 0.0
        for batch in pbar:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs, None)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)
        val_res = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch} train_loss: {avg_loss:.4f} val_acc: {val_res['accuracy']:.4f} val_f1: {val_res['macro_f1']:.4f}")
        is_best = val_res['macro_f1'] > best_f1
        best_f1 = max(best_f1, val_res['macro_f1'])
        ck = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'classes': classes,
            'best_val_f1': best_f1
        }
        save_checkpoint(ck, is_best, out_dir, filename=f'checkpoint_epoch_{epoch}.pth')
    print("Train finished. Best val f1:", best_f1)

if __name__ == "__main__":
    train_main()
