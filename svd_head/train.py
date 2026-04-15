import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from dataset import InteractionDataset, loader, N_USERS, N_VIDEOS
from model import MatrixFactorization

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent
TRAIN_PATH = BASE / "training_data" / "train.csv"
OUT_DIR    = BASE / "training_data"
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
 
# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBEDDING_DIM = 64      # size of each user/video vector
EPOCHS        = 10
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
TOP_K         = 500     # candidates to retrieve per user
 

print("\n Training Matrix Factorization model...")
 
model     = MatrixFactorization(N_USERS, N_VIDEOS, EMBEDDING_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()
 
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for users, videos, labels in loader:
        users, videos, labels = users.to(DEVICE), videos.to(DEVICE), labels.to(DEVICE)
 
        preds = model(users, videos)
        loss  = criterion(preds, labels)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * len(labels)
 
    avg_loss = total_loss / len(train)
    print(f"  Epoch {epoch:02d}/{EPOCHS}  |  loss: {avg_loss:.4f}")
 
# Save model
torch.save({
    "model_state": model.state_dict(),
    "user2idx":    user2idx,
    "video2idx":   video2idx,
    "idx2video":   {i: v for v, i in video2idx.items()},
    "config":      {"dim": EMBEDDING_DIM, "n_users": N_USERS, "n_videos": N_VIDEOS}
}, OUT_DIR / "mf_model.pt")
print(f"\n  Model saved → training_data/mf_model.pt")