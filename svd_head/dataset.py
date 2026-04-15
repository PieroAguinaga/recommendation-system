import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
 
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent
TRAIN_PATH = BASE / "training_data" / "train.csv"
OUT_DIR    = BASE / "training_data"
 
BATCH_SIZE    = 8   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


print("\n Loading training data...")
train = pd.read_csv(TRAIN_PATH, usecols=["user_id", "video_id", "label"])
print(f"  Interactions: {len(train):,}")
print(f"  Positive rate: {train['label'].mean()*100:.1f}%")
 
# Re-index user_id and video_id to 0..N (required for embedding layers)
user_ids  = train["user_id"].unique()
video_ids = train["video_id"].unique()
 
user2idx  = {u: i for i, u in enumerate(sorted(user_ids))}
video2idx = {v: i for i, v in enumerate(sorted(video_ids))}
 
train["user_idx"]  = train["user_id"].map(user2idx)
train["video_idx"] = train["video_id"].map(video2idx)
 
N_USERS  = len(user2idx)
N_VIDEOS = len(video2idx)
print(f"  Users: {N_USERS:,}  |  Videos: {N_VIDEOS:,}")
 


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users  = torch.LongTensor(df["user_idx"].values)
        self.videos = torch.LongTensor(df["video_idx"].values)
        self.labels = torch.FloatTensor(df["label"].values)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        return self.users[idx], self.videos[idx], self.labels[idx]


loader = DataLoader(
    InteractionDataset(train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
