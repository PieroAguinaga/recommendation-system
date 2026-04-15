import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os


class MatrixFactorization(nn.Module):
    """
    Each user and video gets an embedding vector of size EMBEDDING_DIM.
    The predicted score = dot(user_emb, video_emb) + user_bias + video_bias.
    Sigmoid maps it to (0, 1) — interpreted as P(engagement).
    """
    def __init__(self, n_users, n_videos, dim):
        super().__init__()
        self.user_emb   = nn.Embedding(n_users,  dim)
        self.video_emb  = nn.Embedding(n_videos, dim)
        self.user_bias  = nn.Embedding(n_users,  1)
        self.video_bias = nn.Embedding(n_videos, 1)
 
        # Smart initialization: small random values
        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.video_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)
 
    def forward(self, user_idx, video_idx):
        u = self.user_emb(user_idx)           # (batch, dim)
        v = self.video_emb(video_idx)         # (batch, dim)
        dot = (u * v).sum(dim=1, keepdim=True)  # (batch, 1)
        bias = self.user_bias(user_idx) + self.video_bias(video_idx)
        return torch.sigmoid(dot + bias).squeeze()  # (batch,)
 
    def get_user_embedding(self, user_idx):
        return self.user_emb(torch.LongTensor([user_idx]).to(DEVICE))
 
    def get_all_video_embeddings(self):
        all_idx = torch.arange(N_VIDEOS).to(DEVICE)
        return self.video_emb(all_idx)  # (N_VIDEOS, dim)